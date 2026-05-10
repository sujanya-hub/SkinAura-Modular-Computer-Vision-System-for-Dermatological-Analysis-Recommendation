"""
backend/api/predict.py
Safe TensorFlow prediction endpoint.
Model missing / corrupt → demo-mode stub response, never a 500 crash.
"""
from __future__ import annotations

import logging
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

# Class names and recommendations are loaded dynamically from model_loader.
# The RECOMMENDATIONS dict is keyed by the 8 current class names.
RECOMMENDATIONS = {
    "dark_spots":                        "Use daily broad-spectrum SPF 50+, vitamin C serum, and consider tranexamic acid or kojic acid products.",
    "inflammatory_acne":                 "Use a benzoyl peroxide cleanser, non-comedogenic moisturiser, and consult a dermatologist about topical retinoids.",
    "non_inflammatory_acne_blackheads":  "Incorporate salicylic acid exfoliation 2-3×/week and keep pores clear with a gentle cleanser.",
    "non_inflammatory_acne_whiteheads":  "Use salicylic acid or niacinamide treatments and avoid heavy occlusive products.",
    "pigmentation":                      "Apply broad-spectrum SPF 50+ daily, use niacinamide or alpha-arbutin serums, and consider an AHA exfoliant.",
    "pores":                             "Double-cleanse, use niacinamide to minimise appearance, and exfoliate with BHAs weekly.",
    "redness":                           "Use a gentle fragrance-free cleanser, centella asiatica or azelaic acid serums, and SPF to prevent flares.",
    "wrinkles":                          "Apply a retinoid nightly, use a peptide-rich moisturiser, and wear SPF 50+ every day.",
    # Legacy keys kept for backwards-compat if old model is swapped back in
    "acne":         "Use a gentle cleanser and consider salicylic acid products.",
    "acne_scars":   "Focus on sunscreen, barrier repair, and discuss retinoids with a dermatologist.",
    "normal":       "Maintain a simple routine: cleanser, moisturizer, and sunscreen.",
}

MODEL_CANDIDATES = [
    Path("backend/models/skin_issue_model.keras"),
    Path("backend/models/best_model.keras"),
]


# ---------------------------------------------------------------------------
# Metrics logger — safe, zero-risk import
# ---------------------------------------------------------------------------

try:
    from metrics_logger import log_metric
    _metrics_ok = True
except Exception:
    _metrics_ok = False


def _log(event: str, data: dict) -> None:
    """Thin wrapper — silently does nothing if logger not available."""
    if not _metrics_ok:
        return
    try:
        log_metric(event, data)
    except Exception:
        pass  # metrics NEVER crash the app


# ---------------------------------------------------------------------------
# Lazy model loader — runs once, never crashes the process
# ---------------------------------------------------------------------------

_model      = None          # tf.keras.Model | None
_tf         = None          # tensorflow module | None
_demo_mode  = False         # True → no model available, return safe stub
_input_size = (192, 192)    # updated from model.input_shape after load


def _try_load() -> None:
    """Attempt to load Keras model. Sets module-level _model / _demo_mode."""
    global _model, _tf, _demo_mode, _input_size
    try:
        from backend.models.model_loader import load_model, get_model_input_size
        model, tf = load_model()
        if model is not None:
            _model      = model
            _tf         = tf
            _demo_mode  = False
            # ── KEY FIX: read the true spatial size from the loaded graph ──
            _input_size = get_model_input_size(model)
            logger.info(
                "Keras model loaded successfully via model_loader. "
                "Inference input size: %s", _input_size
            )
            return
    except Exception as exc:
        logger.warning("model_loader.load_model() failed: %s", exc)

    # Direct fallback: try candidates one by one
    try:
        import tensorflow as tf_direct
        tf_direct.get_logger().setLevel("ERROR")
        for candidate in MODEL_CANDIDATES:
            if candidate.exists():
                try:
                    from backend.models.model_loader import (
                        _resolve_preprocess_input,
                        get_model_input_size,
                    )
                    preprocess_input = _resolve_preprocess_input(tf_direct)

                    model = tf_direct.keras.models.load_model(
                        str(candidate),
                        compile=False,
                        custom_objects={"preprocess_input": preprocess_input},
                    )
                    _model      = model
                    _tf         = tf_direct
                    _demo_mode  = False
                    _input_size = get_model_input_size(model)
                    logger.info(
                        "Keras model loaded from %s (direct fallback). "
                        "Inference input size: %s", candidate, _input_size
                    )
                    return
                except Exception as exc:
                    logger.warning("Could not load model from %s: %s", candidate, exc)
    except ImportError:
        pass

    logger.warning(
        "No usable Keras model found at %s — running in DEMO MODE.",
        [str(p) for p in MODEL_CANDIDATES],
    )
    _demo_mode = True


try:
    _try_load()
except Exception as exc:
    logger.warning("Model initialisation failed (%s) — DEMO MODE.", exc)
    _demo_mode = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_settings():
    try:
        from backend.core.config import get_settings
        return get_settings()
    except Exception:
        return None


def _get_class_names() -> list[str]:
    try:
        from backend.models.model_loader import get_class_labels
        return get_class_labels()
    except Exception:
        return ["acne", "acne_scars", "normal", "pigmentation"]


def _to_python(value):
    """
    Recursively convert numpy scalars / arrays to native Python types so
    FastAPI's JSON serialiser never encounters an unserializable object.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def preprocess_image_pil(
    image_bytes: bytes,
    target_size: tuple[int, int] = (192, 192),
) -> np.ndarray:
    """
    PIL-based preprocessor used as fallback when TF preprocess is unavailable.

    Parameters
    ----------
    image_bytes  : raw image bytes (JPEG / PNG / etc.)
    target_size  : (H, W) — must match the model's expected spatial dims.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    img   = img.resize((target_size[1], target_size[0]))   # PIL: (W, H)
    array = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    cfg = _get_settings()
    logger.info(
        "Prediction request received: filename=%s content_type=%s",
        image.filename,
        image.content_type,
    )

    # Content-type guard (skip if settings unavailable)
    if cfg and image.content_type not in cfg.allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {image.content_type}",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")
    logger.info("Prediction upload size: %s bytes", len(image_bytes))

    if cfg and len(image_bytes) > cfg.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds max size of {cfg.max_file_size_mb} MB.",
        )

    class_names = _get_class_names()

    # Demo mode — safe stub so the endpoint stays alive
    if _demo_mode or _model is None:
        logger.info("Demo mode active — returning stub prediction.")
        print("\n========== DEMO MODEL OUTPUT ==========")
        print("Predictions: [demo-mode]")
        print("Predicted Class:", "normal")
        print("Confidence:", 1.0)
        print("Class Mapping:", class_names)
        print("=======================================\n")
        _log("prediction_demo", {"reason": "model_unavailable"})
        return {
            "prediction":      "normal",
            "predicted_class": "normal",
            "confidence":      1.0,
            "recommendation":  RECOMMENDATIONS.get("normal", ""),
            "demo_mode":       True,
            "warning":         "Model unavailable; this is a placeholder result.",
        }

    t_start = time.perf_counter()

    try:
        t_prep = time.perf_counter()

        # ── KEY FIX: always pass the size derived from the loaded model ──────
        # _input_size is set at startup by get_model_input_size(model) and
        # reflects model.input_shape, NOT the env-var default.  This prevents
        # the shape mismatch that caused the original 500 crash.
        inputs: np.ndarray | None = None

        if _tf is not None:
            try:
                from backend.models.model_loader import preprocess_image as tf_preprocess
                inputs = tf_preprocess(image_bytes, _tf, target_size=_input_size)
                if inputs is None:
                    raise ValueError("TF preprocess returned None")
                logger.debug(
                    "TF preprocess output — shape: %s  dtype: %s",
                    inputs.shape, inputs.dtype,
                )
            except Exception as exc:
                logger.warning(
                    "TF preprocess failed (%s) — falling back to PIL.", exc
                )
                inputs = None

        if inputs is None:
            inputs = preprocess_image_pil(image_bytes, target_size=_input_size)
            logger.debug(
                "PIL preprocess output — shape: %s  dtype: %s",
                inputs.shape, inputs.dtype,
            )

        # Sanity-check shape before hitting the model
        expected = (1, _input_size[0], _input_size[1], 3)
        if inputs.shape != expected:
            logger.error(
                "Shape mismatch after preprocessing: got %s, expected %s. "
                "Aborting prediction to avoid a cryptic TF error.",
                inputs.shape, expected,
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Preprocessing produced wrong shape {inputs.shape}; "
                    f"expected {expected}."
                ),
            )

        # Ensure float32 — some PIL paths can produce float64
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)

        prep_ms = round((time.perf_counter() - t_prep) * 1000, 2)

        t_infer = time.perf_counter()
        raw_preds = _model.predict(inputs, verbose=0)
        infer_ms  = round((time.perf_counter() - t_infer) * 1000, 2)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    # ── Safe prediction parsing ───────────────────────────────────────────────
    predictions = raw_preds[0]                        # shape: (num_classes,)
    idx         = int(np.argmax(predictions))         # native int — JSON-safe
    confidence  = float(predictions[idx])             # native float — JSON-safe
    prediction  = (
        class_names[idx] if idx < len(class_names) else "unknown"
    )

    # Full probability map — convert every element to native float
    all_probs = {
        name: round(float(_to_python(predictions[i])), 4)
        for i, name in enumerate(class_names)
        if i < len(predictions)
    }

    print("\n========== RAW MODEL OUTPUT ==========")
    print("Predictions:", predictions)
    print("Predicted Index:", idx)
    print("Predicted Class:", prediction)
    print("Confidence:", confidence)
    print("Class Mapping:", class_names)
    print("All probabilities:", all_probs)
    print("======================================\n")
    logger.info(
        "Prediction completed: predicted_class=%s confidence=%.4f",
        prediction,
        confidence,
    )

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    _log("prediction", {
        "preprocessing_ms": prep_ms,
        "inference_ms":     infer_ms,
        "total_ms":         total_ms,
        "confidence":       round(confidence, 4),
        "predicted_class":  prediction,
        "image_size_kb":    round(len(image_bytes) / 1024, 1),
    })

    return {
        "prediction":      prediction,
        "predicted_class": prediction,
        "confidence":      round(confidence, 4),
        "all_probabilities": all_probs,
        "recommendation":  RECOMMENDATIONS.get(
            prediction,
            "Follow a simple cleanser, moisturizer, and sunscreen routine.",
        ),
        "demo_mode": False,
    }


@router.get("/health")
def health():
    return {
        "status":     "ok",
        "demo_mode":  _demo_mode,
        "input_size": list(_input_size),
    }