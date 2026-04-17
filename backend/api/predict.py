"""
backend/api/predict.py
Safe TensorFlow prediction endpoint.
Model missing / corrupt → demo-mode stub response, never a 500 crash.
"""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

CLASS_NAMES = ["acne", "acne_scars", "normal", "pigmentation"]
RECOMMENDATIONS = {
    "acne":         "Use a gentle cleanser and consider salicylic acid products.",
    "acne_scars":   "Focus on sunscreen, barrier repair, and discuss retinoids with a dermatologist.",
    "pigmentation": "Use daily broad-spectrum sunscreen and consider niacinamide or vitamin C.",
    "normal":       "Maintain a simple routine: cleanser, moisturizer, and sunscreen.",
}
MODEL_CANDIDATES = [
    Path("models/skin_issue_model.keras"),
    Path("models/skin_issue_model.h5"),
]

# ---------------------------------------------------------------------------
# Lazy model loader — runs once, never crashes the process
# ---------------------------------------------------------------------------

_model     = None   # tf.keras.Model | None
_demo_mode = False  # True  → no model available, return safe stub


def _try_load() -> None:
    """Attempt to load Keras model. Sets module-level _model / _demo_mode."""
    global _model, _demo_mode
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            try:
                import tensorflow as tf  # deferred so missing TF ≠ import error
                _model     = tf.keras.models.load_model(str(candidate))
                _demo_mode = False
                logger.info("Keras model loaded from %s.", candidate)
                return
            except Exception as exc:
                logger.warning("Could not load model from %s: %s", candidate, exc)
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


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    img   = img.resize((224, 224))
    array = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    cfg = _get_settings()

    # Content-type guard (skip if settings unavailable)
    if cfg and image.content_type not in cfg.allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {image.content_type}",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    if cfg and len(image_bytes) > cfg.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds max size of {cfg.max_file_size_mb} MB.",
        )

    # Demo mode — safe stub so the endpoint stays alive
    if _demo_mode or _model is None:
        logger.info("Demo mode active — returning stub prediction.")
        return {
            "prediction":    "normal",
            "confidence":    1.0,
            "recommendation": RECOMMENDATIONS["normal"],
            "demo_mode":     True,
            "warning":       "Model unavailable; this is a placeholder result.",
        }

    try:
        inputs      = preprocess_image(image_bytes)
        predictions = _model.predict(inputs, verbose=0)[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    idx        = int(np.argmax(predictions))
    prediction = CLASS_NAMES[idx]
    confidence = float(predictions[idx])

    return {
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        "recommendation": RECOMMENDATIONS.get(
            prediction,
            "Follow a simple cleanser, moisturizer, and sunscreen routine.",
        ),
        "demo_mode": False,
    }


@router.get("/health")
def health():
    return {"status": "ok", "demo_mode": _demo_mode}