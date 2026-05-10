"""
backend/models/model_loader.py
==============================
Singleton model loader with dynamic class-name resolution.

Label-mapping contract
──────────────────────
Class indices are ALWAYS derived from one of three sources, in priority order:

  1. backend/models/class_names.json   ← written by train_model.py at the end
                                          of every training run
  2. Dataset folder scan (alphabetical) ← fallback when the JSON is absent
  3. Hard-coded sentinel list           ← last resort; emits a loud WARNING

This guarantees that what the model learned during training and what inference
maps predictions onto are ALWAYS in sync.

NOTE: The old `CLASS_LABELS` constant and `get_model_registry` / `SkinIssueClassifier`
      have been removed.  All consumers must use get_class_labels() / index_to_label().
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

# CHANGE 1 — requests added for Hugging Face model auto-download.
# Placed alongside the other stdlib / third-party imports; no existing
# import is moved or removed.
import requests

import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.environ.get(
    "SKINAURA_MODEL_PATH",
    "backend/models/skin_issue_model.keras",
))
CLASS_NAMES_PATH = Path(os.environ.get(
    "SKINAURA_CLASS_NAMES_PATH",
    "backend/models/class_names.json",
))
DATASET_DIR = Path(os.environ.get(
    "SKINAURA_DATASET_DIR",
    "datasets/skin_conditions",
))

# CHANGE 2 — Hugging Face model URL.
# Overridable via env-var so deployments can point at a different repo or
# a private mirror without touching code.  Uses /resolve/ (direct download),
# never /blob/ (HTML page).
MODEL_URL: str = os.environ.get(
    "SKINAURA_MODEL_URL",
    "https://huggingface.co/sujanya/SkinAura-model/resolve/main/skin_issue_model.keras",
)

# INPUT_SIZE is now a *default* only.
# preprocess_image() always accepts an explicit size argument so callers
# (e.g. predict.py) can pass the value read directly from model.input_shape,
# guaranteeing the preprocessing and the model are always in sync.
_default_size = int(os.environ.get("SKINAURA_INPUT_SIZE", "192"))
INPUT_SIZE = (_default_size, _default_size)

# ── Internal cache ────────────────────────────────────────────────────────────
_CLASS_LABELS: list[str] = []


# ---------------------------------------------------------------------------
# CHANGE 3 — Hugging Face auto-download
# ---------------------------------------------------------------------------

def download_model_if_missing(
    model_path: Path = MODEL_PATH,
    model_url: str = MODEL_URL,
    chunk_size: int = 8 * 1024 * 1024,   # 8 MB chunks
    timeout: int    = 300,                # 5-minute total timeout
) -> None:
    """
    Download the Keras model from Hugging Face if it is not already present
    on the local filesystem.

    Behaviour
    ---------
    - If ``model_path`` already exists the function returns immediately
      (no network request, no checksum, no lock needed for read-only check).
    - Otherwise the parent directory is created (parents=True, exist_ok=True),
      the file is streamed in ``chunk_size`` chunks, and the total downloaded
      size is logged on completion.
    - Uses ``stream=True`` so the full file is never buffered in RAM.
    - A ``timeout`` guards against hung connections on cold HF servers.

    Parameters
    ----------
    model_path : Path
        Destination path for the .keras file.  Defaults to MODULE-level
        MODEL_PATH so callers never need to pass it explicitly.
    model_url : str
        Direct-download URL (must use /resolve/, not /blob/).
        Defaults to MODULE-level MODEL_URL.
    chunk_size : int
        Bytes per write chunk.  8 MB is a good balance between memory use
        and number of syscalls for a ~100 MB model file.
    timeout : int
        Seconds before the HTTP request is abandoned.  Raise this on very
        slow connections; lower it in unit tests.

    Raises
    ------
    requests.RequestException
        Re-raised after logging so the caller (load_model) can catch it and
        fall through to API-only mode rather than crashing the process.
    """
    if model_path.exists():
        logger.info(f"Model already present at {model_path} — skipping download.")
        return

    logger.info(
        f"Model file not found at '{model_path}'. "
        f"Downloading from Hugging Face: {model_url}"
    )

    # Ensure destination directory exists before opening the file for writing.
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(model_url, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            # Content-Length is optional; log it when available.
            content_length = response.headers.get("Content-Length")
            if content_length:
                mb = int(content_length) / (1024 * 1024)
                logger.info(f"Download size: {mb:.1f} MB — writing to {model_path}")
            else:
                logger.info(f"Download size unknown — writing to {model_path}")

            bytes_written = 0
            with open(model_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive empty chunks
                        fh.write(chunk)
                        bytes_written += len(chunk)

        downloaded_mb = bytes_written / (1024 * 1024)
        logger.info(
            f"Model download complete: {model_path} "
            f"({downloaded_mb:.1f} MB written)."
        )

    except requests.RequestException as exc:
        logger.error(
            f"Failed to download model from {model_url}: {exc}. "
            "Backend will start in API-only / demo mode."
        )
        # Remove a partially-written file so the next startup attempt
        # does not mistake a truncated file for a valid model.
        if model_path.exists():
            try:
                model_path.unlink()
                logger.info(f"Removed partial download at {model_path}.")
            except OSError as rm_exc:
                logger.warning(f"Could not remove partial download: {rm_exc}")
        raise


# ---------------------------------------------------------------------------
# Label resolution  (single source of truth)
# ---------------------------------------------------------------------------

def _load_class_names_from_json(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        names = json.loads(path.read_text())
        if isinstance(names, list) and all(isinstance(n, str) for n in names):
            return [n.lower().strip() for n in names]
    except Exception as exc:
        logger.warning(f"Could not parse {path}: {exc}")
    return None


def _load_class_names_from_dataset(dataset_dir: Path) -> list[str] | None:
    if not dataset_dir.exists():
        return None
    classes = sorted(
        sub.name.lower()
        for sub in dataset_dir.iterdir()
        if sub.is_dir() and not sub.name.startswith(".")
    )
    return classes if classes else None


def _log_label_map(names: list[str], source: str) -> None:
    lines = "\n".join(f"  {i} → {n}" for i, n in enumerate(names))
    logger.info(f"Class labels loaded from {source}:\n{lines}")


def resolve_class_names(
    class_names_path: Path = CLASS_NAMES_PATH,
    dataset_dir: Path = DATASET_DIR,
) -> list[str]:
    """
    Determine class labels in the same alphabetical order Keras uses.

    Priority:
      1. class_names.json  (written by training script — preferred)
      2. Dataset folder scan
      3. Hard-coded fallback  (emits WARNING; fix by running train_model.py)
    """
    names = _load_class_names_from_json(class_names_path)
    if names:
        _log_label_map(names, str(class_names_path))
        return names

    names = _load_class_names_from_dataset(dataset_dir)
    if names:
        logger.warning(
            f"{class_names_path} not found — derived class order from dataset "
            f"folders at '{dataset_dir}'. Re-run train_model.py to generate "
            "class_names.json for a stable, guaranteed-correct mapping."
        )
        _log_label_map(names, f"dataset scan ({dataset_dir})")
        return names

    fallback = [
        "dark_spots",
        "inflammatory_acne",
        "non_inflammatory_acne_blackheads",
        "non_inflammatory_acne_whiteheads",
        "pigmentation",
        "pores",
        "redness",
        "wrinkles",
    ]
    logger.warning(
        " LABEL FALLBACK ACTIVE — could not resolve class names from JSON "
        "or dataset scan. Using hard-coded list. Predictions MAY BE WRONG. "
        "Run train_model.py to fix this permanently."
    )
    _log_label_map(fallback, "hard-coded fallback")
    return fallback


def get_class_labels() -> list[str]:
    """
    Return the globally resolved, cached class labels.

    This is the ONLY function downstream code (predictor, severity engine,
    Grad-CAM, API routes) should call to obtain class names.
    """
    global _CLASS_LABELS
    if not _CLASS_LABELS:
        _CLASS_LABELS = resolve_class_names()
    return _CLASS_LABELS


def index_to_label(index: int) -> str:
    """
    Convert a model output index to its snake_case label.

    Raises IndexError if the index is out of range — which would indicate
    a mismatch between the saved model and class_names.json.
    """
    labels = get_class_labels()
    if 0 <= index < len(labels):
        return labels[index]
    raise IndexError(
        f"Prediction index {index} out of range for {len(labels)} classes. "
        "Ensure class_names.json was produced by the current training run."
    )


# ---------------------------------------------------------------------------
# TensorFlow / model loading
# ---------------------------------------------------------------------------

def _try_load_tensorflow():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        return tf
    except ImportError:
        return None


def _resolve_preprocess_input(tf):
    """
    Return the EfficientNet preprocess_input function — matching the Lambda
    layer serialised inside the .keras graph — then fall through to other
    common backbones, and finally to an identity lambda.

    Crucially this function is passed as a custom_object so Keras can
    deserialise the Lambda(preprocess_input) layer by name.
    """
    # EfficientNet is first because that is what the trained model uses.
    for module_name in (
        "efficientnet",
        "efficientnet_v2",
        "mobilenet_v2",
        "mobilenet",
        "resnet",
        "resnet_v2",
        "vgg16",
        "vgg19",
        "inception_v3",
        "xception",
        "densenet",
        "nasnet",
    ):
        try:
            module = getattr(tf.keras.applications, module_name)
            fn = module.preprocess_input
            logger.info(f"Using preprocess_input from keras.applications.{module_name}")
            return fn
        except Exception:
            continue
    logger.warning(
        "Could not locate a known preprocess_input helper — "
        "falling back to identity function. Predictions may be inaccurate."
    )
    return lambda x: x


def get_model_input_size(model) -> tuple[int, int]:
    """
    Read the spatial input dimensions directly from the loaded model's
    input_shape, e.g. (None, 192, 192, 3) → (192, 192).

    Falls back to the module-level INPUT_SIZE constant if the shape
    cannot be determined (e.g. dynamic/unknown dims).
    """
    try:
        shape = model.input_shape  # (None, H, W, C)
        h, w = int(shape[1]), int(shape[2])
        logger.info(f"Derived input size from model.input_shape: ({h}, {w})")
        return (h, w)
    except Exception as exc:
        logger.warning(
            f"Could not read input size from model.input_shape ({exc}); "
            f"falling back to INPUT_SIZE={INPUT_SIZE}"
        )
        return INPUT_SIZE


def load_model():
    """
    Load the Keras .keras model.
    Returns (model, tf) tuple, or (None, None) if unavailable.

    The Lambda(preprocess_input) layer inside the graph is resolved by
    passing the function as a custom_object so Keras can deserialise it
    by the name 'preprocess_input' without raising.

    Side-effect: eagerly resolves and logs class labels so the mapping
    is visible in startup logs before the first prediction.
    """
    get_class_labels()  # print label map at startup

    # CHANGE 4 — attempt Hugging Face download before checking existence.
    # download_model_if_missing() is a no-op when the file is already present,
    # so this adds zero overhead on subsequent restarts.
    # Any network error is caught inside the function; it logs and re-raises,
    # and we catch it here so a download failure degrades gracefully to
    # API-only mode instead of crashing the process.
    try:
        download_model_if_missing()
    except Exception:
        # Error already logged inside download_model_if_missing().
        # Fall through — the model_path.exists() check below will fail and
        # the function will return (None, None) for API-only / demo mode.
        pass

    tf = _try_load_tensorflow()
    if tf is None:
        logger.warning("TensorFlow not installed — running in API-only mode.")
        return None, None

    model_path = MODEL_PATH  # already a Path
    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path} — running in API-only mode.")
        return None, None

    try:
        preprocess_input = _resolve_preprocess_input(tf)

        # Register the function under the exact name Keras serialised into
        # the Lambda layer config so that from_config() can look it up.
        @tf.keras.utils.register_keras_serializable(package="builtins")
        def preprocess_input_registered(x):  # noqa: F811
            return preprocess_input(x)

        model = tf.keras.models.load_model(
            str(model_path),
            compile=False,
            custom_objects={
                "preprocess_input": preprocess_input,
                "preprocess_input_registered": preprocess_input_registered,
            },
        )

        # Derive the true input size from the model and update the module
        # constant so any code that reads INPUT_SIZE stays in sync.
        global INPUT_SIZE
        INPUT_SIZE = get_model_input_size(model)

        logger.info(
            f"Model loaded: {model_path} | "
            f"Input shape: {model.input_shape} | "
            f"Resolved INPUT_SIZE: {INPUT_SIZE}"
        )

        # Validate output size matches loaded class count
        num_model_outputs = model.output_shape[-1]
        num_labels        = len(get_class_labels())
        if num_model_outputs != num_labels:
            logger.error(
                f"Model output size ({num_model_outputs}) != class count "
                f"({num_labels}). The model and class_names.json are out of sync. "
                "Retrain or restore the correct class_names.json."
            )

        return model, tf
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(
    image_bytes: bytes,
    tf=None,
    target_size: tuple[int, int] | None = None,
) -> Optional[np.ndarray]:
    """
    Preprocess raw image bytes into a normalised numpy batch for inference.

    Parameters
    ----------
    image_bytes : bytes
        Raw JPEG/PNG image data.
    tf : tensorflow module, optional
        Pass the already-imported TF reference to skip a redundant import.
    target_size : (H, W) tuple, optional
        Resize target.  When None the module-level INPUT_SIZE is used.
        Callers that hold a reference to the loaded model should pass
        ``get_model_input_size(model)`` here so the size is always in sync
        with the actual graph, regardless of env-var settings.

    Pipeline
    --------
      1. Decode JPEG/PNG bytes → RGB tensor          (uint8, range [0, 255])
      2. Resize to target_size                        (float, range [0, 255])
      3. Cast to float32                              (float, range [0, 255])
      4. Expand dims → (1, H, W, 3) batch

    CHANGE 5 — preprocessing fix (double-normalisation removed).

    The saved model's first layer is Lambda(efficientnet.preprocess_input).
    EfficientNet's preprocess_input expects raw pixel values in [0, 255] and
    internally scales them to [-1, 1] via ``x / 127.5 - 1.0``.

    The previous code divided by 255.0 BEFORE the model, so the Lambda layer
    received values in [0, 1] and scaled them to approximately [-1, -0.992],
    causing every prediction to be made from near-uniform near-negative input
    — effectively random/garbage output.

    Fix: cast to float32 only.  Do NOT divide by 255.  The Lambda layer
    inside the model graph handles all normalisation exactly as it did
    during training.
    """
    if tf is None:
        tf = _try_load_tensorflow()
    if tf is None:
        logger.error("TensorFlow unavailable — cannot preprocess image.")
        return None

    size = target_size if target_size is not None else INPUT_SIZE

    try:
        img_tensor = tf.image.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        img_tensor = tf.image.resize(img_tensor, size)
        # Cast to float32 only — do NOT divide by 255.
        # The Lambda(preprocess_input) layer inside the model expects raw
        # [0, 255] pixel values and handles normalisation internally.
        # Dividing here causes double-normalisation → garbage predictions.
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        result = img_tensor.numpy()
        logger.debug(
            f"preprocess_image: output shape={result.shape} "
            f"dtype={result.dtype} min={result.min():.3f} max={result.max():.3f}"
        )
        return result
    except Exception as exc:
        logger.error(f"Image preprocessing failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Grad-CAM helper
# ---------------------------------------------------------------------------

def get_last_conv_layer(model) -> Optional[str]:
    """
    Dynamically detect the last convolutional layer name.
    Works for MobileNetV2, EfficientNet, and other common backbones.
    """
    last_conv = None
    for layer in model.layers:
        if hasattr(layer, "filters") or "conv" in layer.name.lower():
            last_conv = layer.name
    return last_conv