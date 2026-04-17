from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.core.config import get_settings
from backend.core.logger import get_logger

router = APIRouter(tags=["Prediction"])
logger = get_logger(__name__)
settings = get_settings()

CLASS_NAMES = ["acne", "acne_scars", "normal", "pigmentation"]
RECOMMENDATIONS = {
    "acne": "Use a gentle cleanser, avoid picking, and consider salicylic acid or benzoyl peroxide products.",
    "acne_scars": "Focus on sunscreen, barrier repair, and discuss retinoids or dermatologist-guided scar treatments.",
    "pigmentation": "Use daily broad-spectrum sunscreen and consider niacinamide, vitamin C, or dermatologist guidance.",
    "normal": "Maintain a simple routine with cleanser, moisturizer, and sunscreen to keep skin balanced.",
}
MODEL_CANDIDATES = [
    Path("models/skin_issue_model.keras"),
    Path("models/skin_issue_model.h5"),
]


def _resolve_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No TensorFlow model file found. Expected one of: "
        + ", ".join(str(path) for path in MODEL_CANDIDATES)
    )


@lru_cache(maxsize=1)
def get_prediction_model() -> tf.keras.Model:
    model_path = _resolve_model_path()
    logger.info("Loading TensorFlow skin issue model from %s", model_path)
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    image = image.resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


@router.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, float | str]:
    if image.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {image.content_type}",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    if len(image_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds max size of {settings.max_file_size_mb} MB.",
        )

    try:
        model = get_prediction_model()
        inputs = preprocess_image(image_bytes)
        predictions = model.predict(inputs, verbose=0)[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    predicted_index = int(np.argmax(predictions))
    prediction = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])
    recommendation = RECOMMENDATIONS.get(
        prediction,
        "Follow a simple cleanser, moisturizer, and sunscreen routine.",
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "recommendation": recommendation,
    }
@router.get("/health")
def health():
    return {"status": "ok"}
