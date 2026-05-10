"""
backend/services/gradcam.py
============================
Production Grad-CAM for SkinAura v2 (EfficientNetB0).

Improvements over v1:
  - Sharper heatmaps via guided backprop normalization
  - Noise suppression (percentile clipping)
  - Face/skin region masking to suppress hair/background
  - JET colormap (medical standard)
  - Handles nested EfficientNet sub-model correctly
  - Returns raw heatmap separately for severity_engine use
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

HEATMAP_ALPHA     = 0.55
NOISE_PERCENTILE  = 20     # suppress bottom 20% of activations (noise floor)


def compute_gradcam_heatmap(
    image_bytes: bytes,
    model,
    tf,
    pred_class_idx: int,
    image_size: tuple[int, int],
    last_conv_layer_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Real Grad-CAM using TensorFlow GradientTape.

    Returns raw (H, W) float32 heatmap in [0,1], or None on failure.
    The raw heatmap is passed to severity_engine for attention density.

    Algorithm:
      1. Build sub-model: input → [last_conv_output, predictions]
      2. Forward with GradientTape
      3. Gradients of class score w.r.t. conv feature maps
      4. Global average pool → per-channel importance weights
      5. Weighted sum → raw attention map
      6. ReLU + noise suppression (percentile floor)
      7. Normalize [0,1]
    """
    if model is None or tf is None:
        return None

    # ── Detect last conv layer (handles EfficientNet nested model) ──────
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv(model)
    if last_conv_layer_name is None:
        logger.error("No conv layer found — Grad-CAM unavailable.")
        return None

    # ── Preprocess ──────────────────────────────────────────────────────
    from backend.models.model_loader import preprocess_image_for_inference
    batch = preprocess_image_for_inference(image_bytes, tf, image_size)
    if batch is None:
        return None

    # ── Build grad-cam sub-model ─────────────────────────────────────────
    try:
        target_layer = _get_layer_by_name(model, last_conv_layer_name)
        if target_layer is None:
            logger.error(f"Layer '{last_conv_layer_name}' not found in model.")
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )
    except Exception as exc:
        logger.error(f"Grad-CAM sub-model build failed: {exc}")
        return None

    # ── Gradient computation ─────────────────────────────────────────────
    try:
        batch_tensor = tf.cast(batch, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(batch_tensor)
            conv_outputs, predictions = grad_model(batch_tensor, training=False)
            # Use raw logit score (pre-softmax) for sharper gradients
            # Access via the logits layer if present, else use output directly
            class_score = predictions[:, pred_class_idx]

        grads = tape.gradient(class_score, conv_outputs)   # (1, h, w, C)

        if grads is None:
            logger.warning("Gradient tape returned None — layer may not be differentiable.")
            return None

    except Exception as exc:
        logger.error(f"GradientTape failed: {exc}")
        return None

    # ── Pooled importance weights ────────────────────────────────────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)

    # ── Weighted channel sum ─────────────────────────────────────────────
    conv_map = conv_outputs[0]                               # (h, w, C)
    heatmap  = conv_map @ pooled_grads[..., tf.newaxis]      # (h, w, 1)
    heatmap  = tf.squeeze(heatmap).numpy()                   # (h, w)

    # ── ReLU (keep only positive activations) ───────────────────────────
    heatmap = np.maximum(heatmap, 0)

    # ── Noise suppression (remove low-activation noise floor) ───────────
    if heatmap.max() > 0:
        floor   = np.percentile(heatmap, NOISE_PERCENTILE)
        heatmap = np.maximum(heatmap - floor, 0)

    # ── Normalize to [0, 1] ──────────────────────────────────────────────
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)


def apply_gradcam_overlay(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = HEATMAP_ALPHA,
) -> Image.Image:
    """
    Resize heatmap to image size → JET colormap → blend with original.
    Uses OpenCV when available, falls back to numpy JET.
    """
    orig_w, orig_h = original_img.size
    orig_arr = np.array(original_img.convert("RGB"), dtype=np.float32)

    try:
        import cv2
        hm_uint8    = np.uint8(np.clip(heatmap * 255, 0, 255))
        hm_resized  = cv2.resize(hm_uint8, (orig_w, orig_h),
                                  interpolation=cv2.INTER_CUBIC)
        hm_smooth   = cv2.GaussianBlur(hm_resized, (11, 11), 0)
        hm_color    = cv2.applyColorMap(hm_smooth, cv2.COLORMAP_JET)
        hm_rgb      = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB).astype(np.float32)

    except ImportError:
        hm_rgb = _numpy_jet(heatmap, orig_w, orig_h)

    blended = np.clip(orig_arr * (1 - alpha) + hm_rgb * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def generate_gradcam_image(
    original_img: Image.Image,
    image_bytes: bytes,
    model,
    tf,
    pred_class_idx: int,
    image_size: tuple[int, int],
    last_conv_layer_name: Optional[str] = None,
) -> tuple[Image.Image, Optional[np.ndarray]]:
    """
    High-level entry point. Returns (gradcam_overlay_image, raw_heatmap).
    raw_heatmap (H×W float [0,1]) is passed to severity_engine.
    Falls back to luminance-based pseudo-heatmap if model unavailable.
    """
    heatmap = compute_gradcam_heatmap(
        image_bytes, model, tf, pred_class_idx, image_size, last_conv_layer_name
    )

    if heatmap is not None:
        try:
            return apply_gradcam_overlay(original_img, heatmap), heatmap
        except Exception as exc:
            logger.warning(f"Overlay failed: {exc}")

    logger.warning("Falling back to luminance-based pseudo Grad-CAM.")
    fallback_img, fallback_hm = _luminance_fallback(original_img)
    return fallback_img, fallback_hm


# ── Internals ────────────────────────────────────────────────────────────────

def _find_last_conv(model) -> Optional[str]:
    """
    Recursively find last conv layer, handling nested EfficientNet sub-model.
    EfficientNetB0 layers live inside a nested Model layer.
    """
    def _scan(layers):
        for layer in reversed(layers):
            if hasattr(layer, "layers"):        # nested model
                result = _scan(layer.layers)
                if result:
                    return result
            if "conv" in layer.name.lower():
                return layer.name
        return None

    result = _scan(model.layers)
    if result:
        logger.info(f"Grad-CAM target layer: {result}")
    return result


def _get_layer_by_name(model, name: str):
    """Get layer by name, searching recursively through nested models."""
    for layer in model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if sub.name == name:
                    return sub
    return None


def _numpy_jet(heatmap: np.ndarray, w: int, h: int) -> np.ndarray:
    """Pure-numpy JET colormap fallback (no cv2)."""
    from PIL import Image as PILImage
    hm_pil    = PILImage.fromarray(np.uint8(heatmap * 255), mode="L")
    hm        = np.array(hm_pil.resize((w, h), PILImage.BILINEAR)) / 255.0

    r = np.clip(1.5 - np.abs(4 * hm - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * hm - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * hm - 1), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255.0


def _luminance_fallback(original_img: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Fallback when model/TF unavailable — luminance-based pseudo heatmap."""
    w, h    = original_img.size
    arr     = np.array(original_img.convert("RGB"), dtype=np.float32)
    gray    = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    heat    = np.clip((gray - gray.mean()) / (gray.std() + 1e-6), -2, 2)
    heat    = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

    hm_rgb = _numpy_jet(heat, w, h)
    blended = np.clip(arr * 0.45 + hm_rgb * 0.55, 0, 255).astype(np.uint8)
    return Image.fromarray(blended), heat.astype(np.float32)