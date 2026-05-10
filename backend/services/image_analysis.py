"""
backend/services/image_analysis.py
====================================
Real image analysis + face/skin validation for SkinAura v2.
- Face presence validation
- Blur detection (Laplacian variance)
- Low-light / overexposure detection
- Lesion coverage (OpenCV adaptive threshold + contours)
- Redness score (HSV analysis)
- Texture irregularity (Laplacian variance)
- Pigmentation clustering (Lab colorspace)
- Attention hotspot density from Grad-CAM heatmap
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ANALYSIS_SIZE = (512, 512)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE QUALITY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_image_quality(image_bytes: bytes) -> dict[str, Any]:
    """
    Gate-check before running inference.
    Rejects blurry, dark, overexposed, or too-small images.

    Returns:
        {"passed": bool, "issues": [str], "metrics": {sharpness, luminance, ...}}
    """
    issues: list[str] = []
    metrics: dict[str, float] = {}

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h    = pil_img.size
        metrics["width"]  = w
        metrics["height"] = h

        if w < 100 or h < 100:
            issues.append(f"Resolution too low ({w}×{h}px). Minimum 100×100.")

        arr  = np.array(pil_img.convert("L"), dtype=np.float64)

        # Sharpness via Laplacian variance
        try:
            import cv2
            lap_var = float(cv2.Laplacian(arr.astype(np.uint8), cv2.CV_64F).var())
        except ImportError:
            gx      = np.diff(arr, axis=1)
            gy      = np.diff(arr, axis=0)
            lap_var = float(np.var(gx) + np.var(gy))

        metrics["sharpness"] = round(lap_var, 2)
        if lap_var < 40:
            issues.append(f"Image too blurry (sharpness={lap_var:.1f}). Please use a clearer photo.")

        # Luminance
        mean_lum = float(arr.mean())
        metrics["mean_luminance"] = round(mean_lum, 2)
        if mean_lum < 35:
            issues.append(f"Image too dark (luminance={mean_lum:.1f}). Please use better lighting.")
        elif mean_lum > 225:
            issues.append(f"Image overexposed (luminance={mean_lum:.1f}). Reduce brightness.")

        # Saturated pixel fraction
        sat_frac = float((arr > 250).mean())
        metrics["saturated_fraction"] = round(sat_frac, 3)
        if sat_frac > 0.30:
            issues.append(f"Excessive highlight clipping ({sat_frac:.0%} saturated pixels).")

    except Exception as exc:
        logger.warning(f"Image quality check failed: {exc}")
        issues.append("Could not verify image quality — proceeding with caution.")

    return {
        "passed":  len(issues) == 0,
        "issues":  issues,
        "metrics": metrics,
    }


def validate_skin_region(image_bytes: bytes) -> dict[str, Any]:
    """
    Lightweight face/skin region validation.
    Uses OpenCV Haar cascade if available, falls back to skin tone HSV check.

    Returns:
        {"face_detected": bool, "skin_detected": bool, "passed": bool, "note": str}
    """
    try:
        import cv2
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr_rgb = np.array(pil_img.resize((400, 400)))
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)

        # Try Haar cascade face detection
        face_detected = False
        try:
            import os
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                detector = cv2.CascadeClassifier(cascade_path)
                faces    = detector.detectMultiScale(gray, scaleFactor=1.1,
                                                      minNeighbors=4, minSize=(60, 60))
                face_detected = len(faces) > 0
        except Exception:
            face_detected = False

        # Skin tone HSV check (fallback + supplement)
        hsv          = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue, sat, val= hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Broad skin HSV range covering fair to dark tones
        skin_mask    = ((hue >= 0) & (hue <= 35) &
                         (sat >= 20) & (sat <= 200) &
                         (val >= 50))
        skin_ratio   = float(skin_mask.mean())
        skin_detected= skin_ratio > 0.10   # >10% skin-tone pixels

        passed = skin_detected
        if face_detected:
            note = "Face region detected successfully."
        elif skin_detected:
            note = f"Skin region identified ({skin_ratio:.0%} skin-tone pixels). No face cascade match."
        else:
            note = "Low skin-region signal. Ensure the image shows a clear face/skin area."

        return {
            "face_detected":  face_detected,
            "skin_detected":  skin_detected,
            "skin_ratio":     round(skin_ratio, 3),
            "passed":         passed,
            "note":           note,
        }

    except ImportError:
        # No cv2 — use PIL-only HSV approximation
        return _skin_check_pil(image_bytes)
    except Exception as exc:
        logger.warning(f"Skin validation failed: {exc}")
        return {"face_detected": False, "skin_detected": True, "passed": True,
                "note": "Validation skipped — proceeding."}


def _skin_check_pil(image_bytes: bytes) -> dict[str, Any]:
    """PIL-only skin detection fallback."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((200, 200))
        arr = np.array(img, dtype=np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        # Simple skin rule: R > G > B, not too dark or bright
        skin = (r > g) & (g > b) & (r > 0.3) & (r < 0.97)
        ratio = float(skin.mean())
        return {
            "face_detected": False,
            "skin_detected": ratio > 0.08,
            "skin_ratio": round(ratio, 3),
            "passed": ratio > 0.08,
            "note": "PIL skin check (no OpenCV).",
        }
    except Exception:
        return {"face_detected": False, "skin_detected": True, "passed": True, "note": "Skipped."}


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE IMAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_image(
    pil_img: Image.Image,
    gradcam_heatmap: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """
    Full quantitative image analysis.

    Args:
        pil_img:         PIL RGB image
        gradcam_heatmap: Raw Grad-CAM (H×W float [0,1]) from gradcam.py

    Returns:
        Dict of real metrics used by severity_engine and insight_engine.
    """
    try:
        import cv2
        return _analyze_cv2(pil_img, gradcam_heatmap, cv2)
    except ImportError:
        return _analyze_numpy(pil_img, gradcam_heatmap)


def _analyze_cv2(pil_img, gradcam_heatmap, cv2) -> dict[str, Any]:
    img_rgb = np.array(pil_img.convert("RGB").resize(ANALYSIS_SIZE))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_gray= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Redness (HSV) ────────────────────────────────────────────────────
    hue, sat, val = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    red_lo = (hue < 15)  & (sat > 55) & (val > 55)
    red_hi = (hue > 160) & (sat > 55) & (val > 55)
    redness_score = float((red_lo | red_hi).mean())

    # ── Lesion coverage (adaptive threshold on a* channel) ───────────────
    a_chan  = np.clip(img_lab[:, :, 1], 0, 255).astype(np.uint8)
    a_thresh= cv2.adaptiveThreshold(a_chan, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, -5)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(a_thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel)
    lesion_coverage = float(np.clip(cleaned.mean() / 255.0, 0.0, 1.0))

    # ── Texture (Laplacian variance) ──────────────────────────────────────
    lap          = cv2.Laplacian(img_gray, cv2.CV_64F)
    texture_norm = float(np.clip(lap.var() / 3000.0, 0.0, 1.0))

    # ── Pigmentation (saturation std dev) ────────────────────────────────
    pigment_score = float(np.clip(sat.std() / 60.0, 0.0, 1.0))

    # ── Lesion count (contours) ───────────────────────────────────────────
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_count = len([c for c in contours if cv2.contourArea(c) > 50])

    # ── Hotspot density (Grad-CAM) ────────────────────────────────────────
    hotspot_density, hotspot_ratio = _hotspot_metrics(gradcam_heatmap, ANALYSIS_SIZE)

    # ── Evenness (L* std dev) ─────────────────────────────────────────────
    l_std         = float(img_lab[:, :, 0].std())
    evenness_score= float(np.clip(1.0 - l_std / 50.0, 0.0, 1.0))

    return {
        "redness_score":    round(redness_score,    4),
        "lesion_coverage":  round(lesion_coverage,  4),
        "texture_score":    round(texture_norm,     4),
        "pigment_score":    round(pigment_score,     4),
        "lesion_count":     lesion_count,
        "hotspot_density":  round(hotspot_density,  4),
        "hotspot_ratio":    round(hotspot_ratio,     4),
        "evenness_score":   round(evenness_score,    4),
        "using_opencv":     True,
    }


def _analyze_numpy(pil_img, gradcam_heatmap) -> dict[str, Any]:
    arr   = np.array(pil_img.convert("RGB").resize(ANALYSIS_SIZE)).astype(np.float32)
    r, g, b = arr[:, :, 0] / 255, arr[:, :, 1] / 255, arr[:, :, 2] / 255
    light = 0.299 * r + 0.587 * g + 0.114 * b

    redness_score   = float(((r > 0.55) & (r - g > 0.10) & (r - b > 0.10)).mean())
    anomaly         = ((r - g > 0.08) | (light < 0.35)) & (light > 0.10)
    lesion_coverage = float(np.clip(anomaly.mean(), 0, 1))
    gx              = np.diff(light, axis=1)
    gy              = np.diff(light, axis=0)
    texture_norm    = float(np.clip((gx.var() + gy.var()) / 0.05, 0, 1))
    pigment_score   = float(np.clip(np.std([r.std(), g.std(), b.std()]) / 0.2, 0, 1))
    evenness_score  = float(np.clip(1 - light.std() / 0.3, 0, 1))

    hotspot_density, hotspot_ratio = _hotspot_metrics(gradcam_heatmap, ANALYSIS_SIZE)

    return {
        "redness_score":   round(redness_score,   4),
        "lesion_coverage": round(lesion_coverage, 4),
        "texture_score":   round(texture_norm,    4),
        "pigment_score":   round(pigment_score,    4),
        "lesion_count":    -1,
        "hotspot_density": round(hotspot_density, 4),
        "hotspot_ratio":   round(hotspot_ratio,    4),
        "evenness_score":  round(evenness_score,   4),
        "using_opencv":    False,
    }


def _hotspot_metrics(
    gradcam_heatmap: Optional[np.ndarray],
    target_size: tuple,
) -> tuple[float, float]:
    if gradcam_heatmap is None:
        return 0.0, 0.0
    try:
        hm = np.array(
            Image.fromarray((gradcam_heatmap * 255).astype(np.uint8)).resize(
                target_size, Image.BILINEAR
            )
        ).astype(np.float32) / 255.0
        hm_max = hm.max()
        if hm_max > 0:
            hm = hm / hm_max
        threshold   = float(np.percentile(hm, 75))
        hotspot_mask = hm > threshold
        density = float(hm[hotspot_mask].mean()) if hotspot_mask.any() else 0.0
        ratio   = float(hotspot_mask.mean())
        return density, ratio
    except Exception:
        return 0.0, 0.0