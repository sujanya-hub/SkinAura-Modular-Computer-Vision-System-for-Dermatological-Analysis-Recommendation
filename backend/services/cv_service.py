"""
backend/services/cv_service.py
================================
Computer vision inference service for SkinAura.

Responsibilities
----------------
- Orchestrate the end-to-end CV pipeline: image loading → preprocessing →
  skin issue prediction → skin tone/undertone prediction.
- Return a plain ``dict`` whose keys match exactly what
  :class:`~backend.services.recommendation_service.RecommendationService._build_cv_output`
  expects, keeping this layer independent from FastAPI Pydantic models.
- Optionally refine the hex colour estimate for the tone result by sampling
  the median BGR value from skin-coloured pixels detected in the original
  (pre-resize) image.

Output contract
---------------
:meth:`CVService.analyze_skin` always returns a dict with these top-level
keys (consumed verbatim by ``recommendation_service._build_cv_output``):

    ``face_detected``  bool
    ``skin_issues``    List[Dict]   — each item: {name, confidence, severity}
    ``skin_tone``      Dict         — {tone_label, hex_color, undertone, confidence}
    ``bounding_box``   Dict | None  — {x, y, width, height} | None

Severity thresholds
-------------------
Derived from the softmax confidence of the skin-issue classifier:

    confidence ≥ 0.70  →  "severe"
    confidence ≥ 0.40  →  "moderate"
    confidence  < 0.40  →  "mild"

These are heuristic thresholds tuned for the placeholder model; they
will be re-calibrated once real trained weights are available.

Hex colour refinement
---------------------
When skin pixels are successfully extracted from the original image,
:meth:`CVService._median_hex` computes a perceptual median BGR value and
converts it to ``#RRGGBB`` format.  The result is guaranteed to satisfy
the ``pattern=r"^#[0-9A-Fa-f]{6}$"`` constraint on
:class:`~backend.schemas.responses.SkinToneResult`.  If skin pixel
extraction fails (e.g. no skin detected, OpenCV error), the fallback hex
from :attr:`~backend.models.model_loader.SkinToneClassifier.TONE_HEX` is
used instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from backend.core.exceptions import PredictionError
from backend.core.logger import get_logger
from backend.models.model_loader import (
    SkinIssueClassifier,
    SkinToneClassifier,
    get_model_registry,
)
from backend.services.preprocessing_service import get_preprocessing_service
from backend.utils.image_utils import load_bgr

logger = get_logger(__name__)

# Number of top-k skin issues returned by default.
_DEFAULT_TOP_K: int = 3

# Severity confidence boundaries (inclusive lower bound).
_SEVERITY_SEVERE:   float = 0.70
_SEVERITY_MODERATE: float = 0.40


class CVService:
    """
    Computer vision analysis service.

    Stateless per-request — safe to share as a process-wide singleton.
    All mutable state lives inside :class:`~backend.models.model_loader.ModelRegistry`
    and :class:`~backend.services.preprocessing_service.PreprocessingService`.
    """

    def __init__(self) -> None:
        self._registry     = get_model_registry()
        self._preprocessor = get_preprocessing_service()

    # ── Public entry-point ────────────────────────────────────────────

    def analyze_skin(
        self,
        source: Union[str, Path, bytes],
        *,
        top_k: int = _DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Execute the full CV analysis pipeline on an image.

        Loads the image, runs face detection and preprocessing, then
        invokes both the skin-issue and skin-tone classifiers.  The hex
        colour in the tone result is refined from sampled skin pixels when
        possible.

        Args:
            source: Image file path (``str`` / :class:`~pathlib.Path`) or
                    raw bytes.  The same source is decoded twice if skin
                    pixel extraction is needed — this is acceptable because
                    both decodings are fast CPU operations.
            top_k:  Maximum number of skin issues to return, sorted by
                    descending confidence.  Defaults to ``3``.

        Returns:
            A plain ``dict`` with keys:

            .. code-block:: python

                {
                    "face_detected": bool,
                    "skin_issues": [
                        {"name": str, "confidence": float, "severity": str},
                        ...
                    ],
                    "skin_tone": {
                        "tone_label": str,
                        "hex_color":  str,   # "#RRGGBB"
                        "undertone":  str,   # "warm" | "neutral" | "cool"
                        "confidence": float,
                    },
                    "bounding_box": {"x": int, "y": int,
                                     "width": int, "height": int} | None,
                }

        Raises:
            :class:`~backend.core.exceptions.ImageProcessingError`:
                If the image cannot be loaded or decoded.
            :class:`~backend.core.exceptions.PredictionError`:
                If either model inference call fails.
        """
        # ── Step 1: Preprocess (face detect + resize + normalise + tensor) ──
        tensor, face_detected, bounding_box = self._preprocessor.preprocess(
            source, require_face=False
        )

        # ── Step 2: Extract skin pixels for hex colour refinement ──────────
        # This is a best-effort operation; failure is non-fatal.
        skin_pixels: Optional[np.ndarray] = None
        try:
            bgr         = load_bgr(source)
            skin_pixels = self._preprocessor.extract_skin_pixels(bgr)
        except Exception as exc:
            logger.debug(
                "Skin pixel extraction skipped — will use model tone hex: %s", exc
            )

        # ── Step 3: Inference ────────────────────────────────────────────────
        skin_issues = self._predict_skin_issues(tensor, top_k=top_k)
        skin_tone   = self._predict_skin_tone(tensor, skin_pixels=skin_pixels)

        return {
            "face_detected": face_detected,
            "skin_issues":   skin_issues,
            "skin_tone":     skin_tone,
            "bounding_box":  bounding_box,
        }

    # ── Skin issue prediction ─────────────────────────────────────────

    def _predict_skin_issues(
        self,
        tensor: torch.Tensor,
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Run the skin-issue classifier on a preprocessed image tensor.

        Args:
            tensor: ``(1, 3, H, W)`` float32 CPU tensor from preprocessing.
            top_k:  Number of top results to return.

        Returns:
            List of issue dicts ``{name, confidence, severity}``, sorted by
            descending confidence.

        Raises:
            :class:`~backend.core.exceptions.PredictionError`:
                If the forward pass raises an unexpected exception.
        """
        model: SkinIssueClassifier = self._registry.skin_issue_model
        device = self._registry.device

        try:
            tensor_on_device = tensor.to(device)
            with torch.no_grad():
                logits: torch.Tensor = model(tensor_on_device)
                probs:  torch.Tensor = F.softmax(logits, dim=-1).squeeze(0)
        except Exception as exc:
            raise PredictionError(
                f"Skin issue classifier inference failed: {exc}",
                details={"model": "SkinIssueClassifier"},
            ) from exc

        k = min(top_k, probs.shape[0])
        topk_probs, topk_indices = torch.topk(probs, k=k)

        results: List[Dict[str, Any]] = []
        for prob_t, idx_t in zip(topk_probs, topk_indices):
            prob  = float(prob_t.item())
            idx   = int(idx_t.item())
            label = SkinIssueClassifier.LABELS[idx]
            results.append({
                "name":       label,
                "confidence": round(prob, 4),
                "severity":   self._severity_label(prob),
            })

        logger.debug(
            "Skin issues (top %d): %s",
            k,
            [(r["name"], r["confidence"]) for r in results],
        )
        return results

    # ── Skin tone prediction ──────────────────────────────────────────

    def _predict_skin_tone(
        self,
        tensor: torch.Tensor,
        *,
        skin_pixels: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Run the dual-head skin-tone / undertone classifier.

        If *skin_pixels* is provided and non-empty, the hex colour is
        refined from the median pixel value rather than using the static
        palette from :attr:`~backend.models.model_loader.SkinToneClassifier.TONE_HEX`.

        Args:
            tensor:       ``(1, 3, H, W)`` float32 tensor.
            skin_pixels:  Optional ``(N, 3)`` BGR uint8 array of detected
                          skin pixels.  May be ``None`` or empty.

        Returns:
            Dict ``{tone_label, hex_color, undertone, confidence}``.

        Raises:
            :class:`~backend.core.exceptions.PredictionError`:
                If the forward pass raises an unexpected exception.
        """
        model: SkinToneClassifier = self._registry.skin_tone_model
        device = self._registry.device

        try:
            tensor_on_device = tensor.to(device)
            with torch.no_grad():
                tone_logits, undertone_logits = model(tensor_on_device)
                tone_probs:      torch.Tensor = F.softmax(tone_logits,      dim=-1).squeeze(0)
                undertone_probs: torch.Tensor = F.softmax(undertone_logits, dim=-1).squeeze(0)
        except Exception as exc:
            raise PredictionError(
                f"Skin tone classifier inference failed: {exc}",
                details={"model": "SkinToneClassifier"},
            ) from exc

        tone_idx:      int   = int(tone_probs.argmax().item())
        undertone_idx: int   = int(undertone_probs.argmax().item())
        tone_conf:     float = round(float(tone_probs[tone_idx].item()), 4)

        # Use sampled skin pixels for colour when available; fall back to
        # the static Fitzpatrick palette entry otherwise.
        has_pixels = skin_pixels is not None and len(skin_pixels) > 0
        hex_color: str = (
            self._median_hex(skin_pixels)
            if has_pixels
            else SkinToneClassifier.TONE_HEX[tone_idx]
        )

        result = {
            "tone_label": SkinToneClassifier.TONE_LABELS[tone_idx],
            "hex_color":  hex_color,
            "undertone":  SkinToneClassifier.UNDERTONE_LABELS[undertone_idx],
            "confidence": tone_conf,
        }

        logger.debug(
            "Skin tone: %s, undertone: %s, hex: %s, confidence: %.4f.",
            result["tone_label"], result["undertone"],
            result["hex_color"],  result["confidence"],
        )
        return result

    # ── Static helpers ────────────────────────────────────────────────

    @staticmethod
    def _severity_label(confidence: float) -> str:
        """
        Map a classifier confidence score to a severity label.

        Thresholds:
            - ``≥ 0.70``  →  ``"severe"``
            - ``≥ 0.40``  →  ``"moderate"``
            - ``< 0.40``  →  ``"mild"``

        Args:
            confidence: Softmax probability in ``[0.0, 1.0]``.

        Returns:
            One of ``"severe"``, ``"moderate"``, or ``"mild"``.
        """
        if confidence >= _SEVERITY_SEVERE:
            return "severe"
        if confidence >= _SEVERITY_MODERATE:
            return "moderate"
        return "mild"

    @staticmethod
    def _median_hex(bgr_pixels: np.ndarray) -> str:
        """
        Compute a representative skin colour as a ``#RRGGBB`` hex string.

        The channel-wise median of *bgr_pixels* is used rather than the
        mean because it is more robust to outliers (e.g. specular
        highlights, shadows at image edges).

        The output is guaranteed to match the pattern
        ``r"^#[0-9A-Fa-f]{6}$"`` enforced by
        :class:`~backend.schemas.responses.SkinToneResult`.

        Args:
            bgr_pixels: ``(N, 3)`` uint8 array of BGR pixel values.

        Returns:
            Hex string in the format ``"#RRGGBB"``.
        """
        median = np.median(bgr_pixels, axis=0).astype(np.uint8)
        b, g, r = int(median[0]), int(median[1]), int(median[2])
        return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_cv_service: Optional[CVService] = None


def get_cv_service() -> CVService:
    """
    Return the application-wide :class:`CVService` singleton.

    Models are not loaded at this point — they are loaded lazily on the
    first call to :meth:`CVService.analyze_skin`.  To force eager loading
    at startup, access ``get_model_registry().skin_issue_model`` and
    ``get_model_registry().skin_tone_model`` in the lifespan handler.
    """
    global _cv_service
    if _cv_service is None:
        _cv_service = CVService()
    return _cv_service


__all__: list[str] = [
    "CVService",
    "get_cv_service",
]