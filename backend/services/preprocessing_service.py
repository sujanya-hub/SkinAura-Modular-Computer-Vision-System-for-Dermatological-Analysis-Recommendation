"""
backend/services/preprocessing_service.py
==========================================
Image preprocessing pipeline for the SkinAura CV inference layer.

Responsibilities
----------------
1. Load an image from a file path or raw bytes via :func:`~backend.utils.image_utils.load_bgr`.
2. Detect the largest frontal face using the OpenCV Haar cascade classifier.
3. Crop the detected face region with a configurable margin, or fall back
   to the full image if no face is found (controlled by ``require_face``).
4. Extract skin-coloured pixels from the (pre-crop) image using YCrCb
   thresholding — used by :class:`~backend.services.cv_service.CVService`
   to refine the hex colour estimate for the tone result.
5. Resize the crop to ``settings.target_image_size``, normalise pixels to
   ``float32 [0, 1]``, and return a ``(1, 3, H, W)`` PyTorch tensor ready
   for model inference.

Integration
-----------
- Called exclusively by :class:`~backend.services.cv_service.CVService`.
- Depends on :func:`~backend.utils.image_utils.load_bgr`,
  :func:`~backend.utils.image_utils.bgr_to_rgb`,
  :func:`~backend.utils.image_utils.normalize_to_float32`, and
  :func:`~backend.utils.image_utils.resize_bgr` from the image utilities
  layer.  No image I/O logic is duplicated here.
- All OpenCV imports are lazy so the module can be imported in test
  environments where OpenCV may not be installed.

Output contract
---------------
:meth:`PreprocessingService.preprocess` returns a named tuple-like
three-tuple:

    ``(tensor, face_detected, bounding_box)``

    - ``tensor``:         ``torch.Tensor`` of shape ``(1, 3, H, W)``,
                          dtype ``float32``, values in ``[0.0, 1.0]``.
    - ``face_detected``:  ``bool`` — ``True`` when a face was found and
                          cropped, ``False`` when the full image was used.
    - ``bounding_box``:   ``dict[str, int] | None`` with keys
                          ``x, y, width, height`` in original image pixel
                          coordinates.  ``None`` when no face was detected.

:meth:`PreprocessingService.extract_skin_pixels` returns a
``numpy.ndarray`` of shape ``(N, 3)`` containing the BGR values of pixels
classified as skin, or an empty array when no skin pixels are found.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from backend.core.config import get_settings
from backend.core.exceptions import FaceDetectionError, ImageProcessingError
from backend.core.logger import get_logger
from backend.utils.image_utils import (
    bgr_to_rgb,
    load_bgr,
    normalize_to_float32,
    resize_bgr,
)

logger   = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# BoundingBox type alias — matches the shape consumed by recommendation_service
# and the CVOutput.bounding_box field in responses.py.
# Keys: x (left), y (top), width, height — all in original image pixels.
# ---------------------------------------------------------------------------
BoundingBox = dict[str, int]

# ---------------------------------------------------------------------------
# Cascade classifier — loaded once per process, shared across all requests.
# ---------------------------------------------------------------------------

_cascade_detector: Optional[object] = None  # cv2.CascadeClassifier | None

# Margin as a fraction of the smaller face dimension (10 %).
_FACE_MARGIN_FRACTION: float = 0.10

# detectMultiScale tuning parameters.
_CASCADE_SCALE_FACTOR: float = 1.1
_CASCADE_MIN_NEIGHBORS: int  = 5
_CASCADE_MIN_SIZE: tuple[int, int] = (60, 60)


def _get_cascade() -> object:
    """
    Return the Haar cascade face detector, loading it on first call.

    The classifier XML is bundled with OpenCV and resolved via
    ``cv2.data.haarcascades``.

    Returns:
        A loaded ``cv2.CascadeClassifier`` instance.

    Raises:
        :class:`~backend.core.exceptions.ImageProcessingError`:
            If the cascade XML cannot be located or parsed.
    """
    global _cascade_detector
    if _cascade_detector is not None:
        return _cascade_detector

    import cv2  # noqa: PLC0415

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise ImageProcessingError(
            "Haar cascade face detector could not be loaded.  "
            "Verify that OpenCV is correctly installed.",
            details={"cascade_path": cascade_path},
        )

    _cascade_detector = detector
    logger.debug("Haar cascade loaded from %s.", cascade_path)
    return _cascade_detector


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class PreprocessingService:
    """
    Stateless image preprocessing service for the SkinAura CV pipeline.

    All public methods are thread-safe because they operate only on
    method-local state.  The singleton returned by
    :func:`get_preprocessing_service` is safe to share across concurrent
    requests.
    """

    # ── Face detection ────────────────────────────────────────────────

    def detect_face(
        self,
        bgr: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[BoundingBox]]:
        """
        Detect the largest frontal face in a BGR image.

        The detector uses histogram equalisation on the grayscale channel
        to improve detection under varied lighting conditions.  When
        multiple faces are found the one with the largest area is selected.
        A 10 % margin is added around the detected region and clamped to
        image boundaries to ensure the crop includes eyebrows and chin.

        Args:
            bgr: BGR ``uint8`` NumPy array ``(H, W, 3)`` from OpenCV.

        Returns:
            A two-tuple ``(face_crop, bounding_box)``:

            - ``face_crop``:    BGR ``uint8`` crop array, or ``None``.
            - ``bounding_box``: Dict ``{x, y, width, height}`` in the
                                coordinate space of *bgr*, or ``None``.
        """
        import cv2  # noqa: PLC0415

        detector = _get_cascade()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=_CASCADE_SCALE_FACTOR,
            minNeighbors=_CASCADE_MIN_NEIGHBORS,
            minSize=_CASCADE_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            logger.debug("No faces detected in image of shape %s.", bgr.shape)
            return None, None

        # Select the largest face by pixel area.
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

        # Add a proportional margin and clamp to image boundaries.
        margin = int(_FACE_MARGIN_FRACTION * min(w, h))
        img_h, img_w = bgr.shape[:2]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_w, x + w + margin)
        y2 = min(img_h, y + h + margin)

        face_crop: np.ndarray = bgr[y1:y2, x1:x2]
        bounding_box: BoundingBox = {
            "x":      x1,
            "y":      y1,
            "width":  x2 - x1,
            "height": y2 - y1,
        }

        logger.debug(
            "Face detected: bbox=%s, original_rect=(%d,%d,%d,%d).",
            bounding_box, x, y, w, h,
        )
        return face_crop, bounding_box

    # ── Skin pixel extraction ─────────────────────────────────────────

    def extract_skin_pixels(self, bgr: np.ndarray) -> np.ndarray:
        """
        Extract pixels classified as skin from a BGR image.

        Uses YCrCb colour-space thresholding — a computationally cheap
        and reasonably robust method for diverse skin tones when lighting
        is controlled.  The thresholds (Cr: 133–173, Cb: 77–127) are the
        standard human-skin ranges from the computer vision literature.

        Args:
            bgr: BGR ``uint8`` NumPy array ``(H, W, 3)``.

        Returns:
            NumPy array of shape ``(N, 3)`` containing the BGR values of
            all pixels classified as skin.  Returns an empty array of
            shape ``(0, 3)`` when no skin pixels are found, so callers can
            always safely check ``len(result) > 0`` without handling
            ``None``.
        """
        import cv2  # noqa: PLC0415

        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0,   133, 77],  dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask  = cv2.inRange(ycrcb, lower, upper)

        skin_pixels: np.ndarray = bgr[mask > 0]
        logger.debug(
            "Skin pixel extraction: %d / %d pixels classified as skin.",
            len(skin_pixels), bgr.shape[0] * bgr.shape[1],
        )
        return skin_pixels if len(skin_pixels) > 0 else np.empty((0, 3), dtype=np.uint8)

    # ── Full preprocessing pipeline ───────────────────────────────────

    def preprocess(
        self,
        source: Union[str, Path, bytes],
        *,
        require_face: bool = False,
    ) -> tuple[torch.Tensor, bool, Optional[BoundingBox]]:
        """
        Execute the full preprocessing pipeline for CV inference.

        Pipeline stages
        ~~~~~~~~~~~~~~~
        1. **Load**: decode the source image into a BGR NumPy array.
        2. **Detect**: locate the largest frontal face in the image.
        3. **Crop / fallback**: use the face crop; fall back to the full
           image when no face is detected (unless *require_face* is ``True``).
        4. **Resize**: scale the crop to ``settings.target_image_size``
           (width × height) using area interpolation.
        5. **Normalise**: convert ``uint8 [0, 255]`` → ``float32 [0, 1]``.
        6. **Tensor**: reshape ``(H, W, 3)`` → ``(1, 3, H, W)`` PyTorch
           tensor on CPU.

        Args:
            source:       Image file path (``str`` / :class:`~pathlib.Path`)
                          or raw bytes.
            require_face: When ``True``, raises
                          :class:`~backend.core.exceptions.FaceDetectionError`
                          if no face is found.  When ``False`` (default),
                          the full image is used as a fallback with a
                          warning log.

        Returns:
            Three-tuple ``(tensor, face_detected, bounding_box)``:

            - ``tensor``:         ``(1, 3, H, W)`` float32 CPU tensor.
            - ``face_detected``:  Whether a face region was successfully found.
            - ``bounding_box``:   Face coordinates in original image pixels,
                                  or ``None`` when no face was detected.

        Raises:
            :class:`~backend.core.exceptions.ImageProcessingError`:
                If the image cannot be loaded or decoded.
            :class:`~backend.core.exceptions.FaceDetectionError`:
                If ``require_face=True`` and no face is found.
        """
        # 1. Load.
        bgr: np.ndarray = load_bgr(source)

        # 2 & 3. Detect + crop / fallback.
        face_crop, bounding_box = self.detect_face(bgr)
        face_detected: bool = face_crop is not None

        if not face_detected:
            if require_face:
                raise FaceDetectionError(
                    "No face could be detected in the provided image.  "
                    "Ensure the image is well-lit and the face is clearly visible.",
                    details={"image_shape": list(bgr.shape)},
                )
            logger.warning(
                "No face detected — using full image (%dx%d) as fallback.",
                bgr.shape[1], bgr.shape[0],
            )
            face_crop = bgr

        # 4. Resize (uses settings.target_image_size by default).
        resized: np.ndarray = resize_bgr(face_crop)

        # 5. BGR → RGB → float32 [0, 1].
        rgb:  np.ndarray = bgr_to_rgb(resized)
        norm: np.ndarray = normalize_to_float32(rgb)   # (H, W, 3) float32

        # 6. HWC → CHW → add batch dim → tensor.
        chw:    np.ndarray   = norm.transpose(2, 0, 1)  # (3, H, W)
        tensor: torch.Tensor = torch.from_numpy(chw).unsqueeze(0)  # (1, 3, H, W)

        return tensor, face_detected, bounding_box


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_service: Optional[PreprocessingService] = None


def get_preprocessing_service() -> PreprocessingService:
    """
    Return the application-wide :class:`PreprocessingService` singleton.

    Thread-safe: the singleton is only written once at startup.  After
    that, concurrent requests read the same immutable reference.
    """
    global _service
    if _service is None:
        _service = PreprocessingService()
    return _service


__all__: list[str] = [
    "BoundingBox",
    "PreprocessingService",
    "get_preprocessing_service",
]