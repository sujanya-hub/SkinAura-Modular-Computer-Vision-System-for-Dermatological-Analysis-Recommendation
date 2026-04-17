"""
backend/utils/image_utils.py
=============================
Image I/O, validation, and format-conversion utilities for the SkinAura
CV pipeline.

Design
------
- Every public function is stateless and side-effect-free except
  :func:`save_upload`, which writes a temp file and returns its path.
- Functions that can fail on bad input raise
  :class:`~backend.core.exceptions.ImageProcessingError` so the API layer
  receives a domain exception with a meaningful HTTP status code (422).
- OpenCV (``cv2``) is imported lazily inside functions that need it.
  This keeps the module importable in test environments where OpenCV may
  not be installed, and avoids paying the import cost at startup if only
  PIL-based helpers are used.
- The public surface is exactly what the existing callers require:

  Routes (API layer)
      :func:`validate_upload`, :func:`save_upload`

  Preprocessing service
      :func:`load_bgr`, :func:`bgr_to_rgb`, :func:`normalize_to_float32`

  CV service
      :func:`load_bgr`

  Additional helpers are provided for completeness and reuse in future
  services (e.g. thumbnail generation, base64 encoding for debugging).
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.core.config import get_settings
from backend.core.exceptions import ImageProcessingError
from backend.core.logger import get_logger

logger   = get_logger(__name__)
settings = get_settings()

# Type alias used in load_bgr to document the accepted source types.
ImageSource = Union[str, Path, bytes]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_upload(data: bytes, *, content_type: str | None = None) -> None:
    """
    Validate raw upload bytes before any processing occurs.

    Raises:
        :class:`~backend.core.exceptions.ImageProcessingError`:
            - If the payload exceeds ``settings.max_file_size_bytes``.
            - If *content_type* is provided and not in
              ``settings.allowed_content_types``.
            - If Pillow cannot identify the data as a valid image.

    Args:
        data:         Raw bytes from the uploaded file.
        content_type: MIME type declared by the client, e.g.
                      ``"image/jpeg"``.  Optional — when provided it is
                      checked against the configured allow-list before
                      attempting to decode.
    """
    if len(data) > settings.max_file_size_bytes:
        raise ImageProcessingError(
            f"Uploaded file exceeds the {settings.max_file_size_mb} MB limit.",
            details={
                "size_bytes":    len(data),
                "limit_bytes":   settings.max_file_size_bytes,
            },
        )

    if content_type is not None and content_type not in settings.allowed_content_types:
        raise ImageProcessingError(
            f"Content type {content_type!r} is not accepted.",
            details={
                "content_type": content_type,
                "allowed":      settings.allowed_content_types,
            },
        )

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # Raises for corrupt files without fully decoding them.
    except UnidentifiedImageError as exc:
        raise ImageProcessingError(
            "Cannot identify the uploaded file as a valid image.",
            details={"content_type": content_type},
        ) from exc
    except Exception as exc:
        raise ImageProcessingError(
            f"Image validation failed: {exc}",
            details={"content_type": content_type},
        ) from exc


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------


def save_upload(data: bytes, suffix: str = ".jpg") -> Path:
    """
    Persist raw image bytes to ``settings.upload_dir`` and return the path.

    The directory is created on first call if it does not exist.  The
    filename is a UUID4 hex string to prevent collisions across concurrent
    requests.

    Args:
        data:   Raw image bytes to persist.
        suffix: File extension including the leading dot.  Defaults to
                ``".jpg"`` regardless of the original MIME type since the
                file is consumed as a byte stream by OpenCV.

    Returns:
        :class:`~pathlib.Path` to the saved file.

    Raises:
        :class:`~backend.core.exceptions.ImageProcessingError`:
            If the upload directory cannot be created or the file cannot
            be written.
    """
    upload_dir: Path = settings.upload_dir
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ImageProcessingError(
            f"Cannot create upload directory: {upload_dir}",
            details={"path": str(upload_dir), "error": str(exc)},
        ) from exc

    path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        path.write_bytes(data)
    except OSError as exc:
        raise ImageProcessingError(
            f"Failed to write upload to disk: {path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc

    logger.debug("Saved upload → %s (%d bytes).", path, len(data))
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_bgr(source: ImageSource) -> np.ndarray:
    """
    Load an image as a BGR ``uint8`` NumPy array via OpenCV.

    This is the entry-point for the CV pipeline.  OpenCV uses BGR channel
    ordering by convention; callers that need RGB must pass the result
    through :func:`bgr_to_rgb`.

    Args:
        source: One of:
                - An absolute or relative file path (``str`` or
                  :class:`~pathlib.Path`).
                - Raw image bytes (decoded in-memory without writing to disk).

    Returns:
        NumPy array of shape ``(H, W, 3)``, dtype ``uint8``, BGR channel order.

    Raises:
        :class:`~backend.core.exceptions.ImageProcessingError`:
            If OpenCV cannot read or decode the provided source.
    """
    import cv2  # noqa: PLC0415 — lazy import

    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise ImageProcessingError(
                f"OpenCV could not read image file: {source}",
                details={"path": str(source)},
            )
        return img

    # Bytes path: decode in-memory without touching the filesystem.
    arr = np.frombuffer(source, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageProcessingError(
            "OpenCV could not decode the provided image bytes.  "
            "The data may be corrupt or in an unsupported format.",
        )
    return img


def bytes_to_pil(data: bytes) -> Image.Image:
    """
    Decode raw bytes into an RGB :class:`~PIL.Image.Image`.

    Args:
        data: Raw image bytes (JPEG, PNG, WEBP, etc.).

    Returns:
        An RGB PIL Image.

    Raises:
        :class:`~backend.core.exceptions.ImageProcessingError`:
            If Pillow cannot decode the bytes.
    """
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ImageProcessingError(
            f"Pillow could not decode image bytes: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def bgr_to_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Flip channel order from BGR (OpenCV) to RGB.

    Args:
        arr: NumPy array of shape ``(H, W, 3)``, dtype ``uint8``, BGR order.

    Returns:
        New NumPy array of the same shape and dtype with RGB channel order.
        Returns a contiguous copy — does not modify *arr* in place.
    """
    return arr[:, :, ::-1].copy()


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a ``uint8`` NumPy array ``(H, W, C)``.

    The image is converted to RGB before conversion if it is not already.
    """
    return np.array(img.convert("RGB"), dtype=np.uint8)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert a ``uint8`` NumPy array ``(H, W, 3)`` to a PIL Image.

    Args:
        arr: RGB uint8 array (H, W, 3).

    Returns:
        PIL Image in mode "RGB".
    """
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_to_float32(arr: np.ndarray) -> np.ndarray:
    """
    Scale a ``uint8`` pixel array from ``[0, 255]`` to ``float32 [0.0, 1.0]``.

    This is the final pre-inference normalisation step applied to the
    resized face crop before converting to a PyTorch tensor.

    Args:
        arr: uint8 NumPy array of any shape.

    Returns:
        float32 NumPy array of the same shape with values in ``[0.0, 1.0]``.
    """
    return arr.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------


def resize_pil(
    img: Image.Image,
    size: tuple[int, int] | None = None,
) -> Image.Image:
    """
    Resize a PIL Image to *size* ``(width, height)`` using Lanczos resampling.

    Args:
        img:  Source PIL Image.
        size: Target ``(width, height)`` in pixels.  Defaults to
              ``settings.target_image_size`` when omitted.

    Returns:
        Resized PIL Image (new object; source is not modified).
    """
    target = size or settings.target_image_size
    return img.resize(target, Image.LANCZOS)


def resize_bgr(
    arr: np.ndarray,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Resize a BGR NumPy array to *size* ``(width, height)`` using OpenCV.

    Args:
        arr:  Source BGR uint8 array ``(H, W, 3)``.
        size: Target ``(width, height)`` in pixels.  Defaults to
              ``settings.target_image_size`` when omitted.

    Returns:
        Resized BGR uint8 array ``(target_H, target_W, 3)``.
    """
    import cv2  # noqa: PLC0415

    w, h = size or settings.target_image_size
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)


__all__: list[str] = [
    # Validation & persistence
    "validate_upload",
    "save_upload",
    # Loading
    "load_bgr",
    "bytes_to_pil",
    # Format conversion
    "bgr_to_rgb",
    "pil_to_numpy",
    "numpy_to_pil",
    # Normalisation
    "normalize_to_float32",
    # Resizing
    "resize_pil",
    "resize_bgr",
]