"""
backend/core/exceptions.py
===========================
Domain-specific exception hierarchy for the SkinAura backend.

Design
------
- All exceptions inherit from :class:`SkinAuraError`, enabling callers to
  catch either the base type or a specific subclass as the situation demands.
- Each exception carries a human-readable ``message`` and an optional
  ``details`` dict for structured context (filenames, shapes, model names,
  etc.) that appears in API error responses and log output.
- ``http_status_code`` is declared as a class attribute on every subclass so
  the FastAPI exception handler in :mod:`backend.utils.response_utils` can
  map directly to an HTTP response without a lookup table.
- :meth:`SkinAuraError.to_dict` produces a plain serialisable dict that maps
  cleanly onto :class:`~backend.schemas.responses.ErrorResponse`.

Usage::

    from backend.core.exceptions import FaceDetectionError

    raise FaceDetectionError(
        "No face region detected in the uploaded image.",
        details={"filename": "photo.jpg", "resolution": [48, 48]},
    )
"""
from __future__ import annotations

from http import HTTPStatus
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------


class SkinAuraError(Exception):
    """
    Root exception for all SkinAura domain errors.

    Attributes:
        message:          Short, human-readable description of the failure.
        details:          Optional structured payload for downstream logging
                          and API error responses.
        http_status_code: HTTP status code returned when this exception
                          propagates to an API exception handler.
    """

    http_status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: Dict[str, Any] = details or {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"details={self.details!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise to a plain dict suitable for JSON error responses.

        The ``details`` key is omitted when empty to keep responses clean.
        """
        payload: Dict[str, Any] = {
            "error":   type(self).__name__,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


# ---------------------------------------------------------------------------
# Configuration & startup
# ---------------------------------------------------------------------------


class ConfigurationError(SkinAuraError):
    """
    Raised when the application cannot start or operate correctly due to a
    missing, invalid, or incompatible configuration value.
    """
    http_status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


class ImageProcessingError(SkinAuraError):
    """
    Raised when an uploaded image cannot be validated, decoded, resized,
    or otherwise processed by the imaging pipeline.

    Typically maps to ``422 Unprocessable Entity`` since the request was
    well-formed but the payload could not be acted upon.
    """
    http_status_code: int = HTTPStatus.UNPROCESSABLE_ENTITY.value


class FaceDetectionError(SkinAuraError):
    """
    Raised when the face-detection step fails to locate a valid facial
    region within the provided image.

    Callers should surface this as a user-actionable error — the image
    may need to be retaken with better framing or lighting.
    """
    http_status_code: int = HTTPStatus.UNPROCESSABLE_ENTITY.value


# ---------------------------------------------------------------------------
# ML model lifecycle
# ---------------------------------------------------------------------------


class ModelLoadError(SkinAuraError):
    """
    Raised when a PyTorch model cannot be loaded from disk.

    Common causes: missing weight file, incompatible state dict, device
    OOM, or corrupt checkpoint.  The service is considered unavailable
    until the issue is resolved.
    """
    http_status_code: int = HTTPStatus.SERVICE_UNAVAILABLE.value


class PredictionError(SkinAuraError):
    """
    Raised when a model forward pass fails at runtime.

    Common causes: unexpected input shape, NaN propagation, or an
    unhandled exception inside a custom layer.
    """
    http_status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


class RAGError(SkinAuraError):
    """
    Raised when the retrieval-augmented generation pipeline encounters an
    unrecoverable error — FAISS index missing or corrupt, embedding model
    failure, or metadata deserialization error.
    """
    http_status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value


# ---------------------------------------------------------------------------
# LLM / Groq
# ---------------------------------------------------------------------------


class LLMServiceError(SkinAuraError):
    """
    Raised when the Groq LLM generation step fails after all retries.

    Common causes: API authentication failure, rate limiting, network
    timeout, or a malformed JSON response from the model.

    Maps to ``502 Bad Gateway`` because the failure is in an upstream
    provider, not in SkinAura's own logic.
    """
    http_status_code: int = HTTPStatus.BAD_GATEWAY.value


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class RecommendationError(SkinAuraError):
    """
    Raised by the recommendation service when the full analysis pipeline
    cannot produce a valid, structured response after all sub-steps have
    been attempted and their individual fallbacks exhausted.
    """
    http_status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class ValidationError(SkinAuraError):
    """
    Raised for domain-level input validation failures that fall outside
    Pydantic's built-in schema validation — for example, an uploaded file
    with a disallowed MIME type, an image with insufficient resolution, or
    a request body that is syntactically valid but semantically nonsensical.
    """
    http_status_code: int = HTTPStatus.BAD_REQUEST.value


# ---------------------------------------------------------------------------
# External services
# ---------------------------------------------------------------------------


class ExternalServiceError(SkinAuraError):
    """
    Raised when a call to any third-party service other than Groq fails —
    for example, a future product-database API or a telemetry sink.

    Maps to ``502 Bad Gateway`` to signal that the failure is upstream.
    """
    http_status_code: int = HTTPStatus.BAD_GATEWAY.value


# ---------------------------------------------------------------------------
# Public re-export list
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "SkinAuraError",
    "ConfigurationError",
    "ImageProcessingError",
    "FaceDetectionError",
    "ModelLoadError",
    "PredictionError",
    "RAGError",
    "LLMServiceError",
    "RecommendationError",
    "ValidationError",
    "ExternalServiceError",
]