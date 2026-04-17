"""
backend/utils/response_utils.py
================================
Utilities for generating request correlation IDs and constructing
standardised FastAPI JSON error responses.

Design
------
- :func:`generate_request_id` is the single source of truth for the
  short hex ID attached to every request.  It is called by API route
  functions and forwarded to services for log correlation.
- :func:`build_error_response` converts any
  :class:`~backend.core.exceptions.SkinAuraError` into a JSON
  :class:`~fastapi.responses.JSONResponse` whose body matches the
  :class:`~backend.schemas.responses.ErrorResponse` envelope exactly.
  The HTTP status code is taken from ``exc.http_status_code``, which is
  declared per-exception-class in :mod:`backend.core.exceptions`.
- :func:`build_unhandled_error_response` is the fallback for unexpected
  ``Exception`` instances that escape domain exception handlers.  It
  always returns 500 and logs the full traceback.
- These helpers are consumed by:
    - :mod:`backend.main` (global FastAPI exception handlers)
    - :mod:`backend.api.predict`
    - :mod:`backend.api.analyze`
    - :mod:`backend.services.recommendation_service` (request ID only)
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi.responses import JSONResponse

from backend.core.exceptions import SkinAuraError
from backend.core.logger import get_logger
from backend.schemas.responses import ErrorResponse

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request correlation ID
# ---------------------------------------------------------------------------


def generate_request_id() -> str:
    """
    Generate a short, unique request correlation ID.

    Returns a 12-character lowercase hex string derived from a UUID4.
    Short enough to be readable in log lines; collision probability is
    negligible for the request volumes SkinAura will process.

    Returns:
        A 12-character hex string, e.g. ``"a1b2c3d4e5f6"``.
    """
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Domain error responses
# ---------------------------------------------------------------------------


def build_error_response(
    exc: SkinAuraError,
    *,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """
    Convert a :class:`~backend.core.exceptions.SkinAuraError` into a
    FastAPI :class:`~fastapi.responses.JSONResponse` with the appropriate
    HTTP status code.

    The response body is serialised from an
    :class:`~backend.schemas.responses.ErrorResponse` instance, guaranteeing
    that the error envelope is identical for every domain error across the API.

    Args:
        exc:        The domain exception to convert.
        request_id: Optional correlation ID from the originating request.

    Returns:
        A :class:`~fastapi.responses.JSONResponse` whose ``status_code``
        matches ``exc.http_status_code`` and whose body matches the
        :class:`~backend.schemas.responses.ErrorResponse` schema.
    """
    logger.warning(
        "[%s] %s — %s",
        request_id or "—",
        type(exc).__name__,
        exc.message,
        extra={"details": exc.details} if exc.details else {},
    )

    payload = ErrorResponse(
        error=      type(exc).__name__,
        message=    exc.message,
        request_id= request_id,
        details=    exc.details if exc.details else None,
    )

    return JSONResponse(
        status_code=exc.http_status_code,
        content=payload.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Unhandled / unexpected error response
# ---------------------------------------------------------------------------


def build_unhandled_error_response(
    exc: Exception,
    *,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """
    Produce a generic 500 response for exceptions that are not
    :class:`~backend.core.exceptions.SkinAuraError` subclasses.

    The full traceback is logged at ERROR level.  The client receives a
    safe, non-revealing message regardless of the underlying cause.

    Args:
        exc:        The unexpected exception.
        request_id: Optional correlation ID from the originating request.

    Returns:
        A :class:`~fastapi.responses.JSONResponse` with status 500 and
        an :class:`~backend.schemas.responses.ErrorResponse` body.
    """
    logger.exception(
        "[%s] Unhandled exception: %s",
        request_id or "—",
        exc,
    )

    payload = ErrorResponse(
        error=      "InternalServerError",
        message=    "An unexpected error occurred. Please try again.",
        request_id= request_id,
        details=    None,
    )

    return JSONResponse(
        status_code=500,
        content=payload.model_dump(mode="json"),
    )


__all__: list[str] = [
    "generate_request_id",
    "build_error_response",
    "build_unhandled_error_response",
]