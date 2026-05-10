"""
backend/api/health.py
======================
GET /health — service liveness and readiness probe.

Overall status logic
--------------------
- ``"ok"``       — all three sub-services are fully operational.
- ``"degraded"`` — at least one sub-service is genuinely broken (not merely
                   unconfigured).  Missing Groq key = expected dev state,
                   NOT degraded.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.core.logger import get_logger

router = APIRouter(tags=["Health"])
logger = get_logger(__name__)

# ── What counts as "healthy" for each sub-service ────────────────────────────
_HEALTHY_MODEL_STATUSES: frozenset[str] = frozenset({"loaded"})
_HEALTHY_RAG_STATUSES:   frozenset[str] = frozenset({"ready"})
_HEALTHY_LLM_STATUSES:   frozenset[str] = frozenset({
    "ready",
    "mock_mode",
    "not_initialised",   # No Groq key configured — expected during development.
                         # The pipeline still works via mock fallback, so this
                         # is NOT a degraded state.
})


def _get_settings():
    try:
        from backend.core.config import get_settings
        return get_settings()
    except Exception:
        return None


def _get_model_status() -> str:
    """
    Read model status from the predict module's in-process cache.
    Never re-loads from disk — zero latency on every health poll.
    """
    try:
        import backend.api.predict as _predict_mod
        if not _predict_mod._demo_mode and _predict_mod._model is not None:
            return "loaded"
        return "not_loaded"
    except Exception as exc:
        logger.warning("Could not determine model status: %s", exc)
        return "not_loaded"


def _get_rag_status() -> str:
    try:
        from backend.services.rag_service import get_rag_service
        return get_rag_service().status()
    except Exception as exc:
        logger.warning("RAG service status unavailable: %s", exc)
        return "not_loaded"


def _get_llm_status() -> str:
    try:
        from backend.services.llm_service import get_llm_service
        return get_llm_service().status()
    except Exception as exc:
        logger.warning("LLM service status unavailable: %s", exc)
        return "not_initialised"


def _derive_overall_status(
    model_status: str,
    rag_status:   str,
    llm_status:   str,
) -> str:
    """
    ``"ok"``       — every service is in a healthy/expected state.
    ``"degraded"`` — a service is in a genuinely broken state
                     (not merely unconfigured or in mock mode).
    """
    all_ok = (
        model_status in _HEALTHY_MODEL_STATUSES
        and rag_status   in _HEALTHY_RAG_STATUSES
        and llm_status   in _HEALTHY_LLM_STATUSES
    )
    return "ok" if all_ok else "degraded"


@router.get(
    "/health",
    summary="Service Health Check",
    description=(
        "Returns the operational status of SkinAura and all backend sub-services. "
        "Overall status is ``ok`` when all services are healthy or in an expected "
        "unconfigured state, ``degraded`` when a service is genuinely broken."
    ),
)
async def health_check() -> JSONResponse:
    """
    Aggregate sub-service statuses and return a health payload.
    Never raises — all status lookups are wrapped in try/except.
    """
    settings = _get_settings()
    version  = getattr(settings, "app_version", "1.0.0") if settings else "1.0.0"

    model_status = _get_model_status()
    rag_status   = _get_rag_status()
    llm_status   = _get_llm_status()

    overall_status = _derive_overall_status(model_status, rag_status, llm_status)

    # Only log a warning when something is GENUINELY broken, not on every poll.
    if overall_status == "degraded":
        logger.warning(
            "Health check: degraded (model_loader=%s, rag=%s, llm=%s).",
            model_status, rag_status, llm_status,
        )
    else:
        logger.debug(
            "Health check: ok (model_loader=%s, rag=%s, llm=%s).",
            model_status, rag_status, llm_status,
        )

    payload = {
        "status":   overall_status,
        "version":  version,
        "services": {
            "model_loader": model_status,
            "rag":          rag_status,
            "llm":          llm_status,
        },
    }

    # Prefer typed schema if available; fall back to raw dict (always works).
    try:
        from backend.schemas.responses import HealthResponse, ServiceStatusMap
        typed = HealthResponse(
            status=   overall_status,
            version=  version,
            services= ServiceStatusMap(
                model_loader= model_status,
                rag=          rag_status,
                llm=          llm_status,
            ),
        )
        return JSONResponse(content=typed.model_dump(mode="json"))
    except Exception:
        return JSONResponse(content=payload)