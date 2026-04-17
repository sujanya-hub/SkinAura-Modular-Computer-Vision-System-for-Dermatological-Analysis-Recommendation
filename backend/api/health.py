"""
backend/api/health.py
======================
GET /health — service liveness and readiness probe.

Returns a :class:`~backend.schemas.responses.HealthResponse` summarising
the operational status of every SkinAura backend sub-service.

Overall status logic
--------------------
- ``"ok"``       — all three sub-services are fully operational.
- ``"degraded"`` — at least one sub-service is not ready but the
                   application is still able to serve partial responses
                   (e.g. CV-only when RAG or LLM is unavailable).
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.models.model_loader import get_model_registry
from backend.schemas.responses import HealthResponse, ServiceStatusMap
from backend.services.llm_service import get_llm_service
from backend.services.rag_service import get_rag_service

router   = APIRouter(tags=["Health"])
logger   = get_logger(__name__)
settings = get_settings()

# Sub-service status strings considered "fully operational".
_HEALTHY_MODEL_STATUS: str = "loaded"
_HEALTHY_RAG_STATUS:   str = "ready"
_HEALTHY_LLM_STATUSES: frozenset[str] = frozenset({"ready", "mock_mode"})


def _derive_overall_status(
    model_status: str,
    rag_status:   str,
    llm_status:   str,
) -> str:
    """
    Derive the top-level ``status`` field.

    The application is considered ``"ok"`` when:
      - The model loader has ``"loaded"`` all models.
      - The RAG service is ``"ready"``.
      - The LLM service is ``"ready"`` *or* ``"mock_mode"`` (Groq absent
        but fallback is active — this is an expected deployment state
        during development, not an error).

    Any other combination returns ``"degraded"``.

    Args:
        model_status: Aggregated model-loader status string.
        rag_status:   RAG service status string.
        llm_status:   LLM service status string.

    Returns:
        ``"ok"`` or ``"degraded"``.
    """
    if (
        model_status == _HEALTHY_MODEL_STATUS
        and rag_status == _HEALTHY_RAG_STATUS
        and llm_status in _HEALTHY_LLM_STATUSES
    ):
        return "ok"
    return "degraded"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description=(
        "Returns the operational status of SkinAura and all backend sub-services.  "
        "Overall status is ``ok`` when all services are healthy, ``degraded`` when "
        "one or more services are unavailable."
    ),
)
async def health_check() -> JSONResponse:
    """
    Aggregate sub-service statuses and return a
    :class:`~backend.schemas.responses.HealthResponse`.

    This endpoint never raises — all status lookups are non-blocking
    property reads on already-initialised singletons.
    """
    registry = get_model_registry()
    model_stati = registry.status()  # Dict[str, "loaded" | "not_loaded"]

    # Both models must be loaded for the model_loader to be considered healthy.
    model_status: str = (
        _HEALTHY_MODEL_STATUS
        if all(v == _HEALTHY_MODEL_STATUS for v in model_stati.values())
        else "not_loaded"
    )

    rag_status: str = get_rag_service().status()
    llm_status: str = get_llm_service().status()

    overall_status = _derive_overall_status(model_status, rag_status, llm_status)

    if overall_status == "degraded":
        logger.warning(
            "Health check: degraded (model_loader=%s, rag=%s, llm=%s).",
            model_status, rag_status, llm_status,
        )

    payload = HealthResponse(
        status=   overall_status,
        version=  settings.app_version,
        services= ServiceStatusMap(
            model_loader= model_status,
            rag=          rag_status,
            llm=          llm_status,
        ),
    )
    return JSONResponse(content=payload.model_dump(mode="json"))