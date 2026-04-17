"""
backend/main.py
================
FastAPI application factory and uvicorn entry point for SkinAura.

Run with::

    uvicorn backend.main:app --reload
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.analyze import router as analyze_router
from backend.api.health import router as health_router
from backend.api.predict import router as predict_router
from backend.core.config import get_settings
from backend.core.exceptions import SkinAuraError
from backend.core.logger import get_logger
from backend.models.model_loader import get_model_registry
from backend.utils.response_utils import (
    build_error_response,
    build_unhandled_error_response,
)

logger   = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("═══════════════════════════════════════════")
    logger.info("  %s v%s — starting up", settings.app_name, settings.app_version)
    logger.info("  API prefix : %s", settings.api_prefix or "(none)")
    logger.info("  Debug mode : %s", settings.debug)
    logger.info("═══════════════════════════════════════════")

    # Ensure upload directory exists.
    try:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Upload directory ready: %s", settings.upload_dir)
    except Exception as exc:
        logger.warning(
            "Could not create upload directory %s: %s — uploads may fail.",
            settings.upload_dir, exc,
        )

    # Pre-load PyTorch models to avoid cold-start on first request.
    try:
        registry = get_model_registry()
        _        = registry.skin_issue_model
        _        = registry.skin_tone_model
        logger.info(
            "ML models pre-loaded on %s. Status: %s",
            registry.device, registry.status(),
        )
    except Exception as exc:
        logger.warning(
            "Model pre-load failed — will retry on first request: %s", exc
        )

    logger.info(
        "SkinAura backend ready → http://%s:%d%s/health",
        settings.host, settings.port, settings.api_prefix,
    )

    yield

    logger.info("%s v%s shutting down.", settings.app_name, settings.app_version)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title=       settings.app_name,
        version=     settings.app_version,
        description=(
            "AI-powered skincare analysis system. "
            "Upload a face image to receive skin issue detection, "
            "tone classification, evidence-based knowledge retrieval, "
            "and a Groq-powered personalised routine."
        ),
        docs_url=    "/docs",
        redoc_url=   "/redoc",
        openapi_url= "/openapi.json",
        lifespan=    lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=     settings.allowed_origins,
        allow_credentials= True,
        allow_methods=     ["*"],
        allow_headers=     ["*"],
    )

    # Request timing header
    @app.middleware("http")
    async def _record_process_time(request: Request, call_next):
        t0       = time.perf_counter()
        response = await call_next(request)
        elapsed  = (time.perf_counter() - t0) * 1_000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
        return response

    # Global exception handlers
    @app.exception_handler(SkinAuraError)
    async def _handle_domain_error(request: Request, exc: SkinAuraError) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID")
        return build_error_response(exc, request_id=request_id)

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID")
        return build_unhandled_error_response(exc, request_id=request_id)

    # Routers — direct module imports; backend/api/__init__.py is empty so
    # `from backend.api import health` would raise ImportError at runtime.
    prefix = settings.api_prefix  # e.g. "/api/v1"

    app.include_router(health_router,  prefix=prefix)
    app.include_router(predict_router, prefix=prefix)
    app.include_router(analyze_router, prefix=prefix)

    # Root convenience endpoint
    @app.get("/", include_in_schema=False)
    async def _root() -> JSONResponse:
        return JSONResponse({
            "service": settings.app_name,
            "version": settings.app_version,
            "docs":    "/docs",
            "health":  f"{prefix}/health",
            "predict": f"{prefix}/predict",
            "analyze": f"{prefix}/analyze",
        })

    return app


# ---------------------------------------------------------------------------
# Module-level app instance — referenced by uvicorn
# ---------------------------------------------------------------------------

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=      settings.host,
        port=      settings.port,
        reload=    settings.debug,
        log_level= settings.log_level.lower(),
    )