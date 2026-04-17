"""
backend/main.py
FastAPI entry-point for SkinAura.
Every startup step is wrapped in try/except — the process never exits on
a model error, missing config key, or unavailable directory.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe setting helpers
# ---------------------------------------------------------------------------

def _get_settings():
    try:
        from backend.core.config import get_settings
        return get_settings()
    except Exception as exc:
        logger.warning("Could not load settings: %s", exc)
        return None


def _setting(attr: str, default):
    s = _get_settings()
    return getattr(s, attr, default) if s else default


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app_name    = _setting("app_name",    "SkinAura")
    app_version = _setting("app_version", "1.0.0")
    api_prefix  = _setting("api_prefix",  "/api/v1")
    debug       = _setting("debug",       False)

    logger.info("═══════════════════════════════════════════")
    logger.info("  %s v%s — starting up", app_name, app_version)
    logger.info("  API prefix : %s", api_prefix or "(none)")
    logger.info("  Debug mode : %s", debug)
    logger.info("═══════════════════════════════════════════")

    # Upload directory — failure is non-fatal.
    try:
        upload_dir = _setting("upload_dir", None)
        if upload_dir:
            import pathlib
            pathlib.Path(upload_dir).mkdir(parents=True, exist_ok=True)
            logger.info("Upload directory ready: %s", upload_dir)
    except Exception as exc:
        logger.warning("Could not create upload directory: %s", exc)

    # Model pre-load — failure is non-fatal; lazy load on first request.
    try:
        from backend.models.model_loader import get_model_registry
        registry = get_model_registry()
        _        = registry.skin_issue_model
        _        = registry.skin_tone_model
        logger.info("ML models pre-loaded. Status: %s", registry.status())
    except Exception as exc:
        logger.warning("Model pre-load skipped: %s", exc)

    host = _setting("host", "0.0.0.0")
    port = int(os.environ.get("PORT", _setting("port", 10000)))
    logger.info("SkinAura backend ready → http://%s:%d%s/health", host, port, api_prefix)

    yield

    logger.info("%s v%s shutting down.", app_name, app_version)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings    = _get_settings()
    app_name    = getattr(settings, "app_name",    "SkinAura")    if settings else "SkinAura"
    app_version = getattr(settings, "app_version", "1.0.0")       if settings else "1.0.0"
    api_prefix  = getattr(settings, "api_prefix",  "/api/v1")     if settings else "/api/v1"
    origins     = getattr(settings, "allowed_origins", ["*"])      if settings else ["*"]

    app = FastAPI(
        title=       app_name,
        version=     app_version,
        description= "AI-powered skincare analysis system.",
        docs_url=    "/docs",
        redoc_url=   "/redoc",
        openapi_url= "/openapi.json",
        lifespan=    lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=     origins,
        allow_credentials= True,
        allow_methods=     ["*"],
        allow_headers=     ["*"],
    )

    @app.middleware("http")
    async def _timing(request: Request, call_next):
        t0       = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter()-t0)*1000:.2f}"
        return response

    # Domain error handler — only registered when the exception class exists.
    try:
        from backend.core.exceptions import SkinAuraError
        from backend.utils.response_utils import (
            build_error_response,
            build_unhandled_error_response,
        )

        @app.exception_handler(SkinAuraError)
        async def _domain_err(request: Request, exc: SkinAuraError) -> JSONResponse:
            return build_error_response(exc, request_id=request.headers.get("X-Request-ID"))

        @app.exception_handler(Exception)
        async def _generic_err(request: Request, exc: Exception) -> JSONResponse:
            return build_unhandled_error_response(exc, request_id=request.headers.get("X-Request-ID"))

    except Exception as exc:
        logger.warning("Custom exception handlers not registered: %s", exc)

    # Routers — each wrapped so one bad import never kills the app.
    def _include(module_path: str, attr: str, prefix: str, tag: str):
        try:
            import importlib
            mod    = importlib.import_module(module_path)
            router = getattr(mod, attr)
            app.include_router(router, prefix=prefix)
            logger.info("Router registered: %s%s", prefix, f" ({tag})")
        except Exception as exc:
            logger.warning("Could not register router '%s': %s", tag, exc)

    _include("backend.api.health",  "router", api_prefix, "health")
    _include("backend.api.predict", "router", api_prefix, "predict")
    _include("backend.api.analyze", "router", api_prefix, "analyze")

    @app.get("/", include_in_schema=False)
    async def _root() -> JSONResponse:
        return JSONResponse({
            "service": app_name,
            "version": app_version,
            "docs":    "/docs",
            "health":  f"{api_prefix}/health",
            "predict": f"{api_prefix}/predict",
            "analyze": f"{api_prefix}/analyze",
        })

    return app


# ---------------------------------------------------------------------------
# Module-level app instance
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", _setting("port", 10000)))
    uvicorn.run(
        "backend.main:app",
        host=      "0.0.0.0",
        port=      port,
        reload=    False,
        log_level= "info",
    )