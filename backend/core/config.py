"""
backend/core/config.py
=======================
Centralised, environment-driven configuration for the SkinAura backend.

Design
------
- :class:`Settings` maps every public attribute 1-to-1 to an environment
  variable (case-insensitive).  Pydantic-settings resolves values from the
  environment first, then falls back to a ``.env`` file at the project root.
- Computed runtime values (derived paths, byte limits, feature flags) are
  exposed as ``@property`` methods so they never appear in the env-var
  namespace and cannot be accidentally overridden.
- ``target_image_size`` is stored as two discrete integer fields
  (``target_image_width`` / ``target_image_height``) to avoid pydantic-v2's
  ambiguous tuple parsing for env-vars, and re-combined via a property.
- Call :func:`get_settings` everywhere — never instantiate :class:`Settings`
  directly outside of tests.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project-root anchor
# ---------------------------------------------------------------------------
# Absolute path to SkinAura/ — two levels above backend/core/config.py.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_BACKEND_DIR:  Path = _PROJECT_ROOT / "backend"
_DATA_DIR:     Path = _PROJECT_ROOT / "data"
_MODELS_DIR:   Path = _PROJECT_ROOT / "saved_models"


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Application settings resolved from environment variables and/or a
    ``.env`` file located at the project root.

    All fields are validated at construction time.  Computed helpers are
    exposed as read-only ``@property`` methods.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ── Application metadata ────────────────────────────────────────────
    app_name:    str = Field(default="SkinAura",  description="Service display name.")
    app_version: str = Field(default="1.0.0",     description="Semantic version string.")
    api_prefix:  str = Field(default="/api/v1",   description="Global URL prefix for all routes.")
    debug:       bool = Field(default=False,       description="Enable debug mode and hot-reload.")

    # ── Server ──────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Bind address for the uvicorn server.")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port.")
    allowed_origins: List[str] = Field(
        default=["*"],
        description=(
            "CORS allowed-origin list.  Use explicit origins in production, "
            "e.g. [\"https://app.skinaura.ai\"]."
        ),
    )

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Root log level: DEBUG | INFO | WARNING | ERROR | CRITICAL.",
    )

    # ── Groq LLM ────────────────────────────────────────────────────────
    groq_api_key: str = Field(
        default="",
        alias="GROQ_API_KEY",
        description=(
            "Groq Cloud API key — https://console.groq.com/keys.  "
            "When absent the LLM service operates in mock-fallback mode."
        ),
    )
    groq_model_name: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier used for chat completions.",
    )
    groq_temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for Groq completions.",
    )
    groq_max_tokens: int = Field(
        default=1024,
        ge=128,
        le=8192,
        description="Maximum tokens in the Groq completion response.",
    )
    groq_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        description="HTTP timeout for Groq API requests in seconds.",
    )

    # ── Embeddings & RAG ────────────────────────────────────────────────
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace / sentence-transformers model used for chunk embeddings.",
    )
    faiss_index_path: Path = Field(
        default=_BACKEND_DIR / "rag" / "faiss.index",
        description="Absolute path to the persisted FAISS flat-IP index file.",
    )
    faiss_metadata_path: Path = Field(
        default=_BACKEND_DIR / "rag" / "faiss_metadata.pkl",
        description="Absolute path to the pickled knowledge-chunk metadata list.",
    )
    knowledge_base_path: Path = Field(
        default=_BACKEND_DIR / "rag" / "knowledge_base.json",
        description="Absolute path to the skincare knowledge base JSON file.",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of knowledge chunks to retrieve per query.",
    )

    # ── File uploads ────────────────────────────────────────────────────
    upload_dir: Path = Field(
        default=_DATA_DIR / "uploads",
        description=(
            "Project-relative directory for temporary uploaded image files.  "
            "Created automatically on startup."
        ),
    )
    max_file_size_mb: float = Field(
        default=10.0,
        ge=0.1,
        le=50.0,
        description="Maximum accepted upload size in megabytes.",
    )
    allowed_content_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"],
        description="MIME types accepted for image uploads.",
    )

    # ── ML models ───────────────────────────────────────────────────────
    saved_models_dir: Path = Field(
        default=_MODELS_DIR,
        description="Directory containing serialised PyTorch model weight files.",
    )
    skin_issue_model_filename: str = Field(
        default="skin_issue_model.pt",
        description="Filename of the skin-issue classifier weights inside ``saved_models_dir``.",
    )
    skin_tone_model_filename: str = Field(
        default="skin_tone_model.pt",
        description="Filename of the skin-tone classifier weights inside ``saved_models_dir``.",
    )
    # Stored as two fields to avoid pydantic-v2 env-var tuple ambiguity.
    target_image_width:  int = Field(default=224, ge=32, description="Inference input width in pixels.")
    target_image_height: int = Field(default=224, ge=32, description="Inference input height in pixels.")

    # ── Computed properties ─────────────────────────────────────────────

    @property
    def target_image_size(self) -> Tuple[int, int]:
        """Return ``(width, height)`` used when resizing images before inference."""
        return (self.target_image_width, self.target_image_height)

    @property
    def skin_issue_model_path(self) -> Path:
        """Absolute path to the skin-issue classifier weights file."""
        return self.saved_models_dir / self.skin_issue_model_filename

    @property
    def skin_tone_model_path(self) -> Path:
        """Absolute path to the skin-tone classifier weights file."""
        return self.saved_models_dir / self.skin_tone_model_filename

    @property
    def max_file_size_bytes(self) -> int:
        """``max_file_size_mb`` expressed in bytes."""
        return int(self.max_file_size_mb * 1_024 * 1_024)

    @property
    def has_groq_key(self) -> bool:
        """``True`` when a non-empty Groq API key is configured."""
        return bool(self.groq_api_key.strip())

    # ── Validators ──────────────────────────────────────────────────────

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, value: object) -> str:
        normalised = str(value).upper().strip()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalised not in valid:
            raise ValueError(
                f"Invalid log_level {value!r}. Must be one of: {sorted(valid)}."
            )
        return normalised

    @field_validator("api_prefix", mode="before")
    @classmethod
    def _validate_api_prefix(cls, value: object) -> str:
        prefix = str(value).rstrip("/")
        if prefix and not prefix.startswith("/"):
            raise ValueError(
                f"api_prefix must start with '/' or be empty; got {value!r}."
            )
        return prefix


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application-wide :class:`Settings` singleton.

    The instance is constructed once and cached for the process lifetime.
    Safe to call at module import time from any part of the codebase.
    """
    return Settings()