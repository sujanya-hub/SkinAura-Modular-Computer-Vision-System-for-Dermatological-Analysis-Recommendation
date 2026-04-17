"""
backend/schemas/responses.py
=============================
Outbound Pydantic response models — the single source of truth for all
API response shapes in SkinAura.

Design
------
- All response models inherit from :class:`_BaseResponse` which applies
  ``from_attributes=True`` uniformly, enabling construction from both
  plain dicts and ORM-like objects.
- Models that originate from a user request carry ``request_id`` and
  ``timestamp`` for end-to-end traceability.
- CV analysis results are always grouped inside :class:`CVOutput`.  This
  nested structure is the canonical shape used in both
  :class:`PredictionResponse` and :class:`FullAnalysisResponse`.  Services
  must never flatten these fields into the parent model.
- :class:`ErrorResponse` is the single error envelope emitted by all
  exception handlers.  Route functions must not construct ad-hoc error
  shapes.

Schema hierarchy
----------------
    HealthResponse
        └── ServiceStatusMap

    PredictionResponse              ← POST /predict (CV only)
        └── CVOutput
                ├── List[DetectedSkinIssue]
                └── SkinToneResult

    FullAnalysisResponse            ← POST /analyze (full pipeline)
        ├── CVOutput
        ├── List[RetrievedKnowledgeChunk]
        └── Optional[LLMAnalysisOutput]
                └── List[SkincareRoutineStep]

    ErrorResponse                   ← all exception handlers

Critical
--------
The field names, types, and nesting structure of every model in this file
are locked to the constructor calls in
:mod:`backend.services.recommendation_service`,
:mod:`backend.api.health`, and
:mod:`backend.utils.response_utils`.
Do not rename fields without updating those callers simultaneously.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _BaseResponse(BaseModel):
    """Common configuration applied to all response models."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


class _TimestampedResponse(_BaseResponse):
    """
    Mixin that stamps every user-initiated response with an ISO-8601 UTC
    timestamp.  Inherited by all top-level response models except
    :class:`ErrorResponse` which sets its own.
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this response was generated.",
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ServiceStatusMap(_BaseResponse):
    """
    Operational status strings for each SkinAura backend sub-service.

    Values follow a consistent vocabulary:
      - ``"loaded"`` / ``"not_loaded"`` for the model loader
      - ``"ready"`` / ``"not_loaded"`` for RAG
      - ``"ready"`` / ``"mock_mode"`` / ``"error:<reason>"`` for LLM

    ``protected_namespaces=()`` suppresses the Pydantic v2 warning that
    the ``model_loader`` field name starts with the reserved ``model_``
    prefix.  The field name is intentional and must not be renamed.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
    )

    model_loader: str = Field(
        description="ML model registry status: loaded | not_loaded.",
    )
    rag: str = Field(
        description="RAG service status: ready | not_loaded.",
    )
    llm: str = Field(
        description="LLM service status: ready | mock_mode | error:<reason>.",
    )


class HealthResponse(_TimestampedResponse):
    """Response model for ``GET /health``."""

    status: str = Field(
        default="ok",
        description="Overall service status: ok | degraded.",
        examples=["ok"],
    )
    version: str = Field(
        description="Running application version string.",
        examples=["1.0.0"],
    )
    services: ServiceStatusMap = Field(
        description="Operational status of each backend sub-service.",
    )


# ---------------------------------------------------------------------------
# CV output sub-models
# ---------------------------------------------------------------------------


class DetectedSkinIssue(_BaseResponse):
    """
    A single skin condition surfaced by the issue classifier.

    Constructed in :meth:`~backend.services.recommendation_service.RecommendationService._build_cv_output`
    from the raw CV service dict with keys ``name``, ``confidence``, and
    optionally ``severity``.
    """

    name: str = Field(
        description="Canonical lowercase issue label, e.g. 'acne', 'hyperpigmentation'.",
        examples=["acne"],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Softmax confidence score in [0.0, 1.0].",
        examples=[0.82],
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severity bracket derived from confidence: mild | moderate | severe.",
        examples=["moderate"],
    )


class SkinToneResult(_BaseResponse):
    """
    Fitzpatrick scale tone classification and undertone prediction.

    Constructed in :meth:`~backend.services.recommendation_service.RecommendationService._build_cv_output`
    from the raw CV service dict with keys ``tone_label``, ``hex_color``,
    ``undertone``, and ``confidence``.
    """

    tone_label: str = Field(
        description="Fitzpatrick scale label, e.g. 'Type III (Medium)'.",
        examples=["Type III (Medium)"],
    )
    hex_color: str = Field(
        description=(
            "Approximate skin hex colour code sampled from detected pixels.  "
            "Format: #RRGGBB."
        ),
        examples=["#C68642"],
        pattern=r"^#[0-9A-Fa-f]{6}$",
    )
    undertone: str = Field(
        description="Dominant undertone category: warm | neutral | cool.",
        examples=["warm"],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classifier confidence for the tone prediction in [0.0, 1.0].",
        examples=[0.74],
    )


class CVOutput(_BaseResponse):
    """
    Aggregated computer-vision analysis results.

    This is the **canonical CV container** used in both
    :class:`PredictionResponse` and :class:`FullAnalysisResponse`.  It is
    always accessed via the ``cv_output`` field on the parent model and
    must never be flattened into the parent.

    Constructed by
    :meth:`~backend.services.recommendation_service.RecommendationService._build_cv_output`.
    """

    face_detected: bool = Field(
        description=(
            "True when a face region was successfully detected and used as the "
            "inference input.  False when the full image was used as a fallback."
        ),
    )
    skin_issues: List[DetectedSkinIssue] = Field(
        default_factory=list,
        description="Top-k detected skin conditions sorted by descending confidence.",
    )
    skin_tone: SkinToneResult = Field(
        description="Fitzpatrick tone and undertone classification result.",
    )
    bounding_box: Optional[Dict[str, int]] = Field(
        default=None,
        description=(
            "Face bounding box in image pixel coordinates.  "
            "Shape: {'x': int, 'y': int, 'width': int, 'height': int}.  "
            "None when face detection was skipped or failed."
        ),
    )


# ---------------------------------------------------------------------------
# RAG output sub-model
# ---------------------------------------------------------------------------


class RetrievedKnowledgeChunk(_BaseResponse):
    """
    A single knowledge chunk returned by the FAISS retrieval step.

    Constructed by
    :meth:`~backend.services.recommendation_service.RecommendationService._build_knowledge_chunks`
    from raw RAG service dicts.  Field names match the knowledge base JSON
    schema and the metadata pickle produced by
    :mod:`backend.rag.build_index`.
    """

    chunk_id: str = Field(
        description="Unique identifier of the knowledge chunk (from knowledge_base.json 'id').",
    )
    title: Optional[str] = Field(
        default=None,
        description="Human-readable chunk title.",
    )
    text: str = Field(
        description="Full text content of the knowledge chunk.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Skincare category label, e.g. 'acne', 'hyperpigmentation', 'general'.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Attribution source label, e.g. 'SkinAura Knowledge Base'.",
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity score from FAISS inner-product search on "
            "L2-normalised embeddings.  Higher is more relevant."
        ),
    )


# ---------------------------------------------------------------------------
# LLM output sub-models
# ---------------------------------------------------------------------------


class SkincareRoutineStep(_BaseResponse):
    """
    A single step in the personalised skincare routine.

    Constructed by
    :meth:`~backend.services.recommendation_service.RecommendationService._build_llm_analysis`
    from the parsed Groq JSON response.  Field names must match the JSON
    keys enforced by the LLM prompt schema in :mod:`backend.services.llm_service`.
    """

    step: int = Field(
        ge=1,
        description="Ordinal step number within the phase (1-indexed).",
    )
    phase: str = Field(
        description="Routine phase: morning | evening | weekly.",
        examples=["morning"],
    )
    action: str = Field(
        description="Short imperative description of the step.",
        examples=["Gentle Cleanse"],
    )
    product_type: str = Field(
        description="Product category to use for this step.",
        examples=["Low-pH gel cleanser"],
    )
    key_ingredients: List[str] = Field(
        default_factory=list,
        description="Active ingredients to look for on the product label.",
        examples=[["salicylic acid", "niacinamide"]],
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional usage tip or application note.",
        examples=["Apply to damp skin; rinse after 60 seconds."],
    )


class LLMAnalysisOutput(_BaseResponse):
    """
    Structured output produced by the Groq LLM generation step.

    Constructed by
    :meth:`~backend.services.recommendation_service.RecommendationService._build_llm_analysis`
    from an :class:`~backend.services.llm_service.LLMOutput` dataclass.

    The ``generated_by`` field records which model produced the output, or
    ``"mock_fallback"`` when the Groq API was unavailable.

    This class name and all field names are locked to the constructor call in
    :mod:`backend.services.recommendation_service`.
    """

    explanation: str = Field(
        description=(
            "2–3 paragraph natural-language summary of the skin analysis findings, "
            "written for a general audience.  Includes cosmetic skincare guidance "
            "and a dermatologist-referral note where appropriate."
        ),
    )
    routine: List[SkincareRoutineStep] = Field(
        default_factory=list,
        description="Personalised morning, evening, and optional weekly routine steps.",
    )
    ingredient_suggestions: List[str] = Field(
        default_factory=list,
        description="Active ingredients recommended for this skin profile.",
        examples=[["Niacinamide", "Hyaluronic Acid", "Ceramides"]],
    )
    ingredients_to_avoid: List[str] = Field(
        default_factory=list,
        description="Ingredient classes or specific ingredients to avoid.",
        examples=[["Fragrance / Parfum", "Alcohol Denat."]],
    )
    precautions: List[str] = Field(
        default_factory=list,
        description=(
            "Safety notes, patch-test guidance, and professional referral advice "
            "where relevant."
        ),
    )
    generated_by: str = Field(
        default="unknown",
        description=(
            "Model identifier that produced this output, "
            "e.g. 'llama-3.3-70b-versatile' or 'mock_fallback'."
        ),
        examples=["llama-3.3-70b-versatile"],
    )


# ---------------------------------------------------------------------------
# Top-level response models
# ---------------------------------------------------------------------------


class PredictionResponse(_TimestampedResponse):
    """
    Response model for ``POST /predict`` — CV-only pipeline.

    Does not include RAG knowledge chunks or LLM analysis.
    CV results are grouped inside :attr:`cv_output`.

    Constructed by
    :meth:`~backend.services.recommendation_service.RecommendationService.run_cv_pipeline`.
    """

    request_id: str = Field(
        description="Short hex correlation ID for end-to-end log tracing.",
        examples=["a1b2c3d4e5f6"],
    )
    cv_output: CVOutput = Field(
        description="Aggregated computer-vision analysis results.",
    )
    processing_time_ms: float = Field(
        ge=0.0,
        description="Wall-clock processing time for this request in milliseconds.",
        examples=[145.3],
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached by the pipeline.",
    )


class FullAnalysisResponse(_TimestampedResponse):
    """
    Response model for ``POST /analyze`` — full CV + RAG + LLM pipeline.

    This is the primary consumer-facing response shape.  CV results are
    grouped inside :attr:`cv_output`.  RAG chunks and LLM output are
    present only when the respective pipeline stages were enabled.

    Constructed by
    :meth:`~backend.services.recommendation_service.RecommendationService.run_full_pipeline`.
    """

    request_id: str = Field(
        description="Short hex correlation ID for end-to-end log tracing.",
        examples=["a1b2c3d4e5f6"],
    )
    cv_output: CVOutput = Field(
        description="Aggregated computer-vision analysis results.",
    )
    retrieved_knowledge: List[RetrievedKnowledgeChunk] = Field(
        default_factory=list,
        description=(
            "Top-k knowledge chunks retrieved from the FAISS index.  "
            "Empty when ``enable_rag`` was set to ``False``."
        ),
    )
    analysis: Optional[LLMAnalysisOutput] = Field(
        default=None,
        description=(
            "LLM-generated skincare analysis.  "
            "``None`` when ``enable_llm`` was set to ``False``."
        ),
    )
    processing_time_ms: float = Field(
        ge=0.0,
        description="Wall-clock processing time for the full pipeline in milliseconds.",
        examples=[1240.7],
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached by the pipeline.",
    )


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class ErrorResponse(_TimestampedResponse):
    """
    Standardised error envelope returned by all API exception handlers.

    Constructed in :mod:`backend.utils.response_utils` from
    :class:`~backend.core.exceptions.SkinAuraError` instances.  Route
    functions must not construct ad-hoc error shapes — use
    :func:`~backend.utils.response_utils.build_error_response` instead.

    The ``details`` field is omitted from the JSON output when ``None``
    to keep clean-path error responses minimal.
    """

    error: str = Field(
        description="Exception class name, e.g. 'FaceDetectionError'.",
        examples=["FaceDetectionError"],
    )
    message: str = Field(
        description="Human-readable description of what went wrong.",
        examples=["No face region could be detected in the provided image."],
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Correlation ID from the originating request, when available.",
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured context, e.g. filenames, resolutions, model names.",
    )


# ---------------------------------------------------------------------------
# Public re-export list
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # Health
    "ServiceStatusMap",
    "HealthResponse",
    # CV sub-models
    "DetectedSkinIssue",
    "SkinToneResult",
    "CVOutput",
    # RAG sub-model
    "RetrievedKnowledgeChunk",
    # LLM sub-models
    "SkincareRoutineStep",
    "LLMAnalysisOutput",
    # Top-level responses
    "PredictionResponse",
    "FullAnalysisResponse",
    # Error
    "ErrorResponse",
]