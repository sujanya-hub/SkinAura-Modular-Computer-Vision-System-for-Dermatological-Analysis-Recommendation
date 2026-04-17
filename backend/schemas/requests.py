"""
backend/schemas/requests.py
============================
Inbound Pydantic request models for the SkinAura API.

Design
------
- The image payload itself arrives as ``UploadFile`` (multipart) in the
  route signature.  These schemas carry only the *metadata* that
  accompanies an upload — personalisation context and pipeline flags.
- :class:`SkinConcernsInput` is the primary personalisation model.  It is
  parsed from a JSON string passed as the ``user_context`` form field in
  ``POST /analyze``, then serialised to a plain dict via
  ``model.model_dump(exclude_none=True)`` before being forwarded to services.
- All controlled-vocabulary fields are validated against explicit allow-sets
  so invalid values are rejected at the boundary with a clear error message.
- :class:`ImagePathRequest` covers internal/CLI use cases where an image is
  referenced by path rather than uploaded directly.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------

_VALID_SKIN_TYPES: frozenset[str] = frozenset(
    {"oily", "dry", "combination", "normal", "sensitive"}
)
_VALID_AGE_GROUPS: frozenset[str] = frozenset(
    {"teens", "20s", "30s", "40s", "50s", "60+"}
)
_VALID_ROUTINE_PREFERENCES: frozenset[str] = frozenset(
    {"minimal", "moderate", "comprehensive"}
)


# ---------------------------------------------------------------------------
# User personalisation context
# ---------------------------------------------------------------------------


class SkinConcernsInput(BaseModel):
    """
    Optional personalisation context supplied by the user alongside an image.

    All fields are optional.  The pipeline produces sensible output from CV
    results alone when this model is absent or partially populated.

    This model is consumed by ``POST /analyze`` and forwarded to the
    recommendation service as a plain dict via ``model_dump(exclude_none=True)``.
    """

    skin_type: Optional[str] = Field(
        default=None,
        description=(
            "Self-reported skin type.  "
            "Accepted values: oily | dry | combination | normal | sensitive."
        ),
        examples=["oily"],
    )
    age_group: Optional[str] = Field(
        default=None,
        description=(
            "Broad age bracket used to contextualise routine recommendations.  "
            "Accepted values: teens | 20s | 30s | 40s | 50s | 60+."
        ),
        examples=["30s"],
    )
    known_concerns: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        max_length=10,
        description=(
            "User-reported skin concerns, e.g. ['acne', 'dark spots', 'redness'].  "
            "Entries are lowercased and deduplicated.  Maximum 10 items."
        ),
        examples=[["acne", "hyperpigmentation"]],
    )
    routine_preference: Optional[str] = Field(
        default=None,
        description=(
            "Desired routine complexity.  "
            "Accepted values: minimal | moderate | comprehensive."
        ),
        examples=["moderate"],
    )
    free_text_query: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
        description=(
            "Open-ended note passed verbatim into the LLM prompt.  "
            "Use for preferences not captured by structured fields, "
            "e.g. 'I prefer fragrance-free products' or 'I travel frequently'."
        ),
        examples=["I have very sensitive skin and react to most fragrances."],
    )

    # ── Validators ────────────────────────────────────────────────────

    @field_validator("skin_type", mode="before")
    @classmethod
    def _validate_skin_type(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        normalised = str(value).strip().lower()
        if normalised not in _VALID_SKIN_TYPES:
            raise ValueError(
                f"skin_type {value!r} is not recognised.  "
                f"Must be one of: {sorted(_VALID_SKIN_TYPES)}."
            )
        return normalised

    @field_validator("age_group", mode="before")
    @classmethod
    def _validate_age_group(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        normalised = str(value).strip().lower()
        if normalised not in _VALID_AGE_GROUPS:
            raise ValueError(
                f"age_group {value!r} is not recognised.  "
                f"Must be one of: {sorted(_VALID_AGE_GROUPS)}."
            )
        return normalised

    @field_validator("routine_preference", mode="before")
    @classmethod
    def _validate_routine_preference(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        normalised = str(value).strip().lower()
        if normalised not in _VALID_ROUTINE_PREFERENCES:
            raise ValueError(
                f"routine_preference {value!r} is not recognised.  "
                f"Must be one of: {sorted(_VALID_ROUTINE_PREFERENCES)}."
            )
        return normalised

    @field_validator("known_concerns", mode="before")
    @classmethod
    def _normalise_concerns(cls, value: object) -> Optional[List[str]]:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("known_concerns must be a list of strings.")
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in value:
            entry = str(item).strip().lower()
            if entry and entry not in seen:
                cleaned.append(entry)
                seen.add(entry)
        return cleaned if cleaned else None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "skin_type": "combination",
                    "age_group": "30s",
                    "known_concerns": ["hyperpigmentation", "dryness"],
                    "routine_preference": "comprehensive",
                    "free_text_query": "I prefer fragrance-free products.",
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Route-level request models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """
    Metadata model for ``POST /analyze``.

    The image bytes arrive as a separate ``UploadFile`` form field.
    This schema is parsed from the optional ``user_context`` JSON form
    field.  Pipeline control flags (``enable_rag``, ``enable_llm``) are
    separate boolean form fields parsed directly in the route function.

    This model is kept as a standalone schema for documentation and
    potential future use (e.g. JSON body mode, SDK generation).
    """

    user_context: Optional[SkinConcernsInput] = Field(
        default=None,
        description="Optional personalisation context.",
    )
    enable_rag: bool = Field(
        default=True,
        description="Run the FAISS knowledge-retrieval step.",
    )
    enable_llm: bool = Field(
        default=True,
        description="Run the Groq LLM analysis generation step.",
    )

    @model_validator(mode="after")
    def _validate_pipeline_flags(self) -> "AnalyzeRequest":
        # LLM without RAG is intentionally permitted; the service handles
        # the reduced-context generation gracefully.
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_context": {
                        "skin_type": "oily",
                        "age_group": "20s",
                        "known_concerns": ["acne"],
                        "routine_preference": "minimal",
                    },
                    "enable_rag": True,
                    "enable_llm": True,
                }
            ]
        }
    }


class ImagePathRequest(BaseModel):
    """
    Internal request model for background tasks, CLI tooling, or
    service-to-service calls where an image is referenced by file path
    rather than uploaded directly via multipart.
    """

    image_path: str = Field(
        description="Absolute or project-relative path to the source image file.",
        examples=["/home/app/data/uploads/abc123.jpg"],
    )
    user_context: Optional[SkinConcernsInput] = Field(
        default=None,
        description="Optional personalisation context.",
    )


# ---------------------------------------------------------------------------
# Public re-export list
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "SkinConcernsInput",
    "AnalyzeRequest",
    "ImagePathRequest",
]