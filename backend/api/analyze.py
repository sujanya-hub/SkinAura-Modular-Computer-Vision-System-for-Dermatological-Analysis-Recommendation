"""
backend/api/analyze.py
=======================
POST /analyze — full CV + RAG + LLM skincare analysis pipeline.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.core.exceptions import SkinAuraError, ValidationError
from backend.core.logger import get_logger
from backend.schemas.requests import SkinConcernsInput
from backend.schemas.responses import FullAnalysisResponse
from backend.services.recommendation_service import get_recommendation_service
from backend.utils.image_utils import save_upload, validate_upload
from backend.utils.response_utils import (
    build_error_response,
    build_unhandled_error_response,
    generate_request_id,
)

router = APIRouter(tags=["Analysis"])
logger = get_logger(__name__)


def _parse_user_context(raw: str, request_id: str) -> Optional[dict]:
    """
    Parse and validate the ``user_context`` JSON form field.

    Returns a plain dict from SkinConcernsInput.model_dump(exclude_none=True),
    or None if raw is blank.

    Raises ValidationError on bad JSON or schema validation failure.
    """
    stripped = raw.strip()
    if not stripped:
        return None

    try:
        raw_dict = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            "user_context is not valid JSON.",
            details={"error": str(exc), "preview": stripped[:200]},
        ) from exc

    try:
        validated = SkinConcernsInput(**raw_dict)
    except Exception as exc:
        raise ValidationError(
            "user_context failed schema validation.",
            details={"error": str(exc), "preview": stripped[:200]},
        ) from exc

    parsed = validated.model_dump(exclude_none=True)
    logger.debug("[%s] user_context parsed: %s.", request_id, parsed)
    return parsed


@router.post(
    "/analyze",
    response_model=FullAnalysisResponse,
    summary="Full Skincare Analysis (CV + RAG + LLM)",
    description=(
        "Upload a face image with optional user context to receive a complete "
        "personalised skincare analysis: face detection, skin issue prediction, "
        "skin tone classification, FAISS knowledge retrieval, and a Groq-powered "
        "routine with ingredient recommendations. "
        "Set enable_rag=false or enable_llm=false to skip those stages."
    ),
    responses={
        200: {"description": "Full analysis completed successfully."},
        400: {"description": "Invalid request — malformed user_context or image."},
        422: {"description": "Image could not be decoded or processed."},
        500: {"description": "Internal server error."},
    },
)
async def analyze(
    request: Request,
    image: UploadFile = File(
        ...,
        description="Face image file. Accepted formats: JPEG, PNG, WEBP.",
    ),
    user_context: Optional[str] = Form(
        default=None,
        description=(
            "Optional JSON string conforming to SkinConcernsInput. "
            'Example: {"skin_type":"oily","known_concerns":["acne"],"age_group":"20s"}'
        ),
    ),
    enable_rag: bool = Form(
        default=True,
        description="Run the FAISS knowledge-retrieval step.",
    ),
    enable_llm: bool = Form(
        default=True,
        description="Run the Groq LLM analysis generation step.",
    ),
) -> JSONResponse:
    request_id = generate_request_id()
    tmp_path   = None

    # 1. Parse optional user context before reading image bytes so a
    #    malformed JSON field returns 400 without consuming upload bandwidth.
    parsed_context: Optional[dict] = None
    if user_context:
        try:
            parsed_context = _parse_user_context(user_context, request_id)
        except ValidationError as exc:
            return build_error_response(exc, request_id=request_id)

    try:
        # 2. Read and validate image bytes.
        image_bytes = await image.read()
        validate_upload(image_bytes, content_type=image.content_type)

        # 3. Persist to upload_dir — OpenCV needs a file path.
        tmp_path = save_upload(image_bytes)

        # 4. Run the full CV → RAG → LLM pipeline.
        service  = get_recommendation_service()
        response: FullAnalysisResponse = service.run_full_pipeline(
            image_source=str(tmp_path),
            user_context=parsed_context,
            request_id=request_id,
            enable_rag=enable_rag,
            enable_llm=enable_llm,
        )

        logger.info(
            "[%s] /analyze → %dms, face_detected=%s, rag=%s, llm=%s.",
            request_id,
            int(response.processing_time_ms),
            response.cv_output.face_detected,
            enable_rag,
            enable_llm,
        )
        return JSONResponse(content=response.model_dump(mode="json"))

    except SkinAuraError as exc:
        return build_error_response(exc, request_id=request_id)
    except Exception as exc:
        return build_unhandled_error_response(exc, request_id=request_id)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass