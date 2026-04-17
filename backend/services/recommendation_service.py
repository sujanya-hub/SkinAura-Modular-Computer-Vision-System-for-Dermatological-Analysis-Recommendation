"""
backend/services/recommendation_service.py
==========================================
Orchestrates the full CV → RAG → LLM pipeline and assembles the final
validated Pydantic response objects consumed by the API layer.

Schema contract
---------------
All response classes are imported directly from
:mod:`backend.schemas.responses` — no ad-hoc dict shapes are constructed
in this module.  The mapping is:

    CVService output dict
        → :class:`~backend.schemas.responses.CVOutput` (
            face_detected, skin_issues: List[DetectedSkinIssue],
            skin_tone: SkinToneResult, bounding_box
          )

    RAGService output dicts
        → List[:class:`~backend.schemas.responses.RetrievedKnowledgeChunk`]

    :class:`~backend.services.llm_service.LLMOutput` dataclass
        → :class:`~backend.schemas.responses.LLMAnalysisOutput` (
            explanation, routine: List[SkincareRoutineStep],
            ingredient_suggestions, ingredients_to_avoid,
            precautions, generated_by
          )

    All assembled into
    :class:`~backend.schemas.responses.PredictionResponse` or
    :class:`~backend.schemas.responses.FullAnalysisResponse`.

Public interface
----------------
:meth:`RecommendationService.run_cv_pipeline`
    CV-only path called by ``POST /predict``.

:meth:`RecommendationService.run_full_pipeline`
    Full CV → RAG → LLM path called by ``POST /analyze``.
    Supports ``enable_rag`` and ``enable_llm`` flags to selectively
    skip pipeline stages.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from backend.core.logger import get_logger
from backend.schemas.responses import (
    CVOutput,
    DetectedSkinIssue,
    FullAnalysisResponse,
    LLMAnalysisOutput,
    PredictionResponse,
    RetrievedKnowledgeChunk,
    SkincareRoutineStep,
    SkinToneResult,
)
from backend.services.cv_service import get_cv_service
from backend.services.llm_service import LLMOutput, get_llm_service
from backend.services.rag_service import get_rag_service
from backend.utils.response_utils import generate_request_id

logger = get_logger(__name__)


class RecommendationService:
    """
    Pipeline orchestrator that ties together
    :class:`~backend.services.cv_service.CVService`,
    :class:`~backend.services.rag_service.RAGService`, and
    :class:`~backend.services.llm_service.LLMService`.

    Both public methods return fully validated Pydantic response objects.
    The API routes call these methods and serialise the result directly via
    ``response.model_dump(mode="json")``.
    """

    def __init__(self) -> None:
        self._cv  = get_cv_service()
        self._rag = get_rag_service()
        self._llm = get_llm_service()

    # ── CV-only pipeline  (POST /predict) ────────────────────────────────

    def run_cv_pipeline(
        self,
        image_source: Any,
        request_id: Optional[str] = None,
    ) -> PredictionResponse:
        """
        Execute the CV-only pipeline.

        Args:
            image_source: Image file path (str) or raw bytes.
            request_id:   Correlation ID; auto-generated when omitted.

        Returns:
            A fully populated
            :class:`~backend.schemas.responses.PredictionResponse`.

        Raises:
            :class:`~backend.core.exceptions.ImageProcessingError`:
                If the image cannot be loaded or decoded.
            :class:`~backend.core.exceptions.PredictionError`:
                If model inference fails.
        """
        t0         = time.perf_counter()
        request_id = request_id or generate_request_id()
        logger.info("[%s] Starting CV pipeline.", request_id)

        cv_raw     = self._cv.analyze_skin(image_source)
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        logger.info("[%s] CV pipeline complete in %.0f ms.", request_id, elapsed_ms)

        return PredictionResponse(
            request_id=         request_id,
            cv_output=          self._build_cv_output(cv_raw),
            processing_time_ms= round(elapsed_ms, 2),
            metadata=           {},
        )

    # ── Full pipeline  (POST /analyze) ───────────────────────────────────

    def run_full_pipeline(
        self,
        image_source: Any,
        user_context: Optional[Dict[str, Any]] = None,
        request_id:   Optional[str]            = None,
        enable_rag:   bool                     = True,
        enable_llm:   bool                     = True,
    ) -> FullAnalysisResponse:
        """
        Execute the full CV → RAG → LLM pipeline.

        Args:
            image_source:  Image file path (str) or raw bytes.
            user_context:  Serialised
                           :class:`~backend.schemas.requests.SkinConcernsInput`
                           dict (keys lowercased, ``None`` values excluded).
            request_id:    Correlation ID; auto-generated when omitted.
            enable_rag:    When ``True`` (default) the FAISS retrieval step
                           runs.  When ``False`` ``retrieved_knowledge`` in
                           the response will be an empty list.
            enable_llm:    When ``True`` (default) the Groq LLM generation
                           step runs.  When ``False`` ``analysis`` in the
                           response will be ``None``.

        Returns:
            A fully populated
            :class:`~backend.schemas.responses.FullAnalysisResponse`.

        Raises:
            :class:`~backend.core.exceptions.ImageProcessingError`:
                If the image cannot be loaded or decoded.
            :class:`~backend.core.exceptions.PredictionError`:
                If CV model inference fails.
            :class:`~backend.core.exceptions.RAGError`:
                If the FAISS index cannot be loaded or searched.
            Note: LLM failures are handled internally and never propagate —
            the LLM service always returns either a real or a mock response.
        """
        t0         = time.perf_counter()
        request_id = request_id or generate_request_id()
        logger.info("[%s] Starting full analysis pipeline.", request_id)

        # ── Step 1: CV analysis ───────────────────────────────────────────
        logger.info("[%s] Running CV analysis.", request_id)
        cv_raw = self._cv.analyze_skin(image_source)

        # ── Step 2: RAG retrieval ─────────────────────────────────────────
        raw_chunks: List[Dict[str, Any]] = []
        if enable_rag:
            logger.info("[%s] Retrieving RAG knowledge chunks.", request_id)
            raw_chunks = self._rag.retrieve_for_cv_result(
                cv_result=cv_raw,
                user_context=user_context,
            )

        # ── Step 3: LLM generation (never raises — returns mock on failure) ─
        llm_output: Optional[LLMOutput] = None
        if enable_llm:
            logger.info("[%s] Generating LLM analysis via Groq.", request_id)
            llm_output = self._llm.generate_analysis(
                cv_result=        cv_raw,
                knowledge_chunks= raw_chunks,
                user_context=     user_context,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.info("[%s] Full pipeline complete in %.0f ms.", request_id, elapsed_ms)

        return FullAnalysisResponse(
            request_id=          request_id,
            cv_output=           self._build_cv_output(cv_raw),
            retrieved_knowledge= self._build_knowledge_chunks(raw_chunks),
            analysis=            self._build_llm_analysis(llm_output) if llm_output else None,
            processing_time_ms=  round(elapsed_ms, 2),
            metadata=            {},
        )

    # ── Schema assembly helpers ───────────────────────────────────────────

    @staticmethod
    def _build_cv_output(cv_raw: Dict[str, Any]) -> CVOutput:
        """
        Convert the raw :meth:`~backend.services.cv_service.CVService.analyze_skin`
        output dict into a validated :class:`~backend.schemas.responses.CVOutput`.

        Args:
            cv_raw: Dict with keys ``face_detected``, ``skin_issues``,
                    ``skin_tone``, ``bounding_box``.
        """
        skin_issues: List[DetectedSkinIssue] = [
            DetectedSkinIssue(
                name=       issue["name"],
                confidence= issue["confidence"],
                severity=   issue.get("severity"),
            )
            for issue in cv_raw.get("skin_issues", [])
        ]

        tone_raw = cv_raw.get("skin_tone", {})
        skin_tone = SkinToneResult(
            tone_label= tone_raw.get("tone_label", "Unknown"),
            hex_color=  tone_raw.get("hex_color",  "#C68642"),
            undertone=  tone_raw.get("undertone",  "neutral"),
            confidence= tone_raw.get("confidence", 0.0),
        )

        return CVOutput(
            face_detected= cv_raw.get("face_detected", False),
            skin_issues=   skin_issues,
            skin_tone=     skin_tone,
            bounding_box=  cv_raw.get("bounding_box"),
        )

    @staticmethod
    def _build_knowledge_chunks(
        raw: List[Dict[str, Any]],
    ) -> List[RetrievedKnowledgeChunk]:
        """
        Convert raw RAG service dicts into validated
        :class:`~backend.schemas.responses.RetrievedKnowledgeChunk` instances.

        Args:
            raw: List of chunk dicts from
                 :meth:`~backend.services.rag_service.RAGService.retrieve_for_cv_result`.
        """
        return [
            RetrievedKnowledgeChunk(
                chunk_id=        c.get("chunk_id", c.get("id", "unknown")),
                title=           c.get("title"),
                text=            c["text"],
                category=        c.get("category"),
                source=          c.get("source"),
                relevance_score= c.get("relevance_score", 0.0),
            )
            for c in raw
        ]

    @staticmethod
    def _build_llm_analysis(output: LLMOutput) -> LLMAnalysisOutput:
        """
        Convert an :class:`~backend.services.llm_service.LLMOutput` dataclass
        into a validated
        :class:`~backend.schemas.responses.LLMAnalysisOutput`.

        Malformed routine steps are skipped with a warning rather than
        raising an exception, because LLM output can occasionally deviate
        from the expected JSON schema.

        Args:
            output: The :class:`~backend.services.llm_service.LLMOutput`
                    produced by
                    :meth:`~backend.services.llm_service.LLMService.generate_analysis`.
        """
        routine_steps: List[SkincareRoutineStep] = []
        for step_data in output.routine:
            try:
                routine_steps.append(
                    SkincareRoutineStep(
                        step=            step_data["step"],
                        phase=           step_data["phase"],
                        action=          step_data["action"],
                        product_type=    step_data["product_type"],
                        key_ingredients= step_data.get("key_ingredients", []),
                        notes=           step_data.get("notes"),
                    )
                )
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed routine step %s: %s", step_data, exc
                )

        return LLMAnalysisOutput(
            explanation=            output.explanation,
            routine=                routine_steps,
            ingredient_suggestions= output.ingredient_suggestions,
            ingredients_to_avoid=   output.ingredients_to_avoid,
            precautions=            output.precautions,
            generated_by=           output.generated_by,
        )


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    """Return the application-wide :class:`RecommendationService` singleton."""
    global _service
    if _service is None:
        _service = RecommendationService()
    return _service