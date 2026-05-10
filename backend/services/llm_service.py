"""
backend/services/llm_service.py
================================
LLM service for SkinAura — powered by the Groq API.

Status values returned by GroqProvider.status()
------------------------------------------------
  "mock_mode"       — No GROQ_API_KEY configured. Pipeline works via fallback.
                      Health check treats this as healthy (expected dev state).
  "ready"           — Client initialised and Groq key present.
  "not_initialised" — Client has never been used yet (first call pending).
                      Health check treats this as healthy.
  "error: <msg>"    — Initialisation was attempted but failed. Genuinely broken.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from backend.core.config import get_settings
from backend.core.exceptions import LLMServiceError
from backend.core.logger import get_logger

logger   = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMOutput:
    explanation:            str
    routine:                List[Dict[str, Any]]
    ingredient_suggestions: List[str]
    ingredients_to_avoid:   List[str]
    precautions:            List[str]
    generated_by:           str
    latency_ms:             float = field(default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation":            self.explanation,
            "routine":                self.routine,
            "ingredient_suggestions": self.ingredient_suggestions,
            "ingredients_to_avoid":   self.ingredients_to_avoid,
            "precautions":            self.precautions,
            "generated_by":           self.generated_by,
            "latency_ms":             self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseLLMProvider(Protocol):
    def is_available(self) -> bool: ...
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...
    def status(self) -> str: ...


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------


class GroqProvider:
    """
    LLM provider adapter for the Groq Cloud API.

    Status values
    -------------
    "mock_mode"       — GROQ_API_KEY absent; fallback active, not an error.
    "not_initialised" — Key present but client not yet used (lazy init).
    "ready"           — Client initialised successfully.
    "error: <msg>"    — Init attempted but failed (genuinely broken).
    """

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._init_attempted: bool  = False
        self._init_error: Optional[str] = None

    def _ensure_client(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True

        if not settings.has_groq_key:
            self._init_error = "GROQ_API_KEY is not set."
            logger.info(
                "GROQ_API_KEY not configured — LLM service running in mock-fallback mode."
            )
            return

        try:
            from groq import Groq
            self._client = Groq(
                api_key=settings.groq_api_key,
                timeout=settings.groq_timeout_seconds,
            )
            logger.info(
                "Groq client initialised (model: %s, temperature: %.2f, max_tokens: %d).",
                settings.groq_model_name,
                settings.groq_temperature,
                settings.groq_max_tokens,
            )
        except ImportError:
            self._init_error = "The 'groq' package is not installed. Run: pip install groq"
            logger.error(self._init_error)
        except Exception as exc:
            self._init_error = str(exc)
            logger.error("Failed to initialise Groq client: %s", exc)

    def is_available(self) -> bool:
        self._ensure_client()
        return self._client is not None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_client()
        if self._client is None:
            raise LLMServiceError(
                "Groq client is not available.",
                details={"reason": self._init_error or "unknown"},
            )
        try:
            response = self._client.chat.completions.create(
                model=settings.groq_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise LLMServiceError(
                f"Groq API request failed: {exc}",
                details={"model": settings.groq_model_name},
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMServiceError(
                "Groq returned an empty choices list.",
                details={"model": settings.groq_model_name},
            )

        content: str = choices[0].message.content or ""
        if not content.strip():
            raise LLMServiceError(
                "Groq returned empty message content.",
                details={"model": settings.groq_model_name},
            )
        return content

    def status(self) -> str:
        """
        Return a short status string for the health endpoint.

        Mapping
        -------
        No key configured        → "mock_mode"       (healthy — expected dev state)
        Key present, not used    → "not_initialised" (healthy — lazy init pending)
        Key present, init OK     → "ready"           (healthy)
        Key present, init failed → "error: <msg>"    (degraded)
        """
        if not settings.has_groq_key:
            return "mock_mode"
        if not self._init_attempted:
            return "not_initialised"
        if self._client is not None:
            return "ready"
        return f"error: {self._init_error}"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = (
    "You are SkinAura, a professional AI skincare consultant. "
    "You provide evidence-based, compassionate skincare advice. "
    "You MUST respond with a single valid JSON object and nothing else — "
    "no markdown fences, no prose outside the JSON."
)

_JSON_SCHEMA_HINT: str = """
{
  "explanation": "<2-3 paragraph analysis of findings for a general audience>",
  "routine": [
    {
      "step": 1,
      "phase": "morning",
      "action": "<imperative step name>",
      "product_type": "<product category>",
      "key_ingredients": ["<ingredient>"],
      "notes": "<optional usage tip>"
    }
  ],
  "ingredient_suggestions": ["<ingredient>"],
  "ingredients_to_avoid": ["<ingredient or class>"],
  "precautions": ["<safety note>"]
}
""".strip()


def _build_prompts(
    cv_result:        Dict[str, Any],
    knowledge_chunks: List[Dict[str, Any]],
    user_context:     Optional[Dict[str, Any]],
) -> tuple[str, str]:
    skin_issues = cv_result.get("skin_issues", [])
    skin_tone   = cv_result.get("skin_tone", {})

    issues_lines = "\n".join(
        f"  • {i['name'].replace('_', ' ').title()} "
        f"({i.get('confidence', 0):.0%} confidence, {i.get('severity', 'unknown')} severity)"
        for i in skin_issues
    ) or "  • No significant issues detected"

    tone_line = (
        f"{skin_tone.get('tone_label', 'unknown')}  |  "
        f"Undertone: {skin_tone.get('undertone', 'unknown')}  |  "
        f"Hex: {skin_tone.get('hex_color', 'n/a')}"
    )

    knowledge_lines = "\n\n".join(
        f"[{c.get('title', 'Tip')}]\n{c.get('text', '')}"
        for c in knowledge_chunks[:4]
    ) or "No relevant knowledge retrieved."

    ctx_lines: List[str] = []
    if user_context:
        if user_context.get("skin_type"):
            ctx_lines.append(f"  • Skin type (self-reported): {user_context['skin_type']}")
        if user_context.get("known_concerns"):
            ctx_lines.append(f"  • User concerns: {', '.join(user_context['known_concerns'])}")
        if user_context.get("age_group"):
            ctx_lines.append(f"  • Age group: {user_context['age_group']}")
        if user_context.get("routine_preference"):
            ctx_lines.append(f"  • Routine preference: {user_context['routine_preference']}")
        if user_context.get("free_text_query"):
            ctx_lines.append(f"  • Note from user: {user_context['free_text_query']}")

    context_section = (
        "USER CONTEXT:\n" + "\n".join(ctx_lines)
        if ctx_lines else "USER CONTEXT: none provided"
    )

    user_prompt = f"""
COMPUTER VISION RESULTS:
Detected issues:
{issues_lines}

Skin tone: {tone_line}

{context_section}

RETRIEVED SKINCARE KNOWLEDGE:
{knowledge_lines}

TASK:
Generate a personalised skincare analysis. Cover:
1. explanation  — warm, clear 2-3 paragraph summary of findings
2. routine      — morning AND evening steps (weekly steps if relevant)
3. ingredient_suggestions — recommended actives for this skin profile
4. ingredients_to_avoid  — ingredients/classes to steer clear of
5. precautions  — safety notes and patch-test guidance

Respond with a single JSON object matching this schema exactly:
{_JSON_SCHEMA_HINT}
""".strip()

    return _SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

def _build_mock_output(reason: str) -> LLMOutput:
    return LLMOutput(
        explanation=(
            "Your skin analysis is complete. Based on the computer vision results, "
            "we've identified characteristics that can be addressed with a consistent "
            "targeted routine.\n\n"
            "Your skin tone and undertone have been mapped to help guide product "
            "shade selection and ingredient compatibility.\n\n"
            f"⚠ Live AI analysis unavailable ({reason}). "
            "Add a valid GROQ_API_KEY to .env for personalised Groq-powered recommendations."
        ),
        routine=[
            {
                "step": 1, "phase": "morning",
                "action": "Gentle Cleanse",
                "product_type": "Gel or low-pH foam cleanser",
                "key_ingredients": ["glycerin", "panthenol", "sodium PCA"],
                "notes": "Lukewarm water only — hot water disrupts the barrier.",
            },
            {
                "step": 2, "phase": "morning",
                "action": "Hydrating Toner",
                "product_type": "Alcohol-free essence or toner",
                "key_ingredients": ["niacinamide 4-5%", "hyaluronic acid", "beta-glucan"],
                "notes": "Pat in with palms — do not rub.",
            },
            {
                "step": 3, "phase": "morning",
                "action": "Lightweight Moisturise",
                "product_type": "Gel-cream or emulsion",
                "key_ingredients": ["ceramides", "squalane", "cholesterol"],
                "notes": "Apply to slightly damp skin for best absorption.",
            },
            {
                "step": 4, "phase": "morning",
                "action": "Broad-Spectrum SPF",
                "product_type": "Sunscreen SPF 50+",
                "key_ingredients": ["zinc oxide", "titanium dioxide"],
                "notes": "Last step every morning. Reapply every 2 hours outdoors.",
            },
            {
                "step": 5, "phase": "evening",
                "action": "Double Cleanse",
                "product_type": "Oil / balm cleanser followed by gel cleanser",
                "key_ingredients": ["jojoba oil", "caprylic/capric triglyceride"],
                "notes": "First cleanse removes SPF and pollution; second cleanse cleans skin.",
            },
            {
                "step": 6, "phase": "evening",
                "action": "Targeted Treatment",
                "product_type": "Serum (introduce gradually)",
                "key_ingredients": ["retinol 0.1-0.3%", "azelaic acid", "vitamin C"],
                "notes": "Use one active at a time; start 2x per week.",
            },
            {
                "step": 7, "phase": "evening",
                "action": "Rich Night Moisturise",
                "product_type": "Cream or sleeping mask",
                "key_ingredients": ["ceramides", "peptides", "shea butter"],
                "notes": "Seals in all previous layers overnight.",
            },
        ],
        ingredient_suggestions=[
            "Niacinamide", "Hyaluronic Acid", "Ceramides",
            "Squalane", "Panthenol", "Centella Asiatica",
            "Vitamin C (L-Ascorbic Acid)", "SPF 50+",
        ],
        ingredients_to_avoid=[
            "Fragrance / Parfum",
            "Alcohol Denat. (SD Alcohol)",
            "High-concentration AHA without buffer (>10%)",
        ],
        precautions=[
            "Patch-test every new product on the inner arm for 24 hours before full application.",
            "Introduce only one new active ingredient at a time.",
            "Never use retinol and AHA/BHA on the same evening.",
            "SPF is non-negotiable — apply daily regardless of weather or indoor setting.",
            "Consult a dermatologist if symptoms worsen or persist.",
        ],
        generated_by="mock_fallback",
        latency_ms=0.0,
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(raw_text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed: Dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMServiceError(
            "LLM response could not be parsed as JSON.",
            details={"raw_preview": raw_text[:300], "error": str(exc)},
        ) from exc

    required_keys = {
        "explanation", "routine",
        "ingredient_suggestions", "ingredients_to_avoid", "precautions",
    }
    missing = required_keys - parsed.keys()
    if missing:
        raise LLMServiceError(
            f"LLM response is missing required keys: {missing}.",
            details={"keys_present": list(parsed.keys())},
        )
    return parsed


# ---------------------------------------------------------------------------
# LLM Service — public façade
# ---------------------------------------------------------------------------

class LLMService:
    _MAX_RETRIES:   int   = 2
    _RETRY_DELAY_S: float = 1.5

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        self._provider: BaseLLMProvider = provider or GroqProvider()

    def generate_analysis(
        self,
        cv_result:        Dict[str, Any],
        knowledge_chunks: List[Dict[str, Any]],
        user_context:     Optional[Dict[str, Any]] = None,
    ) -> LLMOutput:
        if not self._provider.is_available():
            logger.info("LLM provider unavailable — returning mock output.")
            return _build_mock_output(reason="provider not configured")

        system_prompt, user_prompt = _build_prompts(cv_result, knowledge_chunks, user_context)

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                raw_text   = self._provider.complete(system_prompt, user_prompt)
                parsed     = _parse_response(raw_text)
                latency_ms = (time.perf_counter() - t0) * 1_000

                logger.info(
                    "Groq analysis generated in %.0f ms (attempt %d/%d).",
                    latency_ms, attempt, self._MAX_RETRIES,
                )

                return LLMOutput(
                    explanation=            str(parsed.get("explanation", "")),
                    routine=                list(parsed.get("routine", [])),
                    ingredient_suggestions= list(parsed.get("ingredient_suggestions", [])),
                    ingredients_to_avoid=   list(parsed.get("ingredients_to_avoid", [])),
                    precautions=            list(parsed.get("precautions", [])),
                    generated_by=           settings.groq_model_name,
                    latency_ms=             round(latency_ms, 2),
                )

            except LLMServiceError as exc:
                last_exc = exc
                logger.warning("LLM attempt %d/%d failed: %s", attempt, self._MAX_RETRIES, exc.message)
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY_S)

            except Exception as exc:
                last_exc = exc
                logger.warning("Unexpected error on LLM attempt %d/%d: %s", attempt, self._MAX_RETRIES, exc)
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY_S)

        logger.error(
            "LLM generation failed after %d attempts — using fallback. Last error: %s",
            self._MAX_RETRIES, last_exc,
        )
        return _build_mock_output(reason=f"API error after {self._MAX_RETRIES} attempts")

    def status(self) -> str:
        """Return provider status string for the health endpoint."""
        return self._provider.status()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service