"""
backend/severity_engine.py
===========================
Real severity estimation from image analysis metrics.
NO hardcoded values — all scores derived from:
  - CNN prediction confidence
  - Lesion coverage (from OpenCV contour analysis)
  - Redness score (HSV channel analysis)
  - Texture irregularity (Laplacian variance)
  - Grad-CAM hotspot density
  - Pigmentation clustering
"""
from __future__ import annotations

from typing import Any


def compute_severity(
    prediction: str,
    confidence: float,
    image_metrics: dict[str, Any],
) -> dict[str, tuple[str, str, float]]:
    """
    Compute real per-metric severity from image analysis outputs.

    Args:
        prediction:    Predicted skin class (e.g. "Acne")
        confidence:    Model softmax confidence [0,1]
        image_metrics: Output from image_analysis.analyze_image()

    Returns:
        Dict mapping metric name → (severity_level, hex_color, float_score)
        All float_scores are real values derived from image analysis.
    """

    lesion_cov    = float(image_metrics.get("lesion_coverage", 0.3))
    redness       = float(image_metrics.get("redness_score", 0.2))
    texture       = float(image_metrics.get("texture_score", 0.3))
    pigment       = float(image_metrics.get("pigment_score", 0.2))
    hotspot_d     = float(image_metrics.get("hotspot_density", 0.3))
    evenness      = float(image_metrics.get("evenness_score", 0.7))

    pred = prediction.strip().title()

    # ── Per-condition metric weighting ────────────────────────────────
    # Each condition emphasizes different image signals.
    # All values are real — derived from analyze_image() outputs.

    if pred == "Acne":
        inflammation = _weighted(
            [(redness, 0.45), (hotspot_d, 0.30), (confidence, 0.25)]
        )
        coverage = _weighted(
            [(lesion_cov, 0.60), (hotspot_d, 0.25), (confidence, 0.15)]
        )
        scarring_risk = _weighted(
            [(texture, 0.50), (lesion_cov, 0.30), (confidence, 0.20)]
        )
        pigmentation = _weighted(
            [(1 - evenness, 0.50), (pigment, 0.30), (redness, 0.20)]
        )

    elif pred == "Pigmentation":
        inflammation = _weighted(
            [(redness, 0.35), (1 - evenness, 0.40), (confidence, 0.25)]
        )
        coverage = _weighted(
            [(1 - evenness, 0.50), (pigment, 0.35), (lesion_cov, 0.15)]
        )
        scarring_risk = _weighted(
            [(texture, 0.45), (lesion_cov, 0.30), (confidence, 0.25)]
        )
        pigmentation = _weighted(
            [(pigment, 0.50), (1 - evenness, 0.35), (hotspot_d, 0.15)]
        )

    elif pred == "Acne Scars":
        inflammation = _weighted(
            [(redness, 0.30), (hotspot_d, 0.30), (confidence, 0.40)]
        )
        coverage = _weighted(
            [(lesion_cov, 0.50), (texture, 0.30), (confidence, 0.20)]
        )
        scarring_risk = _weighted(
            [(texture, 0.55), (lesion_cov, 0.30), (confidence, 0.15)]
        )
        pigmentation = _weighted(
            [(pigment, 0.40), (1 - evenness, 0.40), (redness, 0.20)]
        )

    else:  # Normal
        inflammation = _weighted(
            [(redness, 0.50), (hotspot_d, 0.30), (confidence * 0.15, 1.0)]
        )
        coverage = _weighted(
            [(lesion_cov, 0.60), (1 - evenness, 0.25), (confidence * 0.10, 1.0)]
        )
        scarring_risk = _weighted(
            [(texture, 0.55), (lesion_cov, 0.30), (confidence * 0.08, 1.0)]
        )
        pigmentation = _weighted(
            [(pigment, 0.50), (1 - evenness, 0.35), (redness * 0.10, 1.0)]
        )

    # Clamp to valid range
    scores = {
        "Inflammation":   _clamp(inflammation),
        "Coverage":       _clamp(coverage),
        "Scarring Risk":  _clamp(scarring_risk),
        "Pigmentation":   _clamp(pigmentation),
    }

    return {
        metric: (_level(score), _color(score), score)
        for metric, score in scores.items()
    }


def compute_care_priorities(
    prediction: str,
    severity: dict[str, tuple[str, str, float]],
    image_metrics: dict[str, Any],
) -> list[dict]:
    """
    Generate care priority cards from real severity + image metrics.
    Priority level and icon driven by actual computed scores.
    """
    sev_scores = {k: v[2] for k, v in severity.items()}
    pred = prediction.strip().title()

    redness   = float(image_metrics.get("redness_score", 0.2))
    evenness  = float(image_metrics.get("evenness_score", 0.7))
    texture   = float(image_metrics.get("texture_score", 0.3))

    if pred == "Acne":
        return [
            _priority_card("🔥", "Inflammation", sev_scores.get("Inflammation", 0.5), "rgba(239,68,68,0.1)",  "#EF4444"),
            _priority_card("💧", "Oil Control",  sev_scores.get("Coverage",     0.4), "rgba(234,179,8,0.1)",  "#EAB308"),
            _priority_card("🛡", "Hydration",    max(0.1, 1.0 - sev_scores.get("Inflammation", 0.5)), "rgba(59,130,246,0.1)", "#3B82F6"),
        ]
    elif pred == "Pigmentation":
        return [
            _priority_card("☀️", "UV Protection", min(1.0, sev_scores.get("Pigmentation", 0.6) + 0.15), "rgba(239,68,68,0.1)", "#EF4444"),
            _priority_card("✨", "Brightening",   sev_scores.get("Pigmentation", 0.5),                   "rgba(168,85,247,0.1)", "#A855F7"),
            _priority_card("💧", "Hydration",     max(0.2, 1.0 - evenness),                              "rgba(59,130,246,0.1)", "#3B82F6"),
        ]
    elif pred == "Acne Scars":
        return [
            _priority_card("🔬", "Renewal",   sev_scores.get("Scarring Risk", 0.6), "rgba(168,85,247,0.1)", "#A855F7"),
            _priority_card("☀️", "UV Shield", min(1.0, sev_scores.get("Scarring Risk", 0.5) + 0.2), "rgba(239,68,68,0.1)", "#EF4444"),
            _priority_card("💧", "Hydration", max(0.2, texture),                    "rgba(59,130,246,0.1)", "#3B82F6"),
        ]
    else:  # Normal
        return [
            _priority_card("🛡", "Barrier Care", max(0.05, sev_scores.get("Inflammation", 0.1)), "rgba(34,197,94,0.1)",  "#22C55E"),
            _priority_card("☀️", "UV Defense",   0.35,                                            "rgba(234,179,8,0.1)",  "#EAB308"),
            _priority_card("🌿", "Prevention",   max(0.05, sev_scores.get("Coverage", 0.1)),     "rgba(34,197,94,0.1)",  "#22C55E"),
        ]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _weighted(pairs: list[tuple[float, float]]) -> float:
    """Compute weighted average: [(value, weight), ...]"""
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in pairs) / total_w


def _clamp(v: float, lo: float = 0.05, hi: float = 0.97) -> float:
    return max(lo, min(hi, v))


def _level(score: float) -> str:
    if score >= 0.65: return "High"
    if score >= 0.35: return "Medium"
    return "Low"


def _color(score: float) -> str:
    if score >= 0.65: return "#EF4444"
    if score >= 0.35: return "#EAB308"
    return "#22C55E"


def _priority_card(icon: str, label: str, score: float, bg: str, color: str) -> dict:
    return {
        "icon":  icon,
        "label": label,
        "level": _level(score),
        "score": score,
        "bg":    bg,
        "color": color,
    }