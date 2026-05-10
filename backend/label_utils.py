"""
backend/label_utils.py
======================
Shared label-formatting helpers consumed by:
  - predictor.py
  - severity_engine.py
  - analyze.py / predict.py (API route handlers)
  - frontend rendering helpers

All downstream code should import from here rather than touching
model_loader.CLASS_LABELS directly.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Display-name mapping  (snake_case internal → human-readable)
# ---------------------------------------------------------------------------
_DISPLAY_NAMES: dict[str, str] = {
    "acne":         "Acne",
    "acne_scars":   "Acne Scars",
    "normal":       "Normal",
    "pigmentation": "Pigmentation",
}


def to_display(label: str) -> str:
    """
    Convert an internal snake_case label to its display form.

    Examples
    --------
    >>> to_display("acne_scars")
    'Acne Scars'
    >>> to_display("unknown_label")
    'Unknown Label'          # graceful fallback via title-casing
    """
    return _DISPLAY_NAMES.get(label, label.replace("_", " ").title())


def to_internal(display_or_raw: str) -> str:
    """
    Normalise any incoming label string to lowercase snake_case.

    Accepts:
      "Acne Scars", "acne_scars", "ACNE SCARS", "Acne scars" → "acne_scars"
    """
    return display_or_raw.strip().lower().replace(" ", "_")


def format_prediction_response(
    class_names: list[str],
    probabilities: "list[float] | np.ndarray",
    confidence_threshold: float = 0.55,
) -> dict:
    """
    Build a standardised prediction dict for API responses.

    Parameters
    ----------
    class_names   : ordered list from model_loader.get_class_labels()
    probabilities : softmax output vector (one float per class)
    confidence_threshold : if max prob < this, label is set to "uncertain"

    Returns
    -------
    {
        "label":        "acne",          # internal snake_case
        "display":      "Acne",          # human-readable
        "confidence":   0.87,
        "uncertain":    False,
        "probabilities": {
            "acne": 0.87, "acne_scars": 0.05,
            "normal": 0.04, "pigmentation": 0.04
        }
    }
    """
    import numpy as np

    probs = list(map(float, probabilities))
    pred_idx = int(np.argmax(probs))
    confidence = probs[pred_idx]

    if confidence < confidence_threshold:
        label = "uncertain"
    else:
        label = class_names[pred_idx]

    return {
        "label":         label,
        "display":       to_display(label),
        "confidence":    round(confidence, 4),
        "uncertain":     confidence < confidence_threshold,
        "probabilities": {
            class_names[i]: round(p, 4) for i, p in enumerate(probs)
        },
    }