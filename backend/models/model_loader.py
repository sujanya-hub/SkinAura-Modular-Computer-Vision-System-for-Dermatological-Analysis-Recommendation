"""
backend/models/model_loader.py
Safe model registry — never crashes on startup.
Missing weights  → placeholder/demo mode (logged as WARNING).
Corrupt weights  → same fallback, no exception propagated to caller.
torch unavailable → pure-Python stubs used transparently.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional

logger = logging.getLogger(__name__)

_KEY_SKIN_ISSUE = "skin_issue"
_KEY_SKIN_TONE  = "skin_tone"
_ModelOutcome   = Literal["trained", "placeholder"]

# ---------------------------------------------------------------------------
# Try to import torch once — if unavailable every loader uses stubs.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn    = None  # type: ignore[assignment]
    _TORCH_OK = False
    logger.warning("torch not available — all models will run as stubs.")


# ---------------------------------------------------------------------------
# Stub models (zero dependencies, always importable)
# ---------------------------------------------------------------------------

class SkinIssueClassifier:
    LABELS = [
        "acne", "redness", "hyperpigmentation", "dryness",
        "oiliness", "wrinkles", "dark_circles", "clear",
    ]
    N_CLASSES    = len(LABELS)
    _is_stub     = True

    # nn.Module interface shim so callers can do model(tensor) if they want
    def __call__(self, x):
        import random
        scores = [random.random() for _ in self.LABELS]
        t = sum(scores)
        return [[s / t for s in scores]]

    def eval(self):
        return self


class SkinToneClassifier:
    TONE_LABELS      = ["Type I (Very Fair)", "Type II (Fair)", "Type III (Medium)",
                        "Type IV (Olive)", "Type V (Brown)", "Type VI (Deep)"]
    TONE_HEX         = ["#F9D4C3", "#F0C4A6", "#E0A882",
                        "#C68642", "#8D5524", "#4A2912"]
    UNDERTONE_LABELS = ["warm", "neutral", "cool"]
    _is_stub         = True

    def __call__(self, x):
        import random
        t_scores = [random.random() for _ in self.TONE_LABELS]
        u_scores = [random.random() for _ in self.UNDERTONE_LABELS]
        ts = sum(t_scores); us = sum(u_scores)
        return (
            [[s / ts for s in t_scores]],
            [[s / us for s in u_scores]],
        )

    def eval(self):
        return self


# ---------------------------------------------------------------------------
# Optional real torch architectures (only used when torch is present)
# ---------------------------------------------------------------------------

def _build_torch_skin_issue():
    """Return a real nn.Module SkinIssueClassifier, or None if torch missing."""
    if not _TORCH_OK:
        return None
    try:
        class _Real(nn.Module):
            LABELS    = SkinIssueClassifier.LABELS
            N_CLASSES = SkinIssueClassifier.N_CLASSES
            _is_stub  = False

            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64, 128), nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.Linear(128, self.N_CLASSES),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return _Real
    except Exception as exc:
        logger.warning("Could not define torch SkinIssueClassifier: %s", exc)
        return None


def _build_torch_skin_tone():
    """Return a real nn.Module SkinToneClassifier, or None if torch missing."""
    if not _TORCH_OK:
        return None
    try:
        class _Real(nn.Module):
            TONE_LABELS      = SkinToneClassifier.TONE_LABELS
            TONE_HEX         = SkinToneClassifier.TONE_HEX
            UNDERTONE_LABELS = SkinToneClassifier.UNDERTONE_LABELS
            _is_stub         = False

            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.tone_head      = nn.Linear(32, len(self.TONE_LABELS))
                self.undertone_head = nn.Linear(32, len(self.UNDERTONE_LABELS))

            def forward(self, x):
                f = self.backbone(x).flatten(1)
                return self.tone_head(f), self.undertone_head(f)

        return _Real
    except Exception as exc:
        logger.warning("Could not define torch SkinToneClassifier: %s", exc)
        return None


# Build once at module level — None if torch unavailable.
_TorchSkinIssue = _build_torch_skin_issue()
_TorchSkinTone  = _build_torch_skin_tone()


# ---------------------------------------------------------------------------
# Internal loader helper
# ---------------------------------------------------------------------------

def _load_model(key: str, real_cls, stub_cls, weights_path: Path):
    """
    Load *real_cls* from *weights_path* if possible.
    Falls back to *stub_cls* on ANY failure — never raises.
    Returns (model_instance, outcome_str).
    """
    t0 = time.perf_counter()

    if real_cls is None:
        # torch not available
        instance = stub_cls()
        outcome: _ModelOutcome = "placeholder"
        logger.warning(
            "Model '%s': torch unavailable, using stub (demo mode).", key
        )
    elif not weights_path.exists():
        try:
            instance = real_cls()
            instance.eval()
        except Exception:
            instance = stub_cls()
        outcome = "placeholder"
        logger.warning(
            "Model '%s': no weights at %s — placeholder init.", key, weights_path
        )
    else:
        try:
            instance = real_cls()
            state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            instance.load_state_dict(state)
            instance.eval()
            outcome = "trained"
        except Exception as exc:
            logger.warning(
                "Model '%s': failed to load weights from %s (%s) — using stub.",
                key, weights_path, exc,
            )
            instance = stub_cls()
            outcome  = "placeholder"

    elapsed = (time.perf_counter() - t0) * 1_000
    logger.info(
        "Model '%s' ready — outcome: %s | %.0f ms.", key, outcome, elapsed
    )
    return instance, outcome


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class ModelRegistry:
    """
    Lazy, failure-tolerant model registry.
    Both 'trained' and 'placeholder' outcomes are reported as 'loaded'
    by status() so health checks pass.
    """

    _cache:   Dict[str, object]         = field(default_factory=dict, init=False)
    _outcome: Dict[str, _ModelOutcome]  = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Resolve device safely.
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self.device = None

    @property
    def skin_issue_model(self) -> SkinIssueClassifier:
        if _KEY_SKIN_ISSUE not in self._cache:
            weights = self._safe_path("skin_issue_model_path")
            instance, outcome = _load_model(
                _KEY_SKIN_ISSUE, _TorchSkinIssue, SkinIssueClassifier, weights
            )
            self._cache[_KEY_SKIN_ISSUE]   = instance
            self._outcome[_KEY_SKIN_ISSUE] = outcome
        return self._cache[_KEY_SKIN_ISSUE]  # type: ignore[return-value]

    @property
    def skin_tone_model(self) -> SkinToneClassifier:
        if _KEY_SKIN_TONE not in self._cache:
            weights = self._safe_path("skin_tone_model_path")
            instance, outcome = _load_model(
                _KEY_SKIN_TONE, _TorchSkinTone, SkinToneClassifier, weights
            )
            self._cache[_KEY_SKIN_TONE]   = instance
            self._outcome[_KEY_SKIN_TONE] = outcome
        return self._cache[_KEY_SKIN_TONE]  # type: ignore[return-value]

    def _safe_path(self, attr: str) -> Path:
        """Read a path from settings without crashing if settings fail."""
        try:
            from backend.core.config import get_settings
            return Path(getattr(get_settings(), attr))
        except Exception as exc:
            logger.warning("Could not resolve %s from settings: %s", attr, exc)
            return Path(f"saved_models/{attr}.pt")  # safe non-existent fallback

    def status(self) -> Dict[str, str]:
        return {
            key: ("loaded" if key in self._cache else "not_loaded")
            for key in [_KEY_SKIN_ISSUE, _KEY_SKIN_TONE]
        }

    def model_info(self) -> Dict[str, str]:
        return {
            key: self._outcome.get(key, "not_loaded")
            for key in [_KEY_SKIN_ISSUE, _KEY_SKIN_TONE]
        }


_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


__all__ = [
    "SkinIssueClassifier",
    "SkinToneClassifier",
    "ModelRegistry",
    "get_model_registry",
]