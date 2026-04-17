"""
backend/models/model_loader.py
================================
PyTorch model registry for the SkinAura CV pipeline.

Design
------
Every model follows one of three load outcomes, each with distinct logging
and internal status tracking:

    1. **Trained** — a compatible ``.pt`` state-dict was found and restored.
       ``registry.status()`` → ``"loaded"``.
       ``registry.model_info()`` → ``"trained"``.

    2. **Placeholder** — no weights file exists; the randomly-initialised
       architecture is used.  Inference still works but results are not
       meaningful until real weights are dropped into ``saved_models/``.
       ``registry.status()`` → ``"loaded"`` (operational, not broken).
       ``registry.model_info()`` → ``"placeholder"``.

    3. **Failed** — a weights file exists but cannot be loaded (corrupt
       checkpoint, incompatible architecture).
       ``ModelLoadError`` is raised immediately so the caller and health
       check surface the failure explicitly.

Public interfaces preserved
---------------------------
    registry = get_model_registry()
    model    = registry.skin_issue_model        # SkinIssueClassifier
    model    = registry.skin_tone_model         # SkinToneClassifier
    stati    = registry.status()                # Dict[str, "loaded"|"not_loaded"]
    device   = registry.device                  # torch.device

New non-breaking additions
--------------------------
    info = registry.model_info()
    # Dict[str, "trained"|"placeholder"|"not_loaded"]
    # Used internally for richer logging; health.py does not call this.

Class-level label attributes (consumed by cv_service)
------------------------------------------------------
    SkinIssueClassifier.LABELS
    SkinToneClassifier.TONE_LABELS
    SkinToneClassifier.TONE_HEX
    SkinToneClassifier.UNDERTONE_LABELS
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

from backend.core.config import get_settings
from backend.core.exceptions import ModelLoadError
from backend.core.logger import get_logger

logger   = get_logger(__name__)
settings = get_settings()

# Internal registry keys — must match what health.py iterates.
_KEY_SKIN_ISSUE = "skin_issue"
_KEY_SKIN_TONE  = "skin_tone"

# Possible load outcomes tracked internally.
_ModelOutcome = Literal["trained", "placeholder"]


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------


class SkinIssueClassifier(nn.Module):
    """
    Shallow CNN placeholder for a fine-tuned skin-issue classifier.

    Input:  ``(B, 3, H, W)`` float32 in [0, 1].
    Output: ``(B, N_CLASSES)`` raw logits.

    Class attributes are consumed by :mod:`backend.services.cv_service`
    at the class level (no instance required).
    """

    LABELS: list[str] = [
        "acne",
        "redness",
        "hyperpigmentation",
        "dryness",
        "oiliness",
        "wrinkles",
        "dark_circles",
        "clear",
    ]
    N_CLASSES: int = len(LABELS)

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float32 tensor ``(B, 3, H, W)`` with values in [0, 1].

        Returns:
            Logits tensor ``(B, N_CLASSES)``.
        """
        return self.classifier(self.features(x))


class SkinToneClassifier(nn.Module):
    """
    Dual-head CNN placeholder for Fitzpatrick tone and undertone classification.

    Input:  ``(B, 3, H, W)`` float32 in [0, 1].
    Output: ``(tone_logits, undertone_logits)`` — each ``(B, N)``.

    Class attributes are consumed by :mod:`backend.services.cv_service`
    at the class level (no instance required).
    """

    TONE_LABELS: list[str] = [
        "Type I (Very Fair)",
        "Type II (Fair)",
        "Type III (Medium)",
        "Type IV (Olive)",
        "Type V (Brown)",
        "Type VI (Deep)",
    ]
    TONE_HEX: list[str] = [
        "#F9D4C3",  # Type I
        "#F0C4A6",  # Type II
        "#E0A882",  # Type III
        "#C68642",  # Type IV
        "#8D5524",  # Type V
        "#4A2912",  # Type VI
    ]
    UNDERTONE_LABELS: list[str] = ["warm", "neutral", "cool"]

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        feat_dim = 32
        self.tone_head      = nn.Linear(feat_dim, len(self.TONE_LABELS))
        self.undertone_head = nn.Linear(feat_dim, len(self.UNDERTONE_LABELS))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Float32 tensor ``(B, 3, H, W)`` with values in [0, 1].

        Returns:
            ``(tone_logits, undertone_logits)`` — each ``(B, N)``.
        """
        features = self.backbone(x).flatten(1)
        return self.tone_head(features), self.undertone_head(features)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass
class ModelRegistry:
    """
    Lazy-loading registry that caches and tracks PyTorch model instances.

    Obtain the singleton via :func:`get_model_registry`; do not instantiate
    this class directly in application code.

    Load outcomes
    -------------
    Each model is tagged internally as either ``"trained"`` (real weights
    restored) or ``"placeholder"`` (randomly initialised because no weights
    file was found).  Both outcomes mark the model as operationally
    ``"loaded"`` for the health check — only a load *failure* is treated
    as an error.
    """

    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    _cache:   Dict[str, nn.Module]     = field(default_factory=dict, init=False)
    _outcome: Dict[str, _ModelOutcome] = field(default_factory=dict, init=False)

    # ── Internal loader ──────────────────────────────────────────────────

    def _load(
        self,
        key: str,
        model_cls: type[nn.Module],
        weights_path: Path,
    ) -> nn.Module:
        """
        Instantiate *model_cls* and optionally restore weights from *weights_path*.

        Outcomes
        ~~~~~~~~
        - **Weights present and valid** → restored; tagged ``"trained"``.
        - **Weights absent** → random init; tagged ``"placeholder"``; WARNING logged.
        - **Weights present but invalid** → :class:`ModelLoadError` raised; ERROR logged.

        Args:
            key:          Registry key (e.g. ``"skin_issue"``).
            model_cls:    ``nn.Module`` subclass to instantiate.
            weights_path: Expected path to the ``.pt`` state-dict file.

        Returns:
            Model in ``eval()`` mode on :attr:`device`.

        Raises:
            :class:`~backend.core.exceptions.ModelLoadError`:
                If *weights_path* exists but cannot be loaded.
        """
        if key in self._cache:
            return self._cache[key]

        t0    = time.perf_counter()
        model = model_cls()

        if weights_path.exists():
            try:
                state_dict = torch.load(
                    str(weights_path),
                    map_location=self.device,
                    weights_only=True,
                )
                model.load_state_dict(state_dict)
                outcome: _ModelOutcome = "trained"
            except Exception as exc:
                logger.error(
                    "Model '%s': failed to load weights from %s — %s",
                    key, weights_path, exc,
                )
                raise ModelLoadError(
                    f"Cannot load weights for model '{key}'.",
                    details={
                        "key":          key,
                        "weights_path": str(weights_path),
                        "error":        str(exc),
                    },
                ) from exc
        else:
            outcome = "placeholder"
            logger.warning(
                "Model '%s': no weights file at %s. "
                "Running with randomly-initialised placeholder weights. "
                "Drop a trained .pt state-dict at that path to enable real inference.",
                key, weights_path,
            )

        model = model.to(self.device).eval()
        self._cache[key]   = model
        self._outcome[key] = outcome

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.info(
            "Model '%s' ready — outcome: %s | device: %s | load time: %.0f ms.",
            key, outcome, self.device, elapsed_ms,
        )
        return model

    # ── Public model accessors ───────────────────────────────────────────

    @property
    def skin_issue_model(self) -> SkinIssueClassifier:
        """
        Return the skin-issue classifier, loading it on first access.

        Raises:
            :class:`~backend.core.exceptions.ModelLoadError`:
                If a weights file is present but cannot be loaded.
        """
        return self._load(  # type: ignore[return-value]
            key=          _KEY_SKIN_ISSUE,
            model_cls=    SkinIssueClassifier,
            weights_path= settings.skin_issue_model_path,
        )

    @property
    def skin_tone_model(self) -> SkinToneClassifier:
        """
        Return the skin-tone classifier, loading it on first access.

        Raises:
            :class:`~backend.core.exceptions.ModelLoadError`:
                If a weights file is present but cannot be loaded.
        """
        return self._load(  # type: ignore[return-value]
            key=          _KEY_SKIN_TONE,
            model_cls=    SkinToneClassifier,
            weights_path= settings.skin_tone_model_path,
        )

    # ── Status reporting ─────────────────────────────────────────────────

    def status(self) -> Dict[str, str]:
        """
        Return the operational load state for each registered model.

        Consumed by :mod:`backend.api.health` to populate
        :class:`~backend.schemas.responses.ServiceStatusMap`.

        Both ``"trained"`` and ``"placeholder"`` models are reported as
        ``"loaded"`` because both are operationally available for inference.
        Only models that have not yet been accessed report ``"not_loaded"``.

        Returns:
            ``Dict[str, "loaded" | "not_loaded"]``.

            Example::

                {"skin_issue": "loaded", "skin_tone": "loaded"}
        """
        all_keys = [_KEY_SKIN_ISSUE, _KEY_SKIN_TONE]
        return {
            key: ("loaded" if key in self._cache else "not_loaded")
            for key in all_keys
        }

    def model_info(self) -> Dict[str, str]:
        """
        Return the detailed load outcome for each registered model.

        Unlike :meth:`status`, this method distinguishes between models
        loaded with real trained weights and those running on placeholder
        random initialisations.

        Returns:
            ``Dict[str, "trained" | "placeholder" | "not_loaded"]``.

            Example::

                {
                    "skin_issue": "placeholder",
                    "skin_tone":  "placeholder",
                }
        """
        all_keys = [_KEY_SKIN_ISSUE, _KEY_SKIN_TONE]
        return {
            key: self._outcome.get(key, "not_loaded")
            for key in all_keys
        }


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """
    Return the application-wide :class:`ModelRegistry` singleton.

    Thread-safe for reads after initial construction.  The first call
    creates the registry (but does not load any models — loading is lazy).

    Returns:
        The shared :class:`ModelRegistry` instance.
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


__all__: list[str] = [
    "SkinIssueClassifier",
    "SkinToneClassifier",
    "ModelRegistry",
    "get_model_registry",
]