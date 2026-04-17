"""
backend/core/logger.py
=======================
Centralised logging factory for the SkinAura backend.

Design
------
- A single :func:`_bootstrap_root_logger` call configures the root logger
  exactly once for the process lifetime.  All subsequent calls to
  :func:`get_logger` safely return named child loggers without adding
  duplicate handlers.
- The formatter produces machine-scannable, human-readable lines that work
  well in both local terminals and structured log aggregators (Datadog,
  CloudWatch, etc.).
- Third-party libraries that are excessively noisy at INFO level are
  downgraded to WARNING at bootstrap time.

Usage::

    from backend.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Service initialised.")
    logger.warning("RAG index not found — rebuilding.")
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

from backend.core.config import get_settings

# ---------------------------------------------------------------------------
# Formatting constants
# ---------------------------------------------------------------------------

_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s"
)
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Third-party loggers to suppress to WARNING.
# Add to this tuple as new noisy dependencies are introduced.
# ---------------------------------------------------------------------------
_SUPPRESSED_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "httpcore.http11",
    "httpcore.connection",
    "uvicorn.access",
    "multipart.multipart",
    "PIL.PngImagePlugin",
    "faiss",
)

# Guard: True once the root logger has been configured for this process.
_initialised: bool = False


# ---------------------------------------------------------------------------
# Internal bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_root_logger(level_name: str) -> None:
    """
    Configure the root logger exactly once.

    Subsequent calls are no-ops, making this safe to invoke during module
    import from multiple files without accumulating duplicate handlers.

    Args:
        level_name: A valid :mod:`logging` level name, e.g. ``"INFO"``.
    """
    global _initialised
    if _initialised:
        return

    numeric_level: int = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Only attach our handler if the root logger is currently unconfigured.
    # This prevents duplicate output when pytest or uvicorn has already
    # set up root-level handlers before our application code runs.
    if not root.handlers:
        root.addHandler(handler)

    root.setLevel(numeric_level)

    for name in _SUPPRESSED_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _initialised = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a named :class:`logging.Logger`, bootstrapping the root logger
    on the first call.

    Args:
        name: Dotted module path — pass ``__name__`` from the call site.
              Defaults to ``"skinaura"`` when omitted.

    Returns:
        A fully configured :class:`logging.Logger` instance.

    Example::

        logger = get_logger(__name__)
        logger.debug("Processing image at path: %s", image_path)
    """
    _bootstrap_root_logger(get_settings().log_level)
    return logging.getLogger(name or "skinaura")