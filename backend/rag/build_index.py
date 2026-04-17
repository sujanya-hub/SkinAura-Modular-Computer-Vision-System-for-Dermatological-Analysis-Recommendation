"""
backend/rag/build_index.py
===========================
One-time FAISS index builder for the SkinAura knowledge base.

Usage
-----
Run directly::

    python -m backend.rag.build_index

Or import :func:`build_index` from other modules (e.g. :class:`~backend.services.rag_service.RAGService`
triggers it automatically when the index files are absent)::

    from backend.rag.build_index import build_index
    build_index()

Output
------
Two files are written (paths from ``settings``):

- ``settings.faiss_index_path``    — FAISS ``IndexFlatIP`` serialised to disk.
- ``settings.faiss_metadata_path`` — Pickled ``List[Dict]`` of chunk metadata,
  one entry per row in the FAISS index.

The index uses inner-product similarity on L2-normalised embeddings, which is
equivalent to cosine similarity.  Scores returned by FAISS search are therefore
in ``[0.0, 1.0]`` after normalisation.

Design decisions
----------------
- The encoder is loaded with ``normalize_embeddings=True`` so all vectors are
  on the unit sphere before insertion — no separate normalisation step at
  query time is needed as long as queries are also normalised.
- ``IndexFlatIP`` (exact search) is used rather than an approximate index
  (``IndexIVFFlat``, ``HNSW``).  For a knowledge base of this size (< 1 000
  chunks) exact search is fast enough and avoids the per-query approximation
  error.
- Heavy imports (``sentence_transformers``, ``faiss``) are deferred inside
  :func:`build_index` to keep the module importable in environments where
  those packages are not installed.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure the project root is on sys.path when the script is executed directly
# (``python -m backend.rag.build_index``).
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.config import get_settings  # noqa: E402
from backend.core.exceptions import RAGError   # noqa: E402
from backend.core.logger import get_logger     # noqa: E402

logger   = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def build_index(
    *,
    knowledge_base_path: Path | None = None,
    faiss_index_path:    Path | None = None,
    faiss_metadata_path: Path | None = None,
    force: bool = False,
) -> None:
    """
    Encode all knowledge chunks and persist a FAISS ``IndexFlatIP`` index.

    Args:
        knowledge_base_path: Override the JSON source file (default: ``settings``).
        faiss_index_path:    Override the ``.index`` output path (default: ``settings``).
        faiss_metadata_path: Override the ``.pkl`` metadata output path (default: ``settings``).
        force:               Re-build even if index files already exist.

    Raises:
        :class:`~backend.core.exceptions.RAGError`:
            - If the knowledge base JSON cannot be read or is malformed.
            - If embedding or FAISS operations fail.
    """
    kb_path   = knowledge_base_path or settings.knowledge_base_path
    idx_path  = faiss_index_path    or settings.faiss_index_path
    meta_path = faiss_metadata_path or settings.faiss_metadata_path

    # ── 1. Short-circuit if index already exists ──────────────────────────
    if not force and idx_path.exists() and meta_path.exists():
        logger.info(
            "FAISS index already exists at %s — skipping build.  "
            "Pass force=True to rebuild.",
            idx_path,
        )
        return

    # ── 2. Load and validate the knowledge base ───────────────────────────
    chunks = _load_knowledge_base(kb_path)

    # ── 3. Encode with sentence-transformers ──────────────────────────────
    embeddings = _encode_chunks(chunks)

    # ── 4. Build and persist the FAISS index ─────────────────────────────
    _write_index(embeddings, chunks, idx_path, meta_path)

    logger.info(
        "Index build complete: %d chunks → %s",
        len(chunks), idx_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_knowledge_base(kb_path: Path) -> List[Dict[str, Any]]:
    """
    Load and minimally validate the knowledge base JSON.

    Args:
        kb_path: Absolute path to ``knowledge_base.json``.

    Returns:
        List of chunk dicts.

    Raises:
        :class:`~backend.core.exceptions.RAGError`:
            If the file is missing, not valid JSON, or not a list.
    """
    if not kb_path.exists():
        raise RAGError(
            f"Knowledge base file not found: {kb_path}",
            details={"path": str(kb_path)},
        )

    logger.info("Loading knowledge base from %s …", kb_path)
    try:
        with open(kb_path, encoding="utf-8") as fh:
            chunks: Any = json.load(fh)
    except json.JSONDecodeError as exc:
        raise RAGError(
            f"Knowledge base JSON is malformed: {exc}",
            details={"path": str(kb_path), "error": str(exc)},
        ) from exc

    if not isinstance(chunks, list):
        raise RAGError(
            "Knowledge base JSON must be a top-level array of objects.",
            details={"path": str(kb_path), "type_found": type(chunks).__name__},
        )

    # Validate mandatory fields on each chunk.
    for i, chunk in enumerate(chunks):
        for required_key in ("id", "text"):
            if required_key not in chunk:
                raise RAGError(
                    f"Chunk at index {i} is missing required field '{required_key}'.",
                    details={"index": i, "keys_present": list(chunk.keys())},
                )

    logger.info("Loaded %d knowledge chunks.", len(chunks))
    return chunks


def _encode_chunks(chunks: List[Dict[str, Any]]):
    """
    Encode chunk texts with the configured sentence-transformers model.

    Returns:
        ``numpy.ndarray`` of shape ``(N, D)``, L2-normalised float32.

    Raises:
        :class:`~backend.core.exceptions.RAGError`:
            If the encoder cannot be loaded or encoding fails.
    """
    import numpy as np  # noqa: PLC0415 — deferred to avoid import at startup

    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        raise RAGError(
            "sentence-transformers is not installed.  "
            "Run: pip install sentence-transformers",
            details={"error": str(exc)},
        ) from exc

    logger.info(
        "Loading embedding model '%s' …", settings.embedding_model_name
    )
    try:
        encoder = SentenceTransformer(settings.embedding_model_name)
    except Exception as exc:
        raise RAGError(
            f"Could not load embedding model '{settings.embedding_model_name}': {exc}",
            details={"model": settings.embedding_model_name, "error": str(exc)},
        ) from exc

    texts = [chunk["text"] for chunk in chunks]

    logger.info("Encoding %d texts …", len(texts))
    try:
        embeddings = encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit sphere → cosine via IndexFlatIP
        ).astype("float32")
    except Exception as exc:
        raise RAGError(
            f"Encoding failed: {exc}",
            details={"error": str(exc)},
        ) from exc

    logger.info(
        "Encoding complete: shape=%s, dtype=%s.",
        embeddings.shape, embeddings.dtype,
    )
    return embeddings


def _write_index(
    embeddings,
    chunks: List[Dict[str, Any]],
    idx_path: Path,
    meta_path: Path,
) -> None:
    """
    Build a FAISS ``IndexFlatIP`` from *embeddings* and persist both the
    index and chunk metadata to disk.

    Raises:
        :class:`~backend.core.exceptions.RAGError`:
            If faiss is not installed or writing fails.
    """
    try:
        import faiss  # noqa: PLC0415
    except ImportError as exc:
        raise RAGError(
            "faiss-cpu is not installed.  Run: pip install faiss-cpu",
            details={"error": str(exc)},
        ) from exc

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(
        "FAISS IndexFlatIP built: %d vectors, dimension %d.",
        index.ntotal, dim,
    )

    # Ensure parent directories exist.
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist FAISS index.
    try:
        faiss.write_index(index, str(idx_path))
        logger.info("FAISS index written → %s.", idx_path)
    except Exception as exc:
        raise RAGError(
            f"Failed to write FAISS index to {idx_path}: {exc}",
            details={"path": str(idx_path), "error": str(exc)},
        ) from exc

    # Persist chunk metadata (plain list of dicts — no model objects).
    try:
        with open(meta_path, "wb") as fh:
            pickle.dump(chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Metadata written → %s (%d entries).", meta_path, len(chunks))
    except Exception as exc:
        raise RAGError(
            f"Failed to write metadata to {meta_path}: {exc}",
            details={"path": str(meta_path), "error": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the SkinAura RAG FAISS index from knowledge_base.json."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-build the index even if it already exists.",
    )
    args = parser.parse_args()

    try:
        build_index(force=args.force)
        print("✅  FAISS index built successfully.")
    except RAGError as exc:
        print(f"❌  Build failed: {exc.message}")
        if exc.details:
            print(f"    Details: {exc.details}")
        sys.exit(1)