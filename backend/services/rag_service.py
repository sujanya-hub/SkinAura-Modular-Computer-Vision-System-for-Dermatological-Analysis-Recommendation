"""
backend/services/rag_service.py
=================================
FAISS-backed knowledge retrieval service for the SkinAura RAG pipeline.

Responsibilities
----------------
- Load the sentence-transformers encoder and FAISS index, either eagerly
  (when index files are already present) or lazily on first retrieval.
- Auto-build the index from the knowledge base if the index files are absent.
- Construct a retrieval query from raw CV service output and optional user
  context.
- Embed the query, search the FAISS index, and return the top-k most
  relevant knowledge chunks.
- Return ``List[Dict[str, Any]]`` whose keys match exactly what
  :class:`~backend.services.recommendation_service.RecommendationService._build_knowledge_chunks`
  and :class:`~backend.schemas.responses.RetrievedKnowledgeChunk` expect.

Initialisation strategy
------------------------
On construction the service checks whether the index files already exist.
If they do, it attempts an **eager** load immediately so that
:meth:`status` returns ``"ready"`` before any retrieval request arrives —
which is what the ``/health`` endpoint needs to see.

If the files do not exist the load is deferred to the first
:meth:`retrieve` / :meth:`retrieve_for_cv_result` call, which will
auto-build the index first.

If an eager load fails (corrupt files, missing libraries, etc.) the error
is logged and the service falls back to lazy initialisation so startup is
not blocked.

Output chunk contract
---------------------
Every dict returned by :meth:`RAGService.retrieve` contains at minimum:

    ``chunk_id``        str   — from knowledge_base.json ``"id"`` field
    ``title``           str | None
    ``text``            str
    ``category``        str | None
    ``source``          str | None
    ``relevance_score`` float — cosine similarity in ``[0.0, 1.0]``

Health-check contract
---------------------
:meth:`RAGService.status` returns ``"ready"`` or ``"not_loaded"``,
feeding the ``rag`` field of
:class:`~backend.schemas.responses.ServiceStatusMap`.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.core.config import get_settings
from backend.core.exceptions import RAGError
from backend.core.logger import get_logger

logger   = get_logger(__name__)
settings = get_settings()

_STATUS_READY:      str = "ready"
_STATUS_NOT_LOADED: str = "not_loaded"


class RAGService:
    """
    FAISS-based knowledge retrieval service.

    Obtain the singleton via :func:`get_rag_service`.  Do not instantiate
    this class directly.

    The service attempts to load the FAISS index eagerly at construction
    time when the index files already exist, so :meth:`status` returns
    ``"ready"`` immediately and the ``/health`` endpoint reflects the true
    readiness state without waiting for the first retrieval request.
    """

    def __init__(self) -> None:
        self._encoder:  Optional[Any]                   = None
        self._index:    Optional[Any]                   = None
        self._metadata: Optional[List[Dict[str, Any]]]  = None
        self._ready:    bool                            = False

        # Attempt an eager load when index files already exist.
        # This makes status() reflect reality at startup without
        # waiting for the first retrieve() call.
        if (
            settings.faiss_index_path.exists()
            and settings.faiss_metadata_path.exists()
        ):
            try:
                self._load_all()
            except Exception as exc:
                # Log but do not raise — fall back to lazy init on first use.
                logger.error(
                    "RAG eager init failed — will retry on first retrieval. "
                    "Reason: %s",
                    exc,
                )

    # ── Internal loaders ─────────────────────────────────────────────────

    def _load_all(self) -> None:
        """
        Load the encoder, FAISS index, and metadata in sequence.

        Sets :attr:`_ready` to ``True`` only when all three steps succeed.

        Raises:
            :class:`~backend.core.exceptions.RAGError`:
                On any loading failure.
        """
        self._load_encoder()
        self._load_index()
        self._ready = True
        logger.info(
            "RAG service ready — %d chunks indexed, embedding model: '%s'.",
            len(self._metadata) if self._metadata else 0,
            settings.embedding_model_name,
        )

    def _auto_build_if_missing(self) -> None:
        """
        Trigger :func:`~backend.rag.build_index.build_index` if either
        index file is absent.

        Raises:
            :class:`~backend.core.exceptions.RAGError`:
                If the build fails.
        """
        if (
            settings.faiss_index_path.exists()
            and settings.faiss_metadata_path.exists()
        ):
            return

        logger.warning(
            "FAISS index not found at %s — building now. "
            "This may take a moment on first run.",
            settings.faiss_index_path,
        )
        try:
            from backend.rag.build_index import build_index  # noqa: PLC0415
            build_index()
        except Exception as exc:
            raise RAGError(
                f"Automatic FAISS index build failed: {exc}",
                details={
                    "faiss_index_path":    str(settings.faiss_index_path),
                    "faiss_metadata_path": str(settings.faiss_metadata_path),
                    "error":               str(exc),
                },
            ) from exc

    def _load_encoder(self) -> None:
        """
        Load the sentence-transformers embedding model.

        Raises:
            :class:`~backend.core.exceptions.RAGError`:
                If the model cannot be loaded.
        """
        logger.info(
            "Loading embedding model '%s' …", settings.embedding_model_name
        )
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._encoder = SentenceTransformer(settings.embedding_model_name)
            logger.info(
                "Embedding model '%s' loaded.", settings.embedding_model_name
            )
        except Exception as exc:
            raise RAGError(
                f"Failed to load embedding model '{settings.embedding_model_name}': {exc}",
                details={
                    "model": settings.embedding_model_name,
                    "error": str(exc),
                },
            ) from exc

    def _load_index(self) -> None:
        """
        Load the FAISS index and metadata pickle from disk.

        Raises:
            :class:`~backend.core.exceptions.RAGError`:
                On any IO or size-mismatch error.
        """
        try:
            import faiss  # noqa: PLC0415
        except ImportError as exc:
            raise RAGError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu",
                details={"error": str(exc)},
            ) from exc

        # ── FAISS index ─────────────────────────────────────────────
        logger.info("Loading FAISS index from %s …", settings.faiss_index_path)
        try:
            self._index = faiss.read_index(str(settings.faiss_index_path))
            logger.info(
                "FAISS index loaded: %d vectors, dim=%d.",
                self._index.ntotal, self._index.d,
            )
        except Exception as exc:
            raise RAGError(
                f"Cannot load FAISS index from {settings.faiss_index_path}: {exc}",
                details={
                    "path":  str(settings.faiss_index_path),
                    "error": str(exc),
                },
            ) from exc

        # ── Metadata pickle ──────────────────────────────────────────
        logger.info(
            "Loading chunk metadata from %s …", settings.faiss_metadata_path
        )
        try:
            with open(settings.faiss_metadata_path, "rb") as fh:
                self._metadata = pickle.load(fh)
            logger.info(
                "Metadata loaded: %d chunks.", len(self._metadata)
            )
        except Exception as exc:
            raise RAGError(
                f"Cannot load RAG metadata from {settings.faiss_metadata_path}: {exc}",
                details={
                    "path":  str(settings.faiss_metadata_path),
                    "error": str(exc),
                },
            ) from exc

        # ── Integrity check ──────────────────────────────────────────
        if self._index.ntotal != len(self._metadata):
            raise RAGError(
                f"FAISS index ({self._index.ntotal} vectors) and metadata "
                f"({len(self._metadata)} chunks) sizes do not match. "
                "Rebuild the index with: python -m backend.rag.build_index --force",
                details={
                    "index_size":    self._index.ntotal,
                    "metadata_size": len(self._metadata),
                },
            )

    def _ensure_ready(self) -> None:
        """
        Guarantee that encoder, index, and metadata are loaded before use.

        Triggers auto-build + load on first retrieval call when the service
        was not successfully initialised eagerly at construction time.

        Raises:
            :class:`~backend.core.exceptions.RAGError`:
                On any loading or build failure.
        """
        if self._ready:
            return

        # Build index if still missing, then load everything.
        self._auto_build_if_missing()
        self._load_all()

    # ── Query construction ────────────────────────────────────────────────

    def _build_query(
        self,
        cv_result:    Dict[str, Any],
        user_context: Optional[Dict[str, Any]],
    ) -> str:
        """
        Construct a natural-language retrieval query from CV output and
        optional user context.

        Args:
            cv_result:    Raw dict from
                          :class:`~backend.services.cv_service.CVService`.
            user_context: Optional serialised
                          :class:`~backend.schemas.requests.SkinConcernsInput`.

        Returns:
            Query string ready for embedding.
        """
        parts: List[str] = []

        # Skin issues (confidence threshold: ≥ 0.10, exclude "clear").
        issues = cv_result.get("skin_issues", [])
        issue_names = [
            i["name"].replace("_", " ")
            for i in issues
            if i.get("confidence", 0.0) >= 0.10 and i.get("name") != "clear"
        ]
        if issue_names:
            parts.append(f"skin concerns: {', '.join(issue_names)}")

        # Skin tone and undertone.
        tone       = cv_result.get("skin_tone", {})
        tone_label = tone.get("tone_label", "")
        undertone  = tone.get("undertone", "")
        if tone_label:
            parts.append(f"skin tone: {tone_label}")
        if undertone:
            parts.append(f"undertone: {undertone}")

        # User-supplied context.
        if user_context:
            skin_type = user_context.get("skin_type")
            if skin_type:
                parts.append(f"skin type: {skin_type}")

            known_concerns = user_context.get("known_concerns") or []
            if known_concerns:
                parts.append(f"user concerns: {', '.join(known_concerns)}")

            free_text = (user_context.get("free_text_query") or "").strip()
            if free_text:
                parts.append(free_text)

        query = "; ".join(parts) if parts else "general skincare routine and maintenance"
        logger.debug("RAG query: '%s'", query)
        return query

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed *query* and return the top-k most relevant knowledge chunks.

        Args:
            query: Free-text retrieval query string.
            top_k: Number of results; defaults to ``settings.rag_top_k``.

        Returns:
            List of chunk dicts sorted by descending ``relevance_score``.
            See module docstring for the key contract.

        Raises:
            :class:`~backend.core.exceptions.RAGError`:
                If initialisation, encoding, or FAISS search fails.
        """
        self._ensure_ready()
        k = top_k or settings.rag_top_k

        # Encode query with the same normalisation used at index build time.
        try:
            query_vec: np.ndarray = self._encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        except Exception as exc:
            raise RAGError(
                f"Query encoding failed: {exc}",
                details={"query_preview": query[:120], "error": str(exc)},
            ) from exc

        # Inner-product search on normalised vectors == cosine similarity.
        try:
            scores, indices = self._index.search(query_vec, k)
        except Exception as exc:
            raise RAGError(
                f"FAISS search failed: {exc}",
                details={"error": str(exc)},
            ) from exc

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self._metadata):
                continue  # FAISS may pad with -1 when fewer results exist.

            chunk = dict(self._metadata[idx])   # shallow copy
            chunk["chunk_id"]        = chunk.get("id", f"chunk_{idx}")
            chunk["relevance_score"] = round(float(max(0.0, min(1.0, score))), 4)
            results.append(chunk)

        logger.debug(
            "Retrieved %d / %d chunks for query: '%.80s'",
            len(results), k, query,
        )
        return results

    def retrieve_for_cv_result(
        self,
        cv_result:    Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build a query from CV output and user context, then retrieve chunks.

        Primary entry-point called by
        :class:`~backend.services.recommendation_service.RecommendationService`.

        Args:
            cv_result:    Raw dict from
                          :class:`~backend.services.cv_service.CVService`.
            user_context: Optional serialised
                          :class:`~backend.schemas.requests.SkinConcernsInput`.
            top_k:        Override for number of results.

        Returns:
            List of chunk dicts matching the
            :class:`~backend.schemas.responses.RetrievedKnowledgeChunk` field
            contract.
        """
        query = self._build_query(cv_result, user_context)
        return self.retrieve(query, top_k=top_k)

    # ── Status reporting ──────────────────────────────────────────────────

    def status(self) -> str:
        """
        Return the service status for the health endpoint.

        Returns:
            ``"ready"``      — index, metadata, and encoder are all loaded.
            ``"not_loaded"`` — service has not yet been successfully initialised.
        """
        return _STATUS_READY if self._ready else _STATUS_NOT_LOADED


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Return the application-wide :class:`RAGService` singleton.

    On first call the singleton is constructed, which attempts an eager
    load of the FAISS index if the files already exist.  Subsequent calls
    return the same instance without re-loading anything.
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


__all__: list[str] = [
    "RAGService",
    "get_rag_service",
]