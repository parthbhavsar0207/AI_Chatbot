"""Modular semantic retrieval service for academic RAG corpus.

Embeds queries with BAAI/bge-small-en-v1.5 and searches the ChromaDB
vector store using cosine distance. Optionally reranks candidates using
a cross-encoder (ms-marco-MiniLM-L-6-v2) for improved relevance ordering.
"""

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Optional
from pydantic import BaseModel, Field

from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# Per-call stage timing telemetry: (vector_retrieval_ms, reranking_ms)
_LAST_RETRIEVAL_TIMINGS: ContextVar[Optional[tuple[float, float]]] = ContextVar(
    "last_retrieval_timings", default=None
)


def reset_last_retrieval_timings() -> None:
    """Resets per-call retrieval stage timings so stale measurements are never reused."""
    _LAST_RETRIEVAL_TIMINGS.set(None)


def get_last_retrieval_timings() -> Optional[tuple[float, float]]:
    """Returns (vector_retrieval_ms, reranking_ms) from the most recent retrieve() call."""
    return _LAST_RETRIEVAL_TIMINGS.get()


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store with similarity scoring and optional reranking."""

    chunk_id: str = Field(description="Unique deterministic chunk ID")
    document_id: str = Field(description="Parent document ID")
    text: str = Field(description="Retrieved chunk text content")
    metadata: dict[str, Any] = Field(description="Preserved chunk metadata")
    distance: float = Field(
        description="Cosine distance from ChromaDB (0.0 = identical, 1.0 = orthogonal, 2.0 = opposite)"
    )
    similarity: float = Field(
        description="Cosine similarity calculated as (1.0 - distance); higher is more similar"
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder relevance score (higher is more relevant; None if reranking disabled)",
    )
    original_rank: Optional[int] = Field(
        default=None,
        description="Rank of this chunk from initial vector retrieval (1-indexed)",
    )
    reranked_rank: Optional[int] = Field(
        default=None,
        description="Rank of this chunk after cross-encoder reranking (1-indexed; None if reranking disabled)",
    )


class RetrievalResponse(BaseModel):
    """Structured response container for query retrieval results."""

    query: str = Field(description="Original user query")
    top_k: int = Field(description="Number of results requested")
    total_retrieved: int = Field(description="Number of chunks retrieved")
    results: list[RetrievedChunk] = Field(default_factory=list, description="Ranked list of retrieved chunks")
    retrieval_time_seconds: float = Field(description="Elapsed search time in seconds")
    rerank_enabled: bool = Field(default=False, description="Whether cross-encoder reranking was applied")
    candidate_k: Optional[int] = Field(default=None, description="Initial candidate count retrieved before reranking")
    rerank_time_seconds: Optional[float] = Field(default=None, description="Time spent on reranking in seconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO timestamp of retrieval",
    )


class RetrievalService:
    """Modular semantic search and retrieval service using FastEmbed + ChromaDB + Cross-Encoder."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        reranker: Optional[Any] = None,
        evidence_gate: Optional[Any] = None,
        default_top_k: int = 5,
    ):
        self.vector_store = vector_store or VectorStoreService()
        self.embedding_service = embedding_service or EmbeddingService()
        self._reranker = reranker
        self._evidence_gate = evidence_gate
        self.default_top_k = default_top_k

    @property
    def evidence_gate(self) -> Any:
        """Lazy-loaded EvidenceGate instance."""
        if self._evidence_gate is None:
            from app.rag.evidence_gate import EvidenceGate
            self._evidence_gate = EvidenceGate()
        return self._evidence_gate

    @property
    def reranker(self) -> Any:
        """Lazy-loaded CrossEncoderReranker instance."""
        if self._reranker is None:
            from app.rag.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def _build_where_clause(
        self,
        category: Optional[str] = None,
        course_code: Optional[str] = None,
        source_file: Optional[str] = None,
        custom_where: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Constructs a validated ChromaDB where filter dict supporting category, course, and source."""
        conditions: list[dict[str, Any]] = []

        if category is not None:
            if not isinstance(category, str):
                raise ValueError(f"category filter must be a string, got: {type(category).__name__}")
            cat_clean = category.strip()
            if cat_clean:
                cat_lower = cat_clean.lower()
                if cat_lower in ("faq", "faqs"):
                    conditions.append({"source_file": "LPU_CSE_AIML_Student_FAQs.json"})
                elif cat_lower == "syllabus":
                    conditions.append({"category": {"$in": ["syllabus", "Syllabus"]}})
                elif cat_lower in ("regulations", "regulation", "policies", "policy"):
                    conditions.append({"category": "regulations"})
                elif cat_lower in ("notices", "notice", "examination_notice", "examination_notices"):
                    conditions.append({"category": "notices"})
                else:
                    conditions.append({"category": cat_clean})

        if course_code is not None:
            if not isinstance(course_code, str):
                raise ValueError(f"course_code filter must be a string, got: {type(course_code).__name__}")
            code_clean = course_code.strip().upper()
            if code_clean:
                conditions.append({"course_code": code_clean})

        if source_file is not None:
            if not isinstance(source_file, str):
                raise ValueError(f"source_file filter must be a string, got: {type(source_file).__name__}")
            file_clean = source_file.strip()
            if file_clean:
                conditions.append({"source_file": file_clean})

        if custom_where:
            conditions.append(custom_where)

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _vector_search(
        self,
        query: str,
        k: int,
        category: Optional[str] = None,
        course_code: Optional[str] = None,
        source_file: Optional[str] = None,
        where: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """Low-level vector search against ChromaDB using query embedding."""
        chroma_where = self._build_where_clause(
            category=category,
            course_code=course_code,
            source_file=source_file,
            custom_where=where,
        )

        query_vectors = self.embedding_service.embed_texts([query], normalize=True)
        if len(query_vectors) == 0:
            return []
        query_vec = query_vectors[0].tolist()

        raw = self.vector_store.query(
            query_embedding=query_vec,
            top_k=k,
            where=chroma_where,
        )

        results: list[RetrievedChunk] = []
        ids_batch = raw.get("ids", [[]])[0]
        docs_batch = raw.get("documents", [[]])[0]
        metas_batch = raw.get("metadatas", [[]])[0]
        dists_batch = raw.get("distances", [[]])[0]

        for idx, (chunk_id, text, meta, dist) in enumerate(zip(ids_batch, docs_batch, metas_batch, dists_batch)):
            meta_copy = dict(meta or {})
            if "headings" in meta_copy and isinstance(meta_copy["headings"], str):
                try:
                    meta_copy["headings"] = json.loads(meta_copy["headings"])
                except Exception:
                    pass

            doc_id = meta_copy.get("document_id", "")
            dist_val = float(dist)
            sim_val = round(1.0 - dist_val, 6)

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    text=text,
                    metadata=meta_copy,
                    distance=round(dist_val, 6),
                    similarity=sim_val,
                    original_rank=idx + 1,
                )
            )

        return results

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        course_code: Optional[str] = None,
        source_file: Optional[str] = None,
        where: Optional[dict[str, Any]] = None,
        enable_rerank: bool = False,
        candidate_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Performs semantic retrieval, with optional cross-encoder reranking.

        Args:
            query: Non-empty natural language query string.
            top_k: Final number of chunks to return (must be >= 1).
            category: Optional category filter.
            course_code: Optional course code filter.
            source_file: Optional source file filter.
            where: Custom ChromaDB where filter dict.
            enable_rerank: If True, reranks candidate chunks with CrossEncoder.
            candidate_k: Number of initial candidates to retrieve if reranking (default: max(top_k * 2, 10)).

        Returns:
            list[RetrievedChunk]: Ranked retrieved chunks.

        Raises:
            ValueError: If query is empty, top_k is invalid, or filter types are invalid.
        """
        # Reset timing telemetry immediately so an exception can never leave stale timings
        _LAST_RETRIEVAL_TIMINGS.set(None)

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        effective_top_k = top_k if top_k is not None else self.default_top_k
        if not isinstance(effective_top_k, int) or effective_top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {effective_top_k}")

        clean_query = query.strip()

        if not enable_rerank:
            # Baseline single-stage vector retrieval
            t_vec_start = time.perf_counter()
            results = self._vector_search(
                query=clean_query,
                k=effective_top_k,
                category=category,
                course_code=course_code,
                source_file=source_file,
                where=where,
            )
            vec_ms = round((time.perf_counter() - t_vec_start) * 1000.0, 2)
            _LAST_RETRIEVAL_TIMINGS.set((vec_ms, 0.0))
            return results

        # Two-stage retrieval: Initial vector retrieval of candidate_k, then cross-encoder rerank
        if candidate_k is not None:
            if not isinstance(candidate_k, int) or candidate_k <= 0:
                raise ValueError(f"candidate_k must be a positive integer, got: {candidate_k}")
            effective_candidates = max(candidate_k, effective_top_k)
        else:
            effective_candidates = max(effective_top_k * 2, 10)

        t_vec_start = time.perf_counter()
        initial_chunks = self._vector_search(
            query=clean_query,
            k=effective_candidates,
            category=category,
            course_code=course_code,
            source_file=source_file,
            where=where,
        )
        vec_ms = round((time.perf_counter() - t_vec_start) * 1000.0, 2)

        if not initial_chunks:
            _LAST_RETRIEVAL_TIMINGS.set((vec_ms, 0.0))
            return []

        # Rerank candidates with Cross-Encoder
        t_rerank_start = time.perf_counter()
        reranked = self.reranker.rerank(
            query=clean_query,
            candidates=initial_chunks,
            top_k=effective_top_k,
        )
        rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)
        _LAST_RETRIEVAL_TIMINGS.set((vec_ms, rerank_ms))
        return reranked

    def retrieve_with_response(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        course_code: Optional[str] = None,
        source_file: Optional[str] = None,
        where: Optional[dict[str, Any]] = None,
        enable_rerank: bool = False,
        candidate_k: Optional[int] = None,
    ) -> RetrievalResponse:
        """Retrieves chunks and packages the output into a full RetrievalResponse container."""
        start_time = time.perf_counter()
        results = self.retrieve(
            query=query,
            top_k=top_k,
            category=category,
            course_code=course_code,
            source_file=source_file,
            where=where,
            enable_rerank=enable_rerank,
            candidate_k=candidate_k,
        )
        elapsed = time.perf_counter() - start_time
        effective_k = top_k if top_k is not None else self.default_top_k

        return RetrievalResponse(
            query=query.strip(),
            top_k=effective_k,
            total_retrieved=len(results),
            results=results,
            retrieval_time_seconds=round(elapsed, 4),
            rerank_enabled=enable_rerank,
            candidate_k=candidate_k,
        )
