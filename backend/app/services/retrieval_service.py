"""Service layer for semantic retrieval, handling lifecycle and dependency injection."""

import logging
from pathlib import Path
import time
from typing import Optional

from app.core.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.evidence_gate import EvidenceDecision, EvidenceGate
from app.rag.reranker import CrossEncoderReranker
from app.rag.retrieval import RetrievalService, RetrievedChunk
from app.rag.vector_store import VectorStoreService
from app.schemas.retrieval import SearchRequest, SearchResultItem

logger = logging.getLogger(__name__)


class RetrievalServiceManager:
    """Manages the lifecycle of long-lived singleton RAG service instances.

    Ensures that the ONNX/FastEmbed embedding model, Cross-Encoder reranker,
    EvidenceGate, and ChromaDB client are loaded only once at application startup,
    avoiding per-request reinitialization overhead.
    """

    _instance: Optional["RetrievalServiceManager"] = None
    _retrieval_service: Optional[RetrievalService] = None
    _vector_store: Optional[VectorStoreService] = None
    _embedding_service: Optional[EmbeddingService] = None
    _reranker: Optional[CrossEncoderReranker] = None
    _evidence_gate: Optional[EvidenceGate] = None

    @classmethod
    def get_instance(cls) -> "RetrievalServiceManager":
        """Returns the singleton manager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(
        self,
        persist_directory: Optional[str | Path] = None,
        collection_name: str = "lpu_cse_aiml_docs",
        warmup: bool = False,
    ) -> RetrievalService:
        """Initializes the persistent vector store, embedding service, reranker, and retrieval service.

        Args:
            persist_directory: Directory for ChromaDB storage. Defaults to settings.processed_data_dir/chroma.
            collection_name: Name of the ChromaDB collection.
            warmup: If True, executes dummy inference to pre-warm models into memory.

        Returns:
            The initialized RetrievalService instance.
        """
        if self._retrieval_service is None:
            if persist_directory is None:
                chroma_path = settings.resolved_processed_data_dir / "chroma"
            else:
                chroma_path = Path(persist_directory).resolve()

            if not chroma_path.exists():
                logger.warning(
                    "ChromaDB persistence directory does not exist at: %s. "
                    "Directory will be initialized; queries may return 0 results until indexed.",
                    chroma_path,
                )

            logger.info("Initializing VectorStoreService at: %s", chroma_path)
            self._vector_store = VectorStoreService(
                persist_directory=str(chroma_path),
                collection_name=collection_name,
            )

            logger.info("Initializing EmbeddingService (BAAI/bge-small-en-v1.5)...")
            self._embedding_service = EmbeddingService()

            logger.info("Initializing CrossEncoderReranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
            self._reranker = CrossEncoderReranker()

            logger.info("Initializing EvidenceGate...")
            self._evidence_gate = EvidenceGate()

            if warmup:
                try:
                    logger.info("Warming up models...")
                    self._embedding_service.embed_texts(["warmup"], normalize=True)
                    self._reranker.score_pairs("warmup", ["warmup document"])
                except Exception as exc:
                    logger.warning("Model warmup failed: %s", exc)

            self._retrieval_service = RetrievalService(
                vector_store=self._vector_store,
                embedding_service=self._embedding_service,
                reranker=self._reranker,
                evidence_gate=self._evidence_gate,
            )
            logger.info("RetrievalService successfully initialized with Reranker and EvidenceGate.")

        return self._retrieval_service

    def get_service(self) -> RetrievalService:
        """Returns the singleton RetrievalService, initializing with defaults if necessary."""
        if self._retrieval_service is None:
            return self.initialize()
        return self._retrieval_service

    def reset(self) -> None:
        """Resets the singleton instances (for test isolation)."""
        self._retrieval_service = None
        self._vector_store = None
        self._embedding_service = None
        self._reranker = None
        self._evidence_gate = None


def get_retrieval_service() -> RetrievalService:
    """FastAPI dependency provider for injecting the singleton RetrievalService."""
    return RetrievalServiceManager.get_instance().get_service()


def execute_search(
    request: SearchRequest,
    service: RetrievalService,
) -> tuple[list[SearchResultItem], float, Optional[EvidenceDecision]]:
    """Executes search via RetrievalService and maps raw chunks to API response items.

    Args:
        request: Validated SearchRequest.
        service: Active RetrievalService instance.

    Returns:
        tuple[list[SearchResultItem], float, Optional[EvidenceDecision]]:
            List of mapped result items, retrieval duration in ms, and optional evidence decision.
    """
    start_time = time.perf_counter()

    chunks: list[RetrievedChunk] = service.retrieve(
        query=request.query,
        top_k=request.top_k,
        category=request.category,
        course_code=request.course_code,
        source_file=request.source_file,
        enable_rerank=request.enable_rerank,
        candidate_k=request.candidate_k,
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Optional evidence sufficiency evaluation
    decision: Optional[EvidenceDecision] = None
    if request.evaluate_evidence:
        decision = service.evidence_gate.evaluate(
            query=request.query,
            retrieved_chunks=chunks,
        )

    items: list[SearchResultItem] = []
    for chunk in chunks:
        meta = chunk.metadata or {}
        items.append(
            SearchResultItem(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                source_file=meta.get("source_file"),
                category=meta.get("category"),
                course_code=meta.get("course_code"),
                title=meta.get("title"),
                similarity=chunk.similarity,
                distance=chunk.distance,
                metadata=meta,
                reranker_score=chunk.reranker_score,
                original_rank=chunk.original_rank,
                reranked_rank=chunk.reranked_rank,
            )
        )

    return items, elapsed_ms, decision
