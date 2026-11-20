"""Modular Retrieval-Augmented Generation (RAG) components."""

from app.rag.chunker import ChunkerService, estimate_tokens
from app.rag.cleaner import clean_text
from app.rag.embeddings import DEFAULT_MODEL_NAME, DEFAULT_VECTOR_DIM, EmbeddingService
from app.rag.evaluator import (
    EvaluationReport,
    RetrievalEvaluator,
    compute_hit_rate,
    compute_precision,
    compute_recall,
    compute_reciprocal_rank,
)
from app.rag.evidence_gate import EvidenceDecision, EvidenceDecisionState, EvidenceGate
from app.rag.ingestion import DocumentIngestionService, UnsupportedFileError
from app.rag.loaders.base import BaseLoader
from app.rag.loaders.docx_loader import DOCXLoader
from app.rag.loaders.json_loader import JSONLoader
from app.rag.loaders.pdf_loader import PDFLoader
from app.rag.reranker import CrossEncoderReranker
from app.rag.retrieval import RetrievalResponse, RetrievalService, RetrievedChunk
from app.rag.schema import Chunk, ChunkMetadata, Document, DocumentMetadata, IngestionReport
from app.rag.vector_store import VectorStoreService

__all__ = [
    "clean_text",
    "Document",
    "DocumentMetadata",
    "Chunk",
    "ChunkMetadata",
    "IngestionReport",
    "DocumentIngestionService",
    "UnsupportedFileError",
    "BaseLoader",
    "PDFLoader",
    "DOCXLoader",
    "JSONLoader",
    "ChunkerService",
    "EmbeddingService",
    "VectorStoreService",
    "RetrievalService",
    "RetrievedChunk",
    "RetrievalResponse",
    "CrossEncoderReranker",
    "EvidenceGate",
    "EvidenceDecision",
    "EvidenceDecisionState",
    "RetrievalEvaluator",
    "EvaluationReport",
    "compute_hit_rate",
    "compute_recall",
    "compute_precision",
    "compute_reciprocal_rank",
    "estimate_tokens",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_VECTOR_DIM",
]
