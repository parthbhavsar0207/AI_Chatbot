"""Pydantic schemas for the grounded academic RAG chat answer generation endpoint."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

from app.rag.evidence_gate import EvidenceDecisionState


class CitationValidationStatus(str, Enum):
    """Status enumeration for citation verification."""

    VALID = "valid"
    REGENERATED_VALID = "regenerated_valid"
    FAILED_REJECTED = "failed_rejected"
    BYPASSED = "bypassed"


class ChatAnswerRequest(BaseModel):
    """Request payload for grounded academic question answering."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language student question. Must not be empty or whitespace only.",
        examples=["What are the prerequisites and credit count for CSE205?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top reranked chunks to supply as evidence context (1 to 20).",
        examples=[5],
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter (e.g. 'syllabus', 'regulations', 'notices', 'faqs').",
        examples=["syllabus"],
    )
    course_code: Optional[str] = Field(
        default=None,
        description="Optional course code filter (e.g. 'CSE202', 'CSE205', 'CSE306').",
        examples=["CSE205"],
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Optional source file filter (e.g. 'CSE205.pdf').",
        examples=["CSE205.pdf"],
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensures query is non-empty and not just whitespace."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")
        return v.strip()


class CitationItem(BaseModel):
    """Structured citation item populated exclusively from retrieved chunk metadata."""

    chunk_id: str = Field(description="Exact deterministic chunk identifier")
    document_id: str = Field(description="Parent document identifier")
    source_file: Optional[str] = Field(default=None, description="Source filename")
    category: Optional[str] = Field(default=None, description="Document category")
    course_code: Optional[str] = Field(default=None, description="Course code, if applicable")
    title: Optional[str] = Field(default=None, description="Chunk section heading or document title")


class SourceMetadataItem(BaseModel):
    """Source document provenance and authenticity metadata."""

    source_file: str = Field(description="Source document filename")
    authority_level: Optional[str] = Field(default=None, description="Authority classification")
    document_origin: Optional[str] = Field(default=None, description="Origin category")
    institutional_authenticity: Optional[str] = Field(default=None, description="Institutional authenticity status")
    verification_status: Optional[str] = Field(default=None, description="Audit verification status")


class CitationValidationResult(BaseModel):
    """Detailed result of programmatic citation extraction and validation."""

    status: CitationValidationStatus = Field(description="Validation outcome status")
    valid_citations_count: int = Field(default=0, description="Count of valid, verified chunk citations")
    invalid_citations: list[str] = Field(default_factory=list, description="List of rejected or fabricated citation tags")
    substantive_statements_count: int = Field(default=0, description="Count of substantive declarative statements analyzed")
    cited_statements_count: int = Field(default=0, description="Count of statements backed by at least one valid citation")
    structural_coverage_passed: bool = Field(default=False, description="Whether structural citation coverage requirements were met")
    semantic_grounding_note: str = Field(
        default=(
            "Structural citation syntax, chunk-ID whitelist, and 100% statement coverage checked only. "
            "Structural validation does not establish semantic correctness or factual entailment."
        ),
        description="Explicit notice that syntactic validation does not establish semantic truth.",
    )


class ChatAnswerResponse(BaseModel):
    """Grounded question answering response model."""

    query: str = Field(description="Original student query")
    answer: str = Field(description="Grounded, decision-conditioned answer text or deterministic abstention")
    evidence_state: EvidenceDecisionState = Field(description="Evaluated evidence decision state")
    citations: list[CitationItem] = Field(default_factory=list, description="Validated citations mapped to chunk metadata")
    source_metadata: list[SourceMetadataItem] = Field(
        default_factory=list,
        description="Authenticity and provenance metadata for sources cited",
    )
    missing_information: Optional[list[str]] = Field(
        default=None,
        description="Specific aspects identified as missing from university documents (for PARTIAL state)",
    )
    conflict_details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Details of contradictory policies identified across documents (for CONFLICTING state)",
    )
    model_identifier: str = Field(description="Identifier of the model used, or 'deterministic_engine'")
    retrieval_latency_ms: float = Field(description="Latency of vector retrieval in milliseconds")
    reranking_latency_ms: float = Field(description="Latency of cross-encoder reranking in milliseconds")
    generation_latency_ms: float = Field(description="Latency of LLM inference in milliseconds (0.0 if bypassed)")
    total_latency_ms: float = Field(description="Total end-to-end request processing latency in milliseconds")
    citation_validation: CitationValidationResult = Field(description="Detailed citation validation evaluation")
    citation_validation_status: CitationValidationStatus = Field(description="High-level citation validation status")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of response generation",
    )
