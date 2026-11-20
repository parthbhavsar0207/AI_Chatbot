"""Pydantic schemas for the retrieval and semantic search API endpoints."""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

from app.rag.evidence_gate import EvidenceDecision

# Canonical set of supported categories across LPU academic corpus
SUPPORTED_CATEGORIES: set[str] = {
    # Core document categories & aliases
    "syllabus",
    "regulations",
    "regulation",
    "policies",
    "policy",
    "notices",
    "notice",
    "examination_notice",
    "examination_notices",
    "faqs",
    "faq",
    # Specific section categories from FAQ metadata
    "attendance",
    "backlog and reappear",
    "examination information",
    "examinations",
    "general",
    "grading",
    "missed examinations",
    "practical examinations",
    "re-evaluation",
}

# Supported academic course codes in LPU CSE-AIML curriculum
SUPPORTED_COURSE_CODES: set[str] = {
    "CSE202",
    "CSE205",
    "CSE276",
    "CSE306",
    "INT335",
    "MTH401",
    "PEL132",
}


class SearchRequest(BaseModel):
    """Request model for semantic retrieval search."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query string. Must not be empty or whitespace only.",
        examples=["What are the passing criteria and attendance requirement for CSE306?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of most relevant chunks to retrieve (1 to 50).",
        examples=[5],
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter (e.g., 'syllabus', 'regulations', 'notices', 'faqs').",
        examples=["syllabus"],
    )
    course_code: Optional[str] = Field(
        default=None,
        description="Optional course code filter (e.g., 'CSE202', 'CSE306', 'INT335').",
        examples=["CSE306"],
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Optional source document filename filter (e.g., 'CSE306.pdf').",
        examples=["CSE306.pdf"],
    )
    enable_rerank: bool = Field(
        default=False,
        description="Whether to apply cross-encoder reranking to improve candidate chunk ordering.",
        examples=[False],
    )
    candidate_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of initial vector candidates retrieved for reranking (default: max(top_k * 2, 10)).",
        examples=[10],
    )
    evaluate_evidence: bool = Field(
        default=False,
        description="Whether to run the evidence gate to determine evidence sufficiency and abstention status.",
        examples=[False],
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensures query is non-empty and not just whitespace."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")
        return v.strip()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        """Validates that category is supported if provided."""
        if v is None:
            return None
        v_clean = v.strip()
        if not v_clean:
            return None
        if v_clean.lower() not in SUPPORTED_CATEGORIES:
            supported_sorted = sorted(list(SUPPORTED_CATEGORIES))
            raise ValueError(
                f"Unsupported category: '{v_clean}'. Supported categories are: {', '.join(supported_sorted)}"
            )
        return v_clean

    @field_validator("course_code")
    @classmethod
    def validate_course_code(cls, v: Optional[str]) -> Optional[str]:
        """Validates that course_code is a recognized course if provided."""
        if v is None:
            return None
        v_clean = v.strip().upper()
        if not v_clean:
            return None
        if v_clean not in SUPPORTED_COURSE_CODES:
            supported_sorted = sorted(list(SUPPORTED_COURSE_CODES))
            raise ValueError(
                f"Invalid course code: '{v_clean}'. Supported courses are: {', '.join(supported_sorted)}"
            )
        return v_clean

    @field_validator("source_file")
    @classmethod
    def validate_source_file(cls, v: Optional[str]) -> Optional[str]:
        """Strips source_file string if provided."""
        if v is None:
            return None
        v_clean = v.strip()
        return v_clean if v_clean else None

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the passing criteria and attendance requirement for CSE306?",
                "top_k": 3,
                "category": "syllabus",
                "course_code": "CSE306",
                "source_file": "CSE306.pdf",
                "enable_rerank": True,
                "candidate_k": 10,
            }
        }
    }


class SearchResultItem(BaseModel):
    """Schema for an individual retrieved chunk item."""

    chunk_id: str = Field(description="Deterministic SHA-256 chunk identifier")
    document_id: str = Field(description="Deterministic SHA-256 parent document identifier")
    text: str = Field(description="Retrieved chunk text content")
    source_file: Optional[str] = Field(default=None, description="Original source document filename")
    category: Optional[str] = Field(default=None, description="Document or section category")
    course_code: Optional[str] = Field(default=None, description="Associated course code if applicable")
    title: Optional[str] = Field(default=None, description="Document or section title")
    similarity: float = Field(description="Cosine similarity score (higher is more relevant)")
    distance: float = Field(description="Cosine distance from vector store (lower is closer)")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved document and chunk metadata attributes",
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder relevance logit score (None if reranking disabled)",
    )
    original_rank: Optional[int] = Field(
        default=None,
        description="Rank of this chunk from initial vector retrieval (1-indexed)",
    )
    reranked_rank: Optional[int] = Field(
        default=None,
        description="Rank of this chunk after cross-encoder reranking (1-indexed; None if reranking disabled)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "chunk_id": "c1f7a0b3e...",
                "document_id": "d4e2a1c0...",
                "text": "Course Title: COMPUTER NETWORKS... Passing Criteria: 40% aggregate...",
                "source_file": "CSE306.pdf",
                "category": "Syllabus",
                "course_code": "CSE306",
                "title": "Computer Networks Syllabus",
                "similarity": 0.84215,
                "distance": 0.15785,
                "reranker_score": 4.1256,
                "original_rank": 2,
                "reranked_rank": 1,
                "metadata": {
                    "source_file": "CSE306.pdf",
                    "category": "Syllabus",
                    "course_code": "CSE306",
                    "token_count": 312,
                },
            }
        }
    }


class SearchResponse(BaseModel):
    """Structured response container for semantic search results."""

    query: str = Field(description="Original user query that was executed")
    top_k: int = Field(description="Number of top chunks requested")
    total_results: int = Field(description="Total number of chunks returned")
    retrieval_time_ms: float = Field(description="Total retrieval search execution time in milliseconds")
    rerank_enabled: bool = Field(default=False, description="Whether cross-encoder reranking was applied")
    candidate_k: Optional[int] = Field(default=None, description="Number of candidate chunks evaluated by reranker")
    evidence_decision: Optional[EvidenceDecision] = Field(
        default=None,
        description="Structured determination of evidence sufficiency (None if evaluate_evidence is False)",
    )
    results: list[SearchResultItem] = Field(
        default_factory=list,
        description="Ranked list of retrieved chunks ordered by relevance",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the attendance policy for examinations?",
                "top_k": 2,
                "total_results": 2,
                "retrieval_time_ms": 28.54,
                "rerank_enabled": True,
                "candidate_k": 10,
                "results": [
                    {
                        "chunk_id": "c1f7a0b...",
                        "document_id": "d4e2a1...",
                        "text": "Students are required to maintain a minimum of 75% attendance...",
                        "source_file": "LPU_CSE_Academic_Regulations.docx",
                        "category": "regulations",
                        "course_code": None,
                        "title": "Attendance Policy",
                        "similarity": 0.81234,
                        "distance": 0.18766,
                        "reranker_score": 5.4512,
                        "original_rank": 1,
                        "reranked_rank": 1,
                        "metadata": {"source_file": "LPU_CSE_Academic_Regulations.docx"},
                    }
                ],
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    detail: str = Field(description="Human-readable error description")
    error_type: Optional[str] = Field(default=None, description="Category or classification of the error")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Query string cannot be empty or whitespace only.",
                "error_type": "ValidationError",
            }
        }
    }
