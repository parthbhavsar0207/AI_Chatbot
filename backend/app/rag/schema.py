"""Internal document, chunk, and metadata schemas for the RAG pipeline."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with an extracted document chunk or page."""

    source_file: str = Field(description="Name or relative path of the source file")
    file_type: str = Field(description="File extension/type (e.g., pdf, docx, json)")
    category: str = Field(
        default="general",
        description="Corpus category (e.g., syllabus, regulations, notices, faqs)",
    )
    title: str = Field(description="Title of the document or derived heading")
    page_number: Optional[int] = Field(
        default=None,
        description="Page number for multi-page documents (1-indexed)",
    )
    extraction_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp when extracted",
    )
    headings: list[str] = Field(default_factory=list, description="Section heading hierarchy")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class Document(BaseModel):
    """Standard internal document representation for ingestion and retrieval."""

    document_id: str = Field(description="Unique identifier for the document")
    content: str = Field(description="Cleaned textual content of the document")
    metadata: DocumentMetadata = Field(description="Structured metadata")

    @classmethod
    def create(
        cls,
        content: str,
        source_file: str,
        file_type: str,
        category: str,
        title: str,
        page_number: Optional[int] = None,
        headings: Optional[list[str]] = None,
        extra: Optional[dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> "Document":
        """Factory helper generating deterministic ID if not provided."""
        if not document_id:
            # Deterministic SHA-256 identification using normalized source file, heading/page context, and full content
            norm_source = Path(source_file).name.strip().lower()
            page_part = f":p{page_number}" if page_number is not None else ""
            heading_part = f":{headings[-1].strip().lower()}" if headings else ""
            norm_content = "\n".join(line.rstrip() for line in content.strip().splitlines())
            raw_key = f"{norm_source}{page_part}{heading_part}::{norm_content}"
            document_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

        meta = DocumentMetadata(
            source_file=source_file,
            file_type=file_type.lower().lstrip("."),
            category=category,
            title=title,
            page_number=page_number,
            headings=headings or [],
            extra=extra or {},
        )
        return cls(document_id=document_id, content=content, metadata=meta)


class ChunkMetadata(BaseModel):
    """Metadata attached to an individual searchable chunk."""

    source_file: str = Field(description="Source file name")
    category: str = Field(description="Document category: syllabus, regulations, notices, faqs")
    file_type: str = Field(description="Source file extension")
    title: str = Field(description="Document or section title")
    course_code: Optional[str] = Field(default=None, description="Course code if syllabus")
    page_number: Optional[int] = Field(default=None, description="Page number if PDF")
    headings: list[str] = Field(default_factory=list, description="Section heading hierarchy")
    applicable_scope: Optional[str] = Field(default=None, description="Scope of application")
    verification_status: Optional[str] = Field(default=None, description="Verification status")
    token_count: int = Field(default=0, description="Estimated token count")
    chunk_index: int = Field(default=0, description="Sequential index within parent document")
    total_chunks: int = Field(default=1, description="Total chunks produced from parent document")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class Chunk(BaseModel):
    """Structure-aware text chunk ready for embedding and indexing."""

    chunk_id: str = Field(description="Deterministic unique identifier for the chunk")
    document_id: str = Field(description="ID of the parent Document")
    text: str = Field(description="Text content of the chunk")
    metadata: ChunkMetadata = Field(description="Structured metadata")

    @classmethod
    def create(
        cls,
        document_id: str,
        text: str,
        metadata: ChunkMetadata,
        chunk_id: Optional[str] = None,
    ) -> "Chunk":
        """Factory creating a Chunk with deterministic ID based on document_id, index, and full text."""
        if not chunk_id:
            raw_key = f"{document_id}_c{metadata.chunk_index}::{text.strip()}"
            chunk_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        return cls(chunk_id=chunk_id, document_id=document_id, text=text.strip(), metadata=metadata)


class IngestionReport(BaseModel):
    """Summary report produced after ingesting a file or directory."""

    total_files_scanned: int = 0
    successful_files: list[str] = Field(default_factory=list)
    failed_files: list[dict[str, str]] = Field(default_factory=list)
    unsupported_files: list[str] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    total_documents_extracted: int = 0
    documents_by_category: dict[str, int] = Field(default_factory=dict)
