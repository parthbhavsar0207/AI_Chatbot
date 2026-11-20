"""Structure-aware document chunking service for academic RAG corpus."""

import logging
import re
from typing import Optional

from app.rag.cleaner import clean_text
from app.rag.schema import Chunk, ChunkMetadata, Document

logger = logging.getLogger(__name__)

# Syllabus unit and section split patterns
SYLLABUS_SECTION_SPLIT_REGEX = re.compile(
    r"(?=(?:Course Outcomes:|Unit\s+[IVXLCDM\d]+|Text\s*Books?:|References?:))",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    """Estimates token count using standard word/punctuation token heuristic (~4 chars/token)."""
    if not text:
        return 0
    # Count words and punctuation tokens
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return max(len(tokens), len(text) // 4)


class ChunkerService:
    """Modular structure-aware document chunking service."""

    def __init__(
        self,
        target_tokens: int = 450,
        overlap_tokens: int = 75,
        min_token_threshold: int = 15,
        oversized_token_threshold: int = 600,
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_token_threshold = min_token_threshold
        self.oversized_token_threshold = oversized_token_threshold

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Dispatches a document to its category-specific chunking strategy."""
        cat = document.metadata.category.lower()

        if "faq" in cat:
            return self._chunk_faq(document)
        if "notice" in cat:
            return self._chunk_notice(document)
        if "syllabus" in cat:
            return self._chunk_syllabus(document)
        if "regulation" in cat:
            return self._chunk_regulations(document)

        return self._chunk_general(document)

    def _chunk_faq(self, document: Document) -> list[Chunk]:
        """Preserves each FAQ Q&A pair as a single atomic chunk."""
        meta = self._build_chunk_metadata(
            document=document,
            chunk_index=0,
            total_chunks=1,
            text=document.content,
        )
        return [Chunk.create(document_id=document.document_id, text=document.content, metadata=meta)]

    def _chunk_notice(self, document: Document) -> list[Chunk]:
        """Preserves each examination entry notice as an independent chunk."""
        meta = self._build_chunk_metadata(
            document=document,
            chunk_index=0,
            total_chunks=1,
            text=document.content,
        )
        return [Chunk.create(document_id=document.document_id, text=document.content, metadata=meta)]

    def _chunk_syllabus(self, document: Document) -> list[Chunk]:
        """Structure-aware chunking for syllabus pages preserving course codes, titles, and unit headers."""
        content = document.content
        course_code = document.metadata.extra.get("course_code") or ""
        course_title = document.metadata.title

        # Build context header to prepend to syllabus chunks so course identity is never lost
        header_prefix = f"Course: {course_code} - {course_title}".strip(" -")
        page_str = f" (Page {document.metadata.page_number})" if document.metadata.page_number else ""
        context_header = f"[{header_prefix}{page_str}]"

        # Split along Units, Outcomes, and Reference sections
        raw_sections = [s.strip() for s in SYLLABUS_SECTION_SPLIT_REGEX.split(content) if s.strip()]

        # If no split points were matched or section is small, keep as single chunk with header
        if len(raw_sections) <= 1:
            full_text = f"{context_header}\n\n{content}" if not content.startswith(context_header) else content
            meta = self._build_chunk_metadata(document, 0, 1, full_text)
            return [Chunk.create(document_id=document.document_id, text=full_text, metadata=meta)]

        # Group sections into balanced chunks targeting ~450 tokens
        grouped_chunks: list[str] = []
        current_group: list[str] = []
        current_tokens = estimate_tokens(context_header)

        for sec in raw_sections:
            sec_tokens = estimate_tokens(sec)
            # If a single unit/section exceeds target, split across paragraphs
            if sec_tokens > self.target_tokens:
                if current_group:
                    grouped_text = f"{context_header}\n\n" + "\n\n".join(current_group)
                    grouped_chunks.append(clean_text(grouped_text))
                    current_group = []
                    current_tokens = estimate_tokens(context_header)
                sub_parts = self._split_text_paragraphs(sec, context_header)
                grouped_chunks.extend(sub_parts)
            elif current_group and (current_tokens + sec_tokens > self.target_tokens):
                grouped_text = f"{context_header}\n\n" + "\n\n".join(current_group)
                grouped_chunks.append(clean_text(grouped_text))
                current_group = [sec]
                current_tokens = estimate_tokens(context_header) + sec_tokens
            else:
                current_group.append(sec)
                current_tokens += sec_tokens

        if current_group:
            grouped_text = f"{context_header}\n\n" + "\n\n".join(current_group)
            grouped_chunks.append(clean_text(grouped_text))

        chunks: list[Chunk] = []
        total = len(grouped_chunks)
        for idx, text in enumerate(grouped_chunks):
            meta = self._build_chunk_metadata(document, idx, total, text)
            chunks.append(Chunk.create(document_id=document.document_id, text=text, metadata=meta))

        return chunks

    def _split_text_paragraphs(self, text: str, header: str) -> list[str]:
        """Splits an oversized section across paragraphs while preserving context header."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        result: list[str] = []
        curr: list[str] = []
        curr_tok = estimate_tokens(header)

        for p in paragraphs:
            ptok = estimate_tokens(p)
            if curr and (curr_tok + ptok > self.target_tokens):
                body = "\n\n".join(curr)
                result.append(clean_text(f"{header}\n\n{body}"))
                curr = [p]
                curr_tok = estimate_tokens(header) + ptok
            else:
                curr.append(p)
                curr_tok += ptok

        if curr:
            body = "\n\n".join(curr)
            result.append(clean_text(f"{header}\n\n{body}"))

        return result

    def _chunk_regulations(self, document: Document) -> list[Chunk]:
        """Chunks academic regulations, splitting along paragraphs if section exceeds target tokens."""
        content = document.content
        tokens = estimate_tokens(content)

        if tokens <= self.target_tokens + 50:
            meta = self._build_chunk_metadata(document, 0, 1, content)
            return [Chunk.create(document_id=document.document_id, text=content, metadata=meta)]

        return self._sliding_window_chunks(document)

    def _chunk_general(self, document: Document) -> list[Chunk]:
        """General fallback chunking with paragraph-aware sliding window."""
        content = document.content
        tokens = estimate_tokens(content)

        if tokens <= self.target_tokens:
            meta = self._build_chunk_metadata(document, 0, 1, content)
            return [Chunk.create(document_id=document.document_id, text=content, metadata=meta)]

        return self._sliding_window_chunks(document)

    def _sliding_window_chunks(self, document: Document) -> list[Chunk]:
        """Splits long text across paragraph and sentence boundaries with token overlap."""
        paragraphs = [p.strip() for p in document.content.split("\n\n") if p.strip()]
        chunks_text: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        # Optional breadcrumb prefix from headings
        prefix = ""
        if document.metadata.headings:
            prefix = " > ".join(document.metadata.headings) + "\n\n"
        prefix_tokens = estimate_tokens(prefix)

        for p in paragraphs:
            p_tokens = estimate_tokens(p)

            # If a single paragraph itself exceeds target, split by sentences
            if p_tokens > self.target_tokens:
                sentences = re.split(r"(?<=[.!?])\s+", p)
                for s in sentences:
                    s_tokens = estimate_tokens(s)
                    if current_chunk and (current_tokens + s_tokens > self.target_tokens):
                        chunk_body = " ".join(current_chunk)
                        chunks_text.append(f"{prefix}{chunk_body}".strip())
                        # Overlap: keep last sentence
                        current_chunk = [current_chunk[-1], s] if len(current_chunk) > 1 else [s]
                        current_tokens = prefix_tokens + sum(estimate_tokens(x) for x in current_chunk)
                    else:
                        current_chunk.append(s)
                        current_tokens += s_tokens
            elif current_chunk and (current_tokens + p_tokens > self.target_tokens):
                chunk_body = "\n\n".join(current_chunk)
                chunks_text.append(f"{prefix}{chunk_body}".strip())
                # Overlap: keep last paragraph if it's within overlap budget
                overlap_p = [current_chunk[-1]] if estimate_tokens(current_chunk[-1]) <= self.overlap_tokens else []
                current_chunk = overlap_p + [p]
                current_tokens = prefix_tokens + sum(estimate_tokens(x) for x in current_chunk)
            else:
                current_chunk.append(p)
                current_tokens += p_tokens

        if current_chunk:
            chunk_body = "\n\n".join(current_chunk)
            chunks_text.append(f"{prefix}{chunk_body}".strip())

        chunks: list[Chunk] = []
        total = len(chunks_text)
        for idx, text in enumerate(chunks_text):
            meta = self._build_chunk_metadata(document, idx, total, text)
            chunks.append(Chunk.create(document_id=document.document_id, text=text, metadata=meta))

        return chunks

    def _build_chunk_metadata(
        self,
        document: Document,
        chunk_index: int,
        total_chunks: int,
        text: str,
    ) -> ChunkMetadata:
        """Constructs standardized ChunkMetadata preserving document attributes."""
        course_code = document.metadata.extra.get("course_code")
        applicable_scope = document.metadata.extra.get("applicable_scope")
        verification_status = document.metadata.extra.get("verification_status")

        return ChunkMetadata(
            source_file=document.metadata.source_file,
            category=document.metadata.category,
            file_type=document.metadata.file_type,
            title=document.metadata.title,
            course_code=course_code,
            page_number=document.metadata.page_number,
            headings=list(document.metadata.headings),
            applicable_scope=applicable_scope,
            verification_status=verification_status,
            token_count=estimate_tokens(text),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            extra=dict(document.metadata.extra),
        )

    def chunk_documents(self, documents: list[Document]) -> tuple[list[Chunk], dict]:
        """Chunks a batch of documents and returns chunks along with quality diagnostics."""
        all_chunks: list[Chunk] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        duplicates_prevented = 0

        chunks_by_category: dict[str, int] = {}
        small_chunks = 0
        oversized_chunks = 0

        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            for chunk in doc_chunks:
                # Deduplication check
                if chunk.chunk_id in seen_ids or chunk.text in seen_texts:
                    duplicates_prevented += 1
                    logger.debug(f"Prevented duplicate chunk '{chunk.chunk_id}'")
                    continue

                if not chunk.text.strip():
                    continue

                seen_ids.add(chunk.chunk_id)
                seen_texts.add(chunk.text)
                all_chunks.append(chunk)

                cat = chunk.metadata.category
                chunks_by_category[cat] = chunks_by_category.get(cat, 0) + 1

                toks = chunk.metadata.token_count
                if toks < self.min_token_threshold:
                    small_chunks += 1
                if toks > self.oversized_token_threshold:
                    oversized_chunks += 1

        token_counts = [c.metadata.token_count for c in all_chunks]
        stats = {
            "total_documents_input": len(documents),
            "total_chunks_produced": len(all_chunks),
            "chunks_by_category": chunks_by_category,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
            "small_chunks_count": small_chunks,
            "oversized_chunks_count": oversized_chunks,
            "duplicate_chunks_prevented": duplicates_prevented,
        }

        return all_chunks, stats
