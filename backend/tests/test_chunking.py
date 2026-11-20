"""Unit tests for structure-aware document chunking service."""

from app.rag.chunker import ChunkerService, estimate_tokens
from app.rag.schema import Document, DocumentMetadata


def create_sample_document(
    content: str,
    category: str = "general",
    source_file: str = "sample.docx",
    title: str = "Sample Title",
    course_code: str = None,
    page_number: int = None,
    headings: list[str] = None,
    applicable_scope: str = None,
    verification_status: str = None,
) -> Document:
    meta = DocumentMetadata(
        source_file=source_file,
        file_type="docx",
        category=category,
        title=title,
        page_number=page_number,
        headings=headings or [],
        extra={
            "course_code": course_code,
            "applicable_scope": applicable_scope,
            "verification_status": verification_status,
        },
    )
    return Document(
        document_id="doc_sample_123",
        content=content,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# 1. Strategy Tests
# ---------------------------------------------------------------------------

def test_syllabus_chunking_preserves_course_identity():
    content = (
        "Course Outcomes:\nCO1 :: understand data structures.\nCO2 :: analyze algorithms.\n\n"
        "Unit I\nArrays and Linked Lists: memory layout, insertions, deletions.\n\n"
        "Unit II\nStacks and Queues: applications, circular queues.\n\n"
        "Unit III\nTrees and Binary Search Trees: traversals, height balancing.\n\n"
        "Unit IV\nGraphs: representation, BFS and DFS.\n\n"
        "Unit V\nSorting and Searching: quicksort, mergesort, binary search.\n\n"
        "Unit VI\nHashing: hash functions, collision resolution.\n\n"
        "Text Books:\n1. DATA STRUCTURES by SEYMOUR LIPSCHUTZ\n"
        "References:\n1. ALGORITHMS by CORMEN"
    )
    doc = create_sample_document(
        content=content,
        category="syllabus",
        source_file="CSE205.pdf",
        title="DATA STRUCTURES AND ALGORITHMS",
        course_code="CSE205",
        page_number=1,
    )
    chunker = ChunkerService(target_tokens=60)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for chunk in chunks:
        # Every chunk must retain course header prefix
        assert "CSE205" in chunk.text
        assert "DATA STRUCTURES" in chunk.text
        assert chunk.metadata.course_code == "CSE205"
        assert chunk.metadata.page_number == 1
        assert chunk.metadata.category == "syllabus"


def test_regulations_chunking_splits_long_sections():
    # Long text with multiple distinct paragraphs
    paras = [
        f"Paragraph {i}: Academic regulations governing university attendance require students to maintain a minimum of 75% attendance in theory and laboratory courses to qualify for end-term examinations."
        for i in range(1, 15)
    ]
    content = "\n\n".join(paras)
    doc = create_sample_document(
        content=content,
        category="regulations",
        source_file="LPU_CSE_Academic_Regulations.docx",
        title="Academic Regulations",
        headings=["Attendance Policy", "Debarment Criteria"],
    )

    chunker = ChunkerService(target_tokens=100, overlap_tokens=25)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.category == "regulations"
        assert "Attendance Policy" in chunk.metadata.headings


def test_examination_notice_chunking_atomic_unit():
    content = (
        "1. INT335 — 15 Dec 2026\n\n"
        "Status: Upcoming\nTime: 09:30-12:30\nExam type: Objective Type End Term\n"
        "Category: Regular\nFormat/marks: 60 MCQs of 1 Mark each"
    )
    doc = create_sample_document(
        content=content,
        category="notices",
        source_file="lpu_cse_aiml_upcoming_end_term_examinations_2026-27.docx",
        title="Examination Notice",
        headings=["Schedule entries", "1. INT335 — 15 Dec 2026"],
    )

    chunker = ChunkerService()
    chunks = chunker.chunk_document(doc)

    # Examination entries must remain 1 independent chunk
    assert len(chunks) == 1
    assert "INT335" in chunks[0].text
    assert "15 Dec 2026" in chunks[0].text
    assert chunks[0].metadata.category == "notices"


def test_atomic_faq_chunking():
    content = (
        "Question: When is the INT335 examination?\n"
        "Answer: The examination is scheduled for 15 December 2026 from 09:30 to 12:30.\n"
        "Category: Examinations\n"
        "Applicable Scope: INT335 only"
    )
    doc = create_sample_document(
        content=content,
        category="faqs",
        source_file="LPU_CSE_AIML_Student_FAQs.json",
        title="Student FAQs - FAQ003",
        applicable_scope="INT335 only",
        verification_status="verified",
    )

    chunker = ChunkerService()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == content
    assert chunks[0].metadata.applicable_scope == "INT335 only"
    assert chunks[0].metadata.verification_status == "verified"


# ---------------------------------------------------------------------------
# 2. Quality & Integrity Tests
# ---------------------------------------------------------------------------

def test_metadata_preservation():
    doc = create_sample_document(
        content="Sample content clause.",
        category="regulations",
        source_file="LPU_Regs.docx",
        title="Regs",
        course_code="CSE101",
        page_number=3,
        headings=["Section 1", "Clause A"],
        applicable_scope="B.Tech CSE",
        verification_status="official",
    )
    chunker = ChunkerService()
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    m = chunks[0].metadata
    assert m.source_file == "LPU_Regs.docx"
    assert m.category == "regulations"
    assert m.file_type == "docx"
    assert m.title == "Regs"
    assert m.course_code == "CSE101"
    assert m.page_number == 3
    assert m.headings == ["Section 1", "Clause A"]
    assert m.applicable_scope == "B.Tech CSE"
    assert m.verification_status == "official"
    assert m.token_count > 0


def test_deterministic_chunk_ids():
    doc1 = create_sample_document("Identical content for deterministic ID test.", category="faqs")
    doc2 = create_sample_document("Identical content for deterministic ID test.", category="faqs")
    chunker = ChunkerService()
    chunk1 = chunker.chunk_document(doc1)[0]
    chunk2 = chunker.chunk_document(doc2)[0]
    assert chunk1.chunk_id == chunk2.chunk_id


def test_empty_input_handling():
    doc = create_sample_document("", category="general")
    chunker = ChunkerService()
    chunks, stats = chunker.chunk_documents([doc])
    assert len(chunks) == 0
    assert stats["total_chunks_produced"] == 0


def test_duplicate_prevention_in_batch():
    doc1 = create_sample_document("Duplicate text snippet across two entries.", category="faqs")
    doc2 = create_sample_document("Duplicate text snippet across two entries.", category="faqs")
    chunker = ChunkerService()
    chunks, stats = chunker.chunk_documents([doc1, doc2])
    assert len(chunks) == 1
    assert stats["duplicate_chunks_prevented"] == 1
