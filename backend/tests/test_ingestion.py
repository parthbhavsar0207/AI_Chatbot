"""Unit tests for document loaders, text cleaning, metadata generation, and ingestion service."""

import json
from pathlib import Path
import docx
from pypdf import PdfReader, PdfWriter
import pytest

from app.rag.cleaner import clean_text
from app.rag.ingestion import DocumentIngestionService, UnsupportedFileError
from app.rag.loaders.docx_loader import DOCXLoader
from app.rag.loaders.json_loader import JSONLoader
from app.rag.loaders.pdf_loader import PDFLoader, derive_pdf_title, is_footer_only_page
from app.rag.schema import Document


# ---------------------------------------------------------------------------
# 1. Text Cleaning & PDF Column Wrap Tests (FIX 5)
# ---------------------------------------------------------------------------

def test_clean_text_normalizes_whitespace_and_newlines():
    raw = "  CSE205:   Data Structures\r\n\r\n\r\n\r\nUnit I:   Arrays   \n\n  * Item 1   \n"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "CSE205: Data Structures" in cleaned
    assert "Unit I: Arrays" in cleaned
    assert "* Item 1" in cleaned


def test_clean_text_preserves_academic_details():
    academic_text = "Attendance cutoff is 75% for INT335 end-term on 15-Dec-2026."
    cleaned = clean_text(academic_text)
    assert "75%" in cleaned
    assert "INT335" in cleaned
    assert "15-Dec-2026" in cleaned


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text("   \n   \n   ") == ""


def test_clean_text_repairs_pdf_column_wrapping_artifacts():
    # 1. Text\n Books -> Text Books
    raw_tb = "Recommended reading:\nText\n Books\n:\n1\n.\nDATA STRUCTURES by SEYMOUR LIPSCHUTZ"
    cleaned_tb = clean_text(raw_tb)
    assert "Text Books:" in cleaned_tb
    assert "1. DATA STRUCTURES by SEYMOUR LIPSCHUTZ" in cleaned_tb

    # 2. Reference\ns -> References
    raw_ref = "Supplementary material:\nReference\ns\n:\n1\n.\nALGORITHMS by CORMEN"
    cleaned_ref = clean_text(raw_ref)
    assert "References:" in cleaned_ref
    assert "1. ALGORITHMS by CORMEN" in cleaned_ref

    # 3. Course code followed by isolated colon: CSE205\n:\n
    raw_code = "CSE205\n:\nDATA STRUCTURES AND ALGORITHMS"
    cleaned_code = clean_text(raw_code)
    assert "CSE205:\nDATA STRUCTURES AND ALGORITHMS" in cleaned_code

    # 4. Hyphenated line wrap
    raw_hyphen = "We study the algo-\nrithm and its time complexity."
    cleaned_hyphen = clean_text(raw_hyphen)
    assert "algorithm and its time complexity" in cleaned_hyphen


# ---------------------------------------------------------------------------
# 2. Metadata & Document ID Collision Regression Tests (FIX 1)
# ---------------------------------------------------------------------------

def test_document_creation_and_metadata():
    doc = Document.create(
        content="Sample academic regulation clause.",
        source_file="LPU_CSE_Academic_Regulations.docx",
        file_type="docx",
        category="regulations",
        title="Academic Regulations",
        page_number=None,
        headings=["Attendance Policy"],
        extra={"clause": "4.1"},
    )
    assert doc.document_id is not None
    assert len(doc.document_id) == 16
    assert doc.metadata.source_file == "LPU_CSE_Academic_Regulations.docx"
    assert doc.metadata.file_type == "docx"
    assert doc.metadata.category == "regulations"
    assert doc.metadata.title == "Academic Regulations"
    assert doc.metadata.page_number is None
    assert doc.metadata.headings == ["Attendance Policy"]
    assert doc.metadata.extra["clause"] == "4.1"
    assert doc.metadata.extraction_timestamp is not None


def test_document_id_collision_regression_exam_notice():
    """Reproduces the three historical collisions found in the examination notice."""
    source = "lpu_cse_aiml_upcoming_end_term_examinations_2026-27.docx"

    # Collision 1: INT335 vs CSE306 (identical first 100 characters)
    int335_content = (
        "Status: Upcoming\n\nTime: 09:30-12:30\n\nExam type: Objective Type End Term - All MCQ Objective Type\n\n"
        "Exam status/result: Awaited\n\nCategory: Regular\n\nFormat/marks: 60 Multiple Choice Questions of 1 Mark each"
    )
    cse306_content = (
        "Status: Upcoming\n\nTime: 09:30-12:30\n\nExam type: Objective Type End Term - All MCQ Objective Type\n\n"
        "Exam status/result: Awaited\n\nCategory: Regular\n\nFormat/marks: 50 Multiple Choice Questions of 1 Mark each"
    )
    doc_int335 = Document.create(content=int335_content, source_file=source, file_type="docx", category="notices", title="Exam Notice")
    doc_cse306 = Document.create(content=cse306_content, source_file=source, file_type="docx", category="notices", title="Exam Notice")
    assert doc_int335.document_id != doc_cse306.document_id

    # Collision 2: MTH401 vs CSE202 vs CSE276
    mth401_content = (
        "Status: Upcoming\n\nTime: 09:30-12:30\n\nExam type: Theory End Term - Mix MCQ + Subjective\n\n"
        "Exam status/result: Awaited\n\nCategory: Regular\n\nFormat/marks: Part A (30 MCQs), Part B (4 out of 5, 10 marks each)"
    )
    cse202_content = (
        "Status: Upcoming\n\nTime: 09:30-12:30\n\nExam type: Theory End Term - Mix MCQ + Subjective\n\n"
        "Exam status/result: Awaited\n\nCategory: Regular\n\nFormat/marks: Part A (20 MCQs), Part B (Short answer questions)"
    )
    cse276_content = (
        "Status: Upcoming\n\nTime: 09:30-12:30\n\nExam type: Theory End Term - Mix MCQ + Subjective\n\n"
        "Exam status/result: Awaited\n\nCategory: Regular\n\nFormat/marks: Part A (40 MCQs), Part B (3 essay questions)"
    )
    doc_mth = Document.create(content=mth401_content, source_file=source, file_type="docx", category="notices", title="Exam Notice")
    doc_202 = Document.create(content=cse202_content, source_file=source, file_type="docx", category="notices", title="Exam Notice")
    doc_276 = Document.create(content=cse276_content, source_file=source, file_type="docx", category="notices", title="Exam Notice")
    assert len({doc_mth.document_id, doc_202.document_id, doc_276.document_id}) == 3

    # Collision 3: CSE205 Practical vs CSE202 Practical
    p1_content = (
        "Status: Upcoming\n\nTime: 13:30-16:30\n\nExam type: Practical End Term -\n\n"
        "Exam status/result: Awaited\n\nCategory: Regular\n\nSubject: CSE205 Data Structures Lab"
    )
    p2_content = (
        "Status: Upcoming\n\nTime: 13:30-16:30\n\nExam type: Practical End Term -\n\n"
        "Exam status/result: Awaited\n\nCategory: Regular\n\nSubject: CSE202 Object Oriented Programming Lab"
    )
    doc_p1 = Document.create(content=p1_content, source_file=source, file_type="docx", category="notices", title="Exam Notice")
    doc_p2 = Document.create(content=p2_content, source_file=source, file_type="docx", category="notices", title="Exam Notice")
    assert doc_p1.document_id != doc_p2.document_id


def test_document_id_deterministic_stability():
    """Identical source and identical content produce the exact same ID across runs."""
    doc_a = Document.create(content="Exact same content.", source_file="doc.pdf", file_type="pdf", category="syllabus", title="Title", page_number=1)
    doc_b = Document.create(content="Exact same content.", source_file="doc.pdf", file_type="pdf", category="syllabus", title="Title", page_number=1)
    assert doc_a.document_id == doc_b.document_id


# ---------------------------------------------------------------------------
# 3. DOCX Heading Hierarchy Tests (FIX 2)
# ---------------------------------------------------------------------------

def test_docx_heading_hierarchy_nested_and_siblings(tmp_path: Path):
    """Verifies that sibling headings replace each other while nested headings nest properly."""
    doc = docx.Document()
    doc.add_heading("University Academic Guide", level=0)
    doc.add_heading("Section 1: Examinations", level=1)
    doc.add_heading("1.1 Schedule", level=2)
    doc.add_paragraph("Schedule content for 1.1")
    doc.add_heading("1.2 Rules", level=2)
    doc.add_paragraph("Rules content for 1.2")
    doc.add_heading("Section 2: Attendance", level=1)
    doc.add_paragraph("Attendance content for Section 2")

    test_file = tmp_path / "hierarchy_test.docx"
    doc.save(str(test_file))

    loader = DOCXLoader()
    docs = loader.load(test_file)

    assert len(docs) == 3

    # Section 1.1 Schedule should have [Title, Section 1, 1.1 Schedule]
    doc_1_1 = docs[0]
    assert doc_1_1.metadata.headings == ["University Academic Guide", "Section 1: Examinations", "1.1 Schedule"]
    assert "Schedule content for 1.1" in doc_1_1.content

    # Sibling 1.2 Rules should replace 1.1 (NOT accumulate)
    doc_1_2 = docs[1]
    assert doc_1_2.metadata.headings == ["University Academic Guide", "Section 1: Examinations", "1.2 Rules"]
    assert "1.1 Schedule" not in doc_1_2.metadata.headings
    assert "Rules content for 1.2" in doc_1_2.content

    # Higher level Section 2 should pop Section 1 and 1.2
    doc_2 = docs[2]
    assert doc_2.metadata.headings == ["University Academic Guide", "Section 2: Attendance"]
    assert "1.2 Rules" not in doc_2.metadata.headings
    assert "Attendance content for Section 2" in doc_2.content


def test_docx_loader_tables(tmp_path: Path):
    doc = docx.Document()
    doc.add_heading("Exam Table Notice", level=1)
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Course"
    table.rows[0].cells[1].text = "Date"
    table.rows[1].cells[0].text = "INT335"
    table.rows[1].cells[1].text = "15-Dec-2026"
    test_file = tmp_path / "table_test.docx"
    doc.save(str(test_file))

    loader = DOCXLoader()
    docs = loader.load(test_file)
    assert len(docs) >= 1
    assert "|" in docs[0].content
    assert "INT335" in docs[0].content


# ---------------------------------------------------------------------------
# 4. PDF Title & Footer-Only Page Tests (FIX 3 & FIX 4)
# ---------------------------------------------------------------------------

def test_pdf_title_derivation_ignores_placeholders():
    first_page_text = "CSE276\n:\nARTIFICIAL INTELLIGENCE FOUNDATIONS\nCourse Outcomes:\nCO1 :: explain AI"
    # Should ignore generic placeholder 'rdlCourseSyllabusNew' and derive actual title
    derived = derive_pdf_title(first_page_text, course_code="CSE276", default_title="CSE276.pdf")
    assert derived == "CSE276: ARTIFICIAL INTELLIGENCE FOUNDATIONS"

    # Should handle single line format
    single_line_text = "CSE202:OBJECT ORIENTED PROGRAMMING\nCourse Outcomes:\nCO1 :: basic constructs"
    derived_single = derive_pdf_title(single_line_text, course_code="CSE202", default_title="CSE202.pdf")
    assert derived_single == "CSE202:OBJECT ORIENTED PROGRAMMING"


def test_is_footer_only_page_detection():
    # Footer-only samples (should be True)
    assert is_footer_only_page("Session 2026-27 Page: 2/2") is True
    assert is_footer_only_page("Session 20\n26\n-\n27\n Page:\n3\n/\n3") is True
    assert is_footer_only_page("Page: 1 / 1") is True
    assert is_footer_only_page("   \n\n   ") is True

    # Short but meaningful academic pages (should be False)
    assert is_footer_only_page("Passing marks: 40% in end-term examination.") is False
    assert is_footer_only_page("Prerequisites: CSE205 Data Structures.") is False
    assert is_footer_only_page("Credits: 4 (Lecture: 3, Practical: 1).") is False


# ---------------------------------------------------------------------------
# 5. JSON FAQ Loader Tests
# ---------------------------------------------------------------------------

def test_json_faq_loader(tmp_path: Path):
    faq_data = {
        "metadata": {
            "document_title": "LPU CSE-AIML Student FAQs",
            "institution": "Lovely Professional University",
        },
        "faqs": [
            {
                "faq_id": "FAQ101",
                "category": "Examinations",
                "question": "What is the exam date for INT335?",
                "answer": "15 December 2026.",
                "applicable_scope": "INT335 only",
                "verification_status": "verified",
            },
            {
                "faq_id": "FAQ102",
                "category": "Syllabus",
                "question": "Where are syllabus files located?",
                "answer": "In the syllabus folder.",
                "applicable_scope": "CSE-AIML",
                "verification_status": "verified",
            },
        ],
    }
    faq_file = tmp_path / "test_faqs.json"
    with open(faq_file, "w", encoding="utf-8") as f:
        json.dump(faq_data, f)

    loader = JSONLoader()
    docs = loader.load(faq_file)
    assert len(docs) == 2
    assert "FAQ101" in docs[0].metadata.title
    assert "Question: What is the exam date for INT335?" in docs[0].content


# ---------------------------------------------------------------------------
# 6. Service & Error Handling Tests
# ---------------------------------------------------------------------------

def test_unsupported_file_handling(tmp_path: Path):
    unsupported = tmp_path / "data.xyz"
    unsupported.write_text("unsupported content")

    service = DocumentIngestionService()
    with pytest.raises(UnsupportedFileError):
        service.ingest_file(unsupported)

    docs, report = service.ingest_directory(tmp_path)
    assert "data.xyz" in report.unsupported_files
    assert len(docs) == 0


def test_empty_json_handling(tmp_path: Path):
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("[]", encoding="utf-8")
    loader = JSONLoader()
    assert loader.load(empty_json) == []


def test_empty_docx_handling(tmp_path: Path):
    doc = docx.Document()
    empty_docx = tmp_path / "empty.docx"
    doc.save(str(empty_docx))
    loader = DOCXLoader()
    assert loader.load(empty_docx) == []


def test_faq_json_priority_over_docx(tmp_path: Path):
    faq_dir = tmp_path / "faqs"
    faq_dir.mkdir()

    faq_json = faq_dir / "LPU_FAQs.json"
    with open(faq_json, "w", encoding="utf-8") as f:
        json.dump({
            "faqs": [{"question": "Q1", "answer": "A1", "category": "General"}]
        }, f)

    doc = docx.Document()
    doc.add_paragraph("Q1: A1")
    faq_docx = faq_dir / "LPU_FAQs.docx"
    doc.save(str(faq_docx))

    service = DocumentIngestionService()
    docs, report = service.ingest_directory(faq_dir)

    assert len(docs) == 1
    assert docs[0].metadata.file_type == "json"
    assert any("Skipped redundant FAQ docx" in s for s in report.skipped_files)
