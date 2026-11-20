"""Unit tests for the CitationValidator engine (Phase 10.3)."""

import pytest

from app.rag.citation_validator import CitationValidator
from app.rag.retrieval import RetrievedChunk
from app.schemas.chat import CitationValidationStatus


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="b84e8c8b020a1a9c",
            document_id="LPU_CSE_AIML_Student_FAQs_FAQ001",
            text="CSE205 Data Structures has 4 credits and requires CSE101 or CSE202.",
            metadata={
                "source_file": "LPU_CSE_AIML_Student_FAQs.json",
                "category": "faqs",
                "course_code": "CSE205",
                "title": "Course Prerequisites",
            },
            distance=0.1,
            similarity=0.9,
        ),
        RetrievedChunk(
            chunk_id="6a5ba40b1700e31a",
            document_id="LPU_CSE_Academic_Regulations_Sec4",
            text="Every student must maintain at least 75% attendance in theory and practical courses.",
            metadata={
                "source_file": "LPU_CSE_Academic_Regulations.docx",
                "category": "regulations",
                "title": "Attendance Policy",
            },
            distance=0.15,
            similarity=0.85,
        ),
    ]


def test_valid_citation_extracted_and_mapped(sample_chunks):
    validator = CitationValidator()
    answer = (
        "The course CSE205 has 4 credits and requires CSE101 as a prerequisite [[cite:b84e8c8b020a1a9c]].\n"
        "Students must maintain 75% minimum attendance to be eligible for exams [[cite:6a5ba40b1700e31a]]."
    )

    report = validator.validate(answer, sample_chunks)
    assert report.is_valid is True
    assert report.status == CitationValidationStatus.VALID
    assert "b84e8c8b020a1a9c" in report.valid_chunk_ids
    assert "6a5ba40b1700e31a" in report.valid_chunk_ids
    assert len(report.citation_items) == 2
    assert report.citation_items[0].source_file == "LPU_CSE_AIML_Student_FAQs.json"
    assert report.citation_items[1].source_file == "LPU_CSE_Academic_Regulations.docx"


def test_unknown_or_fabricated_chunk_id_rejected(sample_chunks):
    validator = CitationValidator()
    answer = "The prerequisite for CSE205 is programming [[cite:fabricated_id_99999]]."

    report = validator.validate(answer, sample_chunks)
    assert report.is_valid is False
    assert report.status == CitationValidationStatus.FAILED_REJECTED
    assert any("fabricated_id_99999" in tag for tag in report.invalid_tags)


def test_malformed_bracket_filename_rejected(sample_chunks):
    validator = CitationValidator()
    answer = "The prerequisite for CSE205 is programming [CSE205.pdf]."

    report = validator.validate(answer, sample_chunks)
    assert report.is_valid is False
    assert report.status == CitationValidationStatus.FAILED_REJECTED


def test_zero_citations_rejected(sample_chunks):
    validator = CitationValidator()
    answer = "CSE205 requires Programming in C++ with 4 credits."

    report = validator.validate(answer, sample_chunks)
    assert report.is_valid is False
    assert report.status == CitationValidationStatus.FAILED_REJECTED
    assert "No valid [[cite:chunk_id]] citations found" in report.rejection_reason


def test_abstention_bypasses_citation_validation(sample_chunks):
    validator = CitationValidator()
    abstention = "I do not have sufficient information in the provided documents to answer this question."

    report = validator.validate(abstention, sample_chunks, is_abstention=True)
    assert report.is_valid is True
    assert report.status == CitationValidationStatus.BYPASSED
    assert report.structural_coverage_passed is True


def test_unsupported_claim_alongside_valid_citation_fails_coverage(sample_chunks):
    validator = CitationValidator()
    # 4 distinct substantive claims, but only 1 is cited while 3 are completely ungrounded claims
    answer = (
        "The course CSE205 has 4 credits [[cite:b84e8c8b020a1a9c]].\n"
        "Students are automatically refunded 100% of their tuition if they withdraw within 30 days.\n"
        "All residential hostels provide private rooms for undergraduate sophomores.\n"
        "The university grants free laptop computers to every engineering major."
    )

    report = validator.validate(answer, sample_chunks)
    assert report.is_valid is False
    assert report.status == CitationValidationStatus.FAILED_REJECTED
    assert report.structural_coverage_passed is False
    assert "Insufficient citation coverage" in report.rejection_reason


def test_sentence_splitting_rejects_uncited_sentence_in_same_paragraph(sample_chunks):
    """100% sentence-level coverage: an uncited sentence in the same paragraph must fail."""
    validator = CitationValidator()
    answer = (
        "CSE205 has 4 credits and requires CSE101. "
        "Students must maintain 75% attendance to appear in exams [[cite:6a5ba40b1700e31a]]."
    )
    report = validator.validate(answer, sample_chunks)
    assert report.is_valid is False
    assert report.status == CitationValidationStatus.FAILED_REJECTED
    assert report.substantive_statements_count == 2
    assert report.cited_statements_count == 1
    assert report.structural_coverage_passed is False
    assert report.valid_chunk_ids == []
    assert report.citation_items == []


def test_sentence_splitting_handles_abbreviations_decimals_lists_markdown_and_attached_citations(sample_chunks):
    """Valid sentences with abbreviations, decimals, numbered lists, Markdown, and trailing citations pass."""
    validator = CitationValidator()
    answer = (
        "Summary of CSE205 and Attendance Rules:\n"
        "1. **Credits & Prerequisites**: CSE205 carries 4.0 credits (e.g. requiring CSE101, Rs. 0 extra fee) [[cite:b84e8c8b020a1a9c]].\n"
        "- Students must maintain 75.5% attendance (cf. Sec. 4). [[cite:6a5ba40b1700e31a]]\n"
        "---"
    )
    report = validator.validate(answer, sample_chunks)
    assert report.is_valid is True
    assert report.status == CitationValidationStatus.VALID
    assert report.substantive_statements_count == 2
    assert report.cited_statements_count == 2
    assert report.structural_coverage_passed is True


def test_short_factual_statements_and_note_prefixes_not_exempt(sample_chunks):
    """Short factual statements (< 4 words) and 'Note:' statements must NOT be exempted from citation."""
    validator = CitationValidator()
    short_answer = "CSE205 has 4 credits [[cite:b84e8c8b020a1a9c]].\nCredits: 4."
    report_short = validator.validate(short_answer, sample_chunks)
    assert report_short.is_valid is False
    assert report_short.cited_statements_count == 1
    assert report_short.substantive_statements_count == 2

    note_answer = "CSE205 has 4 credits [[cite:b84e8c8b020a1a9c]].\nNote: Attendance must be 75%."
    report_note = validator.validate(note_answer, sample_chunks)
    assert report_note.is_valid is False
    assert report_note.cited_statements_count == 1
    assert report_note.substantive_statements_count == 2


def test_two_statement_answer_with_one_uncited_rejected(sample_chunks):
    """Closes the <= 2 statement loophole: 1/2 cited must fail 100% coverage."""
    validator = CitationValidator()
    answer = (
        "CSE205 has 4 credits [[cite:b84e8c8b020a1a9c]].\n"
        "Students must maintain 75% attendance in all courses."
    )
    report = validator.validate(answer, sample_chunks)
    assert report.is_valid is False
    assert report.substantive_statements_count == 2
    assert report.cited_statements_count == 1


@pytest.mark.parametrize(
    "bad_tag",
    [
        "[1]",
        "[2]",
        "[1, 2]",
        "[CSE202.pdf]",
        "[Source: CSE202.pdf]",
        "[FAQ013]",
        "[FAQ 13]",
        "[Doc 1]",
        "[Page 4]",
        "[[cite:]]",
        "[[citation:b84e8c8b020a1a9c]]",
        "[cite:b84e8c8b020a1a9c]",
        "[[cite:b84e8c8b020a1a9c]",
    ],
)
def test_pseudo_citations_rejected_even_with_valid_canonical_citation(sample_chunks, bad_tag):
    """Pseudo-citations and malformed tags must be rejected even alongside a valid canonical citation."""
    validator = CitationValidator()
    answer = f"CSE205 has 4 credits [[cite:b84e8c8b020a1a9c]] {bad_tag}."
    report = validator.validate(answer, sample_chunks)
    assert report.is_valid is False
    assert report.status == CitationValidationStatus.FAILED_REJECTED
    assert len(report.invalid_tags) >= 1
    assert report.valid_chunk_ids == []


def test_semantic_grounding_note_disclaims_semantic_correctness(sample_chunks):
    """Structural validation note must explicitly state it does not verify semantic correctness."""
    validator = CitationValidator()

    valid_report = validator.validate("CSE205 has 4 credits [[cite:b84e8c8b020a1a9c]].", sample_chunks)
    valid_schema = validator.to_schema_result(valid_report)
    assert "does not establish semantic correctness" in valid_schema.semantic_grounding_note.lower()

    failed_report = validator.validate("CSE205 has 4 credits.", sample_chunks)
    failed_schema = validator.to_schema_result(failed_report)
    assert "rejected" in failed_schema.semantic_grounding_note.lower()
    assert "not verified" in failed_schema.semantic_grounding_note.lower()
    assert failed_schema.valid_citations_count == 0

    bypassed_report = validator.validate("Abstention.", sample_chunks, is_abstention=True)
    bypassed_schema = validator.to_schema_result(bypassed_report)
    assert "bypassed" in bypassed_schema.semantic_grounding_note.lower()
