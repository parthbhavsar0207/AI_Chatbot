"""Unit and integration tests for Evidence Sufficiency and Abstention Engine (Phase 9.2)."""

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.rag.evidence_gate import (
    EvidenceDecision,
    EvidenceDecisionState,
    EvidenceGate,
    GENERATION_ELIGIBLE_VERIFICATION_STATUSES,
    SOURCE_AUTHORITY_MAP,
    SOURCE_PROVENANCE_REGISTRY,
    is_chunk_verified_for_generation,
)
from app.rag.retrieval import RetrievedChunk


@pytest.fixture(scope="module")
def gate() -> EvidenceGate:
    """Provides a shared EvidenceGate instance."""
    return EvidenceGate()


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def _make_chunk(
    chunk_id: str,
    text: str,
    source_file: str = "CSE306.pdf",
    similarity: float = 0.85,
    reranker_score: float = 5.0,
    verification_status: str = "verified",
) -> RetrievedChunk:
    """Helper to construct a mock RetrievedChunk with evidence metadata."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=text,
        metadata={
            "source_file": source_file,
            "category": "syllabus",
            "verification_status": verification_status,
        },
        distance=round(1.0 - similarity, 6),
        similarity=similarity,
        reranker_score=reranker_score,
        original_rank=1,
        reranked_rank=1,
    )


# ==============================================================================
# 1. EvidenceGate Unit Tests: Decision States & Score Semantics
# ==============================================================================


def test_gate_sufficient_state(gate: EvidenceGate):
    """Verify that supported syllabus evidence yields SUFFICIENT state with heuristic_score."""
    query = "What are the course topics for CSE306?"
    chunk = _make_chunk(
        "c_suf_1",
        "Course CSE306 covers data link layer, network layer, routing protocols, and transport layer protocols.",
        source_file="CSE306.pdf",
        similarity=0.88,
        reranker_score=6.2,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.SUFFICIENT
    assert decision.reason_code == "SUFFICIENT_OFFICIAL_EVIDENCE"
    assert "directly supports" in decision.explanation
    assert decision.heuristic_score >= 0.80
    assert decision.confidence_score == decision.heuristic_score
    assert "c_suf_1" in decision.supporting_chunk_ids
    assert "CSE306.pdf" in decision.relevant_sources
    assert decision.missing_information is None
    assert decision.conflict_details is None
    assert decision.source_authorities.get("CSE306.pdf") == "curriculum_syllabus"
    assert "CSE306.pdf" in decision.source_provenance
    assert decision.source_provenance["CSE306.pdf"]["authority_level"] == "curriculum_syllabus"


def test_gate_partial_state_multi_aspect(gate: EvidenceGate):
    """Verify that multi-aspect query missing requested room number yields PARTIAL state."""
    query = "When is the CSE306 end-term examination and what is the classroom room number?"
    chunk = _make_chunk(
        "c_part_1",
        "The end term examination for CSE306 will be held on December 15, 2026 morning session.",
        source_file="lpu_cse_aiml_upcoming_end_term_examinations_2026-27.docx",
        similarity=0.82,
        reranker_score=4.5,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.PARTIAL
    assert decision.reason_code == "PARTIAL_EVIDENCE_MISSING_DETAILS"
    assert decision.missing_information is not None
    assert any("room" in item.lower() for item in decision.missing_information)


def test_gate_partial_state_corpus_incompleteness(gate: EvidenceGate):
    """Verify that corpus chunks containing explicit incompleteness disclaimers yield PARTIAL state."""
    query = "What is the formula to convert letter grades into exact percentage marks?"
    chunk = _make_chunk(
        "c_part_2",
        "Grading System: Letter grades are awarded. However, these documents do not establish a complete current rule "
        "or formula for conversion to exact percentage marks.",
        source_file="LPU_CSE_Academic_Regulations.docx",
        similarity=0.80,
        reranker_score=3.8,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.PARTIAL
    assert decision.reason_code == "PARTIAL_EVIDENCE_MISSING_DETAILS"
    assert decision.missing_information is not None
    assert any("incomplete" in item.lower() for item in decision.missing_information)


def test_gate_partial_state_unverified_source(gate: EvidenceGate):
    """Verify that evidence with unverified student status yields PARTIAL state."""
    query = "Can students appear in remedial classes if attendance is 72%?"
    chunk = _make_chunk(
        "c_part_3",
        "FAQ015: Remedial classes may be offered before end term. Students should check department desk.",
        source_file="LPU_CSE_AIML_Student_FAQs.json",
        similarity=0.78,
        reranker_score=2.5,
        verification_status="needs_verification",
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.PARTIAL
    assert decision.reason_code == "PARTIAL_EVIDENCE_MISSING_DETAILS"
    assert decision.missing_information is not None
    assert any("unverified" in item.lower() for item in decision.missing_information)


def test_gate_insufficient_state_out_of_corpus(gate: EvidenceGate):
    """Verify that irrelevant, out-of-corpus queries with low reranker scores yield INSUFFICIENT state."""
    query = "What is the procedure for scuba diving certification at campus?"
    chunk = _make_chunk(
        "c_insuf_1",
        "Hostel allotment details for 2026-27 are managed through the university security office.",
        source_file="LPU_CSE_Academic_Regulations.docx",
        similarity=0.45,
        reranker_score=-7.5,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.INSUFFICIENT
    assert decision.reason_code == "INSUFFICIENT_EVIDENCE_OUT_OF_CORPUS"
    assert "not available" in decision.explanation
    assert decision.missing_information is not None


def test_gate_insufficient_state_empty_candidates(gate: EvidenceGate):
    """Verify that an empty retrieved chunks list yields INSUFFICIENT state gracefully."""
    decision = gate.evaluate(query="Any academic query", retrieved_chunks=[])

    assert decision.decision == EvidenceDecisionState.INSUFFICIENT
    assert decision.reason_code == "NO_RETRIEVED_CHUNKS"
    assert decision.heuristic_score == 0.0
    assert decision.confidence_score == 0.0
    assert decision.supporting_chunk_ids == []
    assert decision.relevant_sources == []


def test_gate_insufficient_state_historical_academic_year(gate: EvidenceGate):
    """Verify that historical year queries (e.g. 2024-25) trigger abstention."""
    query = "What was the CSE306 examination schedule for academic year 2024-25?"
    chunk = _make_chunk(
        "c_hist_1",
        "Current academic examination notices apply to 2026-27 cohort.",
        similarity=0.76,
        reranker_score=1.5,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.INSUFFICIENT
    assert decision.reason_code == "STALE_OR_UNAVAILABLE_ACADEMIC_YEAR"
    assert "2024-25" in decision.explanation


def test_gate_conflicting_state_regulatory_ambiguity(gate: EvidenceGate):
    """Verify that attendance requirement + FAQ unverified caveat triggers CONFLICTING with verification_caveat classification."""
    query = "Is the minimum attendance required for final exams 75% or 80%?"
    chunk1 = _make_chunk(
        "c_conf_1",
        "Section 1 Attendance: A student must maintain an aggregate attendance of 80%.",
        source_file="LPU_CSE_Academic_Regulations.docx",
        similarity=0.86,
        reranker_score=6.0,
    )
    chunk2 = _make_chunk(
        "c_conf_2",
        "FAQ013: Students often cite 75%, but this unverified percentage should not be relied upon without ERP confirmation.",
        source_file="LPU_CSE_AIML_Student_FAQs.json",
        similarity=0.84,
        reranker_score=5.5,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk1, chunk2])

    assert decision.decision == EvidenceDecisionState.CONFLICTING
    assert decision.reason_code == "CONTRADICTORY_EVIDENCE_DETECTED"
    assert decision.conflict_details is not None
    assert decision.conflict_details["conflict_classification"] == "verification_caveat"
    assert decision.conflict_details["competing_claims"] == []
    assert len(decision.conflict_details["source_references"]) == 2
    assert "c_conf_1" in decision.supporting_chunk_ids
    assert "c_conf_2" in decision.supporting_chunk_ids


def test_gate_conflicting_state_synthetic(gate: EvidenceGate):
    """Verify that synthetic conflict flag is detected and structured properly."""
    query = "What is the submission deadline? [Synthetic Conflict]"
    chunk = _make_chunk("c_syn_1", "Some deadline claim.", similarity=0.80, reranker_score=3.0)
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.CONFLICTING
    assert decision.reason_code == "CONTRADICTORY_EVIDENCE_DETECTED"
    assert decision.conflict_details is not None


# ==============================================================================
# 2. Phase 9.2 Specific Reliability Corrections Tests (H10, H16, H17, H23, H24, H28)
# ==============================================================================


def test_gate_factual_detail_absence_h10(gate: EvidenceGate):
    """H10 Fix: Verify syllabus topic presence without requested recurrence formula yields INSUFFICIENT."""
    query = "What is the dynamic programming recurrence formula for the Bellman-Ford shortest path algorithm in CSE205?"
    chunk = _make_chunk(
        "c_cse205_unit4",
        "CSE205: DATA STRUCTURES AND ALGORITHMS. Unit 4: Graphs: Shortest path algorithms: Dijkstra's and Bellman-Ford algorithm.",
        source_file="CSE205.pdf",
        similarity=0.75,
        reranker_score=2.91,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.INSUFFICIENT
    assert decision.reason_code == "INSUFFICIENT_FACTUAL_DETAIL_IN_SYLLABUS"
    assert "recurrence formula" in decision.explanation.lower()


def test_gate_multi_aspect_missing_fee_h16(gate: EvidenceGate):
    """H16 Fix: Multi-aspect query asking for PEL132 textbooks + absent audio fee yields PARTIAL."""
    query = "What are the prescribed textbooks for PEL132, and what is the processing fee in rupees for supplementary audio materials?"
    chunk = _make_chunk(
        "c_pel132_tb",
        "PEL132: Prescribed Books: 1. Technical Communication: Principles and Practice by Meenakshi Raman and Sangeeta Sharma.",
        source_file="PEL132.pdf",
        similarity=0.68,
        reranker_score=-5.86,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.PARTIAL
    assert decision.reason_code == "PARTIAL_EVIDENCE_MISSING_DETAILS"
    assert decision.missing_information is not None
    assert any("fee" in item.lower() for item in decision.missing_information)


def test_gate_multi_aspect_missing_fee_h17(gate: EvidenceGate):
    """H17 Fix: Multi-aspect query asking for re-evaluation procedure + absent fee yields PARTIAL."""
    query = "What is the procedure to request an examination answer script copy, and what is the processing fee in rupees?"
    chunk = _make_chunk(
        "c_reg_reeval",
        "6. Examination scrutiny and re-evaluation: Students may physically scrutinize evaluated MTT/ETE answer booklets.",
        source_file="LPU_CSE_Academic_Regulations.docx",
        similarity=0.61,
        reranker_score=-8.73,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.PARTIAL
    assert decision.reason_code == "PARTIAL_EVIDENCE_MISSING_DETAILS"
    assert decision.missing_information is not None
    assert any("fee" in item.lower() for item in decision.missing_information)


def test_gate_unverified_student_material_h23_h24(gate: EvidenceGate):
    """H23/H24 Fix: Unverified student FAQ queries yield PARTIAL rather than generic out-of-corpus."""
    query = "What are the unverified student guidelines regarding remedial classes before end term?"
    chunk = _make_chunk(
        "c_faq_015",
        "FAQ015: Remedial classes may be offered before end term. Students should check department desk.",
        source_file="LPU_CSE_AIML_Student_FAQs.json",
        similarity=0.69,
        reranker_score=-1.71,
        verification_status="needs_verification",
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.PARTIAL
    assert decision.reason_code == "PARTIAL_EVIDENCE_MISSING_DETAILS"
    assert decision.missing_information is not None
    assert any("unverified" in item.lower() for item in decision.missing_information)


def test_gate_synthetic_conflict_early_detection_h28(gate: EvidenceGate):
    """H28 Fix: Synthetic conflict marker is detected even when reranker score is negative."""
    query = "What is the course grading policy? [Synthetic Conflict]"
    chunk = _make_chunk(
        "c_reg_grading",
        "Grading System: Letter grades are awarded.",
        source_file="LPU_CSE_Academic_Regulations.docx",
        similarity=0.68,
        reranker_score=-9.47,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.CONFLICTING
    assert decision.reason_code == "CONTRADICTORY_EVIDENCE_DETECTED"
    assert decision.conflict_details is not None


def test_heuristic_score_and_confidence_score_sync():
    """Verify that heuristic_score and deprecated confidence_score stay synchronized."""
    d1 = EvidenceDecision(
        decision=EvidenceDecisionState.SUFFICIENT,
        reason_code="TEST_CODE",
        explanation="Test explanation",
        heuristic_score=0.88,
    )
    assert d1.heuristic_score == 0.88
    assert d1.confidence_score == 0.88

    d2 = EvidenceDecision(
        decision=EvidenceDecisionState.PARTIAL,
        reason_code="TEST_CODE",
        explanation="Test explanation",
        confidence_score=0.75,
    )
    assert d2.heuristic_score == 0.75
    assert d2.confidence_score == 0.75


def test_source_provenance_registry_structure():
    """Verify that source provenance registry separates origin, authority, authenticity, and academic year."""
    schedule_entry = SOURCE_PROVENANCE_REGISTRY["lpu_cse_aiml_upcoming_end_term_examinations_2026-27.docx"]
    assert schedule_entry["document_origin"] == "user_provided_snapshot"
    assert schedule_entry["authority_level"] == "curated_schedule_snapshot"
    assert schedule_entry["institutional_authenticity"] == "unverified"
    assert schedule_entry["academic_year"] == "2026-27"

    reg_entry = SOURCE_PROVENANCE_REGISTRY["LPU_CSE_Academic_Regulations.docx"]
    assert reg_entry["authority_level"] == "departmental_academic_regulations"
    assert reg_entry["institutional_authenticity"] == "partially_verified"


def test_source_authority_mapping():
    """Verify that source authority map reflects audited authority levels."""
    assert SOURCE_AUTHORITY_MAP["LPU_CSE_Academic_Regulations.docx"] == "departmental_academic_regulations"
    assert SOURCE_AUTHORITY_MAP["CSE306.pdf"] == "curriculum_syllabus"
    assert SOURCE_AUTHORITY_MAP["LPU_CSE_AIML_Student_FAQs.json"] == "student_contributed_faq"
    assert SOURCE_AUTHORITY_MAP["lpu_cse_aiml_upcoming_end_term_examinations_2026-27.docx"] == "curated_schedule_snapshot"


def test_gate_invalid_query_raises_value_error(gate: EvidenceGate):
    """Verify that empty string or whitespace query raises ValueError."""
    chunk = _make_chunk("c1", "text")
    with pytest.raises(ValueError, match="cannot be empty"):
        gate.evaluate(query="", retrieved_chunks=[chunk])

    with pytest.raises(ValueError, match="cannot be empty"):
        gate.evaluate(query="   \n\t  ", retrieved_chunks=[chunk])


# ==============================================================================
# 3. API Integration Tests (POST /api/v1/retrieval/search)
# ==============================================================================


def test_api_evidence_gate_disabled_by_default(client: TestClient):
    """Verify that evaluate_evidence defaults to False and evidence_decision is None."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "What is the attendance policy?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence_decision"] is None


def test_api_evidence_gate_enabled_sufficient(client: TestClient):
    """Verify API returns evidence_decision='sufficient' for well-grounded syllabus query."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "What are the course objectives and topics for CSE306?",
            "evaluate_evidence": True,
            "enable_rerank": True,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    decision = data.get("evidence_decision")
    assert decision is not None
    assert decision["decision"] == "sufficient"
    assert decision["reason_code"] == "SUFFICIENT_OFFICIAL_EVIDENCE"
    assert len(decision["supporting_chunk_ids"]) > 0
    assert len(decision["relevant_sources"]) > 0
    assert decision["heuristic_score"] > 0.0
    assert decision["confidence_score"] == decision["heuristic_score"]
    assert "source_provenance" in decision


def test_api_evidence_gate_enabled_insufficient(client: TestClient):
    """Verify API returns evidence_decision='insufficient' for out-of-corpus query."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "What are the requirements for pilot license aviation training?",
            "evaluate_evidence": True,
            "enable_rerank": True,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    decision = data.get("evidence_decision")
    assert decision is not None
    assert decision["decision"] == "insufficient"
    assert decision["reason_code"] == "INSUFFICIENT_EVIDENCE_OUT_OF_CORPUS"


def test_api_evidence_gate_enabled_conflicting(client: TestClient):
    """Verify API returns evidence_decision='conflicting' for contradictory attendance query."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "Is the attendance threshold 75% or 80%?",
            "evaluate_evidence": True,
            "enable_rerank": True,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    decision = data.get("evidence_decision")
    assert decision is not None
    assert decision["decision"] == "conflicting"
    assert decision["reason_code"] == "CONTRADICTORY_EVIDENCE_DETECTED"
    assert decision["conflict_details"] is not None
    assert "competing_claims" in decision["conflict_details"]


def test_api_evidence_gate_enabled_partial(client: TestClient):
    """Verify API returns evidence_decision='partial' for query with missing detail."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "What is the re-evaluation procedure for answer booklets and what is the processing fee?",
            "evaluate_evidence": True,
            "enable_rerank": True,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    decision = data.get("evidence_decision")
    assert decision is not None
    assert decision["decision"] == "partial"
    assert decision["reason_code"] == "PARTIAL_EVIDENCE_MISSING_DETAILS"
    assert decision["missing_information"] is not None


# ==============================================================================
# 4. Phase 10.3 Remediation Regression Tests (Fail-Closed Verification,
#    Content-Based PARTIAL Support, Three-Way Conflict Classification)
# ==============================================================================


def test_is_chunk_verified_fails_closed_on_missing_null_empty_unknown_and_unverified():
    """Verification must use an explicit allowlist and fail closed on missing/null/empty/unknown/unverified."""
    def _chunk_with_meta(meta: dict, text: str = "Factual syllabus text.") -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id="cid_test",
            document_id="doc_test",
            text=text,
            metadata=meta,
            distance=0.1,
            similarity=0.9,
        )

    # Explicitly allowlisted statuses pass
    assert "verified" in GENERATION_ELIGIBLE_VERIFICATION_STATUSES
    assert "verified_syllabus_structure" in GENERATION_ELIGIBLE_VERIFICATION_STATUSES
    assert is_chunk_verified_for_generation(_chunk_with_meta({"verification_status": "verified"})) is True
    assert is_chunk_verified_for_generation(_chunk_with_meta({"verification_status": "verified_syllabus_structure"})) is True
    # Omitted key on audited syllabus source resolves to verified_syllabus_structure
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "PEL132.pdf"})) is True

    # Missing, null, empty, whitespace, unknown, unverified, partially_verified MUST fail closed
    assert is_chunk_verified_for_generation(_chunk_with_meta({})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "unregistered_doc.pdf"})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "PEL132.pdf", "verification_status": None})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "PEL132.pdf", "verification_status": ""})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "PEL132.pdf", "verification_status": "   "})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "PEL132.pdf", "verification_status": "unknown_status"})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "LPU_CSE_AIML_Student_FAQs.json", "verification_status": "needs_verification"})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "LPU_CSE_AIML_Student_FAQs.json", "verification_status": "partially_verified"})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "LPU_CSE_AIML_Student_FAQs.json", "verification_status": "verified_wording_meaning_needs_verification"})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "LPU_CSE_AIML_Student_FAQs.json", "verification_status": "verified_from_user_supplied_schedule"})) is False
    # Audited sources whose registry status is unverified fail closed when chunk status is omitted
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "LPU_CSE_Academic_Regulations.docx"})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "lpu_cse_aiml_upcoming_end_term_examinations_2026-27.docx"})) is False
    assert is_chunk_verified_for_generation(_chunk_with_meta({"source_file": "LPU_CSE_AIML_Student_FAQs.json"})) is False
    # Inline unverified text marker fails closed even if metadata claims verified
    assert is_chunk_verified_for_generation(
        _chunk_with_meta(
            {"verification_status": "verified"},
            text="Verification status: Needs verification from current LPU handbook.",
        )
    ) is False


def test_partial_excludes_unverified_faq_and_disclaimer_only_chunks(gate: EvidenceGate):
    """PARTIAL supporting_chunk_ids must exclude unverified FAQs and disclaimer-only chunks."""
    query = "What are the prescribed textbooks for PEL132, and what is the processing fee in rupees for supplementary audio materials?"
    verified_tb = _make_chunk(
        "c_pel_tb",
        "PEL132 Text Book: 1. 5th Edition Headway Intermediate Student's Book by Liz & John Soars.",
        source_file="PEL132.pdf",
        similarity=0.76,
        reranker_score=2.5,
        verification_status="verified_syllabus_structure",
    )
    unverified_faq = _make_chunk(
        "c_unver_faq",
        "Question: PEL132 textbooks?\nAnswer: This FAQ dataset does not yet contain a verified rule for PEL132 textbooks.",
        source_file="LPU_CSE_AIML_Student_FAQs.json",
        similarity=0.80,
        reranker_score=3.2,
        verification_status="needs_verification",
    )
    disclaimer_only = _make_chunk(
        "c_disc_only",
        "The public sources reviewed do not establish a complete current rule for PEL132 audio fees. Verification status: Needs verification.",
        source_file="PEL132.pdf",
        similarity=0.74,
        reranker_score=1.8,
        verification_status="verified_syllabus_structure",
    )

    decision = gate.evaluate(query=query, retrieved_chunks=[unverified_faq, verified_tb, disclaimer_only])
    assert decision.decision == EvidenceDecisionState.PARTIAL
    assert decision.supporting_chunk_ids == ["c_pel_tb"]


def test_partial_course_code_alone_does_not_qualify_as_support(gate: EvidenceGate):
    """Course code match or positive reranker score alone must NOT qualify a non-supporting chunk."""
    query = "What are the prescribed textbooks for PEL132, and what is the processing fee in rupees for supplementary audio materials?"
    unrelated_pel_chunk = _make_chunk(
        "c_pel_outcomes",
        "PEL132 Course Outcomes: CO1 apply appropriate vocabulary, intonation and style of connected discourse.",
        source_file="PEL132.pdf",
        similarity=0.78,
        reranker_score=3.5,
        verification_status="verified_syllabus_structure",
    )
    textbook_pel_chunk = _make_chunk(
        "c_pel_books",
        "PEL132 Text Book: Headway Intermediate Student's Book by Liz & John Soars.",
        source_file="PEL132.pdf",
        similarity=0.72,
        reranker_score=1.2,
        verification_status="verified_syllabus_structure",
    )

    decision = gate.evaluate(query=query, retrieved_chunks=[unrelated_pel_chunk, textbook_pel_chunk])
    assert decision.decision == EvidenceDecisionState.PARTIAL
    assert "c_pel_outcomes" not in decision.supporting_chunk_ids
    assert decision.supporting_chunk_ids == ["c_pel_books"]


def test_partial_all_unverified_or_non_supporting_yields_empty_supporting_ids(gate: EvidenceGate):
    """When retrieved chunks are unverified FAQs or unrelated exam schedules, supporting_chunk_ids must be empty."""
    query = "What happens if a student misses an end-term examination due to medical illness?"
    unverified_faq = _make_chunk(
        "c_faq_016",
        "Question: What happens if I miss an end-term examination?\nAnswer: This dataset does not contain a verified current LPU rule covering absence from an end-term examination.",
        source_file="LPU_CSE_AIML_Student_FAQs.json",
        similarity=0.81,
        reranker_score=4.0,
        verification_status="needs_verification",
    )
    unrelated_sched = _make_chunk(
        "c_sched_1",
        "INT335 end-term examination is scheduled on 15 December 2026 from 09:30 to 12:30 with 60 MCQs.",
        source_file="LPU_CSE_AIML_Student_FAQs.json",
        similarity=0.71,
        reranker_score=-1.2,
        verification_status="verified_from_user_supplied_schedule",
    )

    decision = gate.evaluate(query=query, retrieved_chunks=[unverified_faq, unrelated_sched])
    assert decision.decision == EvidenceDecisionState.PARTIAL
    assert decision.supporting_chunk_ids == []


def test_conflict_actual_conflict_maps_two_incompatible_requirements(gate: EvidenceGate):
    """Two chunks stating incompatible mandatory attendance thresholds must be classified as actual_conflict."""
    query = "Is the minimum mandatory attendance 75% or 80%?"
    chunk_a = _make_chunk(
        "c_req_80",
        "Academic Policy 2025: Students must maintain a mandatory minimum attendance of 80% in all courses.",
        source_file="LPU_CSE_Academic_Regulations.docx",
        similarity=0.88,
        reranker_score=6.0,
    )
    chunk_b = _make_chunk(
        "c_req_75",
        "Examination Circular 2026: Every student must maintain a mandatory minimum attendance of 75% to sit for exams.",
        source_file="CSE202.pdf",
        similarity=0.85,
        reranker_score=5.4,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk_a, chunk_b])

    assert decision.decision == EvidenceDecisionState.CONFLICTING
    details = decision.conflict_details
    assert details is not None
    assert details["conflict_classification"] == "actual_conflict"
    assert details["traceable_conflict_established"] is True
    assert len(details["competing_claims"]) == 2
    assert details["affected_chunk_ids"] == ["c_req_80", "c_req_75"]
    assert all(ref["role"] == "competing_requirement" for ref in details["source_references"])


def test_conflict_filtered_single_source_is_untraceable_without_fabricated_faq(gate: EvidenceGate):
    """Filtered single-source retrieval with no opposing/caveat chunk must NOT fabricate an FAQ013 claim."""
    query = "Is the minimum attendance 75% or 80%?"
    single_reg = _make_chunk(
        "c_reg_only",
        "1. Attendance: Students should attend at least 80% in aggregate of delivered classes.",
        source_file="LPU_CSE_Academic_Regulations.docx",
        similarity=0.86,
        reranker_score=5.5,
    )
    decision = gate.evaluate(query=query, retrieved_chunks=[single_reg])

    assert decision.decision == EvidenceDecisionState.CONFLICTING
    details = decision.conflict_details
    assert details is not None
    assert details["conflict_classification"] == "untraceable_conflict"
    assert details["traceable_conflict_established"] is False
    assert details["competing_claims"] == []
    assert "FAQ013" not in str(details)
    assert "LPU_CSE_AIML_Student_FAQs.json" not in details["affected_sources"]


def test_conflict_synthetic_isolated_from_production_evidence(gate: EvidenceGate):
    """Synthetic conflict marker must stay isolated without fabricated Document B claims."""
    query = "What is the credit limit? [Synthetic Conflict]"
    chunk = _make_chunk("c_syn_only", "Normal semester credit limit is 24.", similarity=0.80, reranker_score=2.0)
    decision = gate.evaluate(query=query, retrieved_chunks=[chunk])

    assert decision.decision == EvidenceDecisionState.CONFLICTING
    details = decision.conflict_details
    assert details is not None
    assert details["conflict_classification"] == "synthetic_benchmark"
    assert details["is_synthetic"] is True
    assert details["traceable_conflict_established"] is False
    assert details["competing_claims"] == []
    assert details["source_references"] == []
