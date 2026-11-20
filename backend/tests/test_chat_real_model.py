"""Live Real-Model Integration Tests for POST /api/v1/chat/answer using Ollama (Phase 10.3).

These tests run against the live local Ollama runtime (qwen3:4b-instruct-2507-q4_K_M)
and actual indexed ChromaDB knowledge base chunks. No mocks are used for model inference.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.evidence_gate import EvidenceDecisionState
from app.schemas.chat import CitationValidationStatus
from app.services.llm_service import reset_llm_service_instances

client = TestClient(app)


@pytest.fixture(autouse=True)
def force_ollama_provider_for_real_model_tests():
    """Ensure real model tests target the local Ollama runtime as originally designed."""
    with patch("app.core.config.settings.llm_provider", "ollama"):
        reset_llm_service_instances()
        yield
        reset_llm_service_instances()


def test_real_model_supported_syllabus_question():
    """Scenario 1: Supported syllabus question (CSE202 topics)."""
    resp = client.post(
        "/api/v1/chat/answer",
        json={"query": "What topics are covered in CSE202 Object Oriented Programming?", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evidence_state"] == EvidenceDecisionState.SUFFICIENT.value
    assert data["model_identifier"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert data["citation_validation_status"] in [
        CitationValidationStatus.VALID.value,
        CitationValidationStatus.REGENERATED_VALID.value,
    ]
    assert len(data["citations"]) >= 1
    assert data["citation_validation"]["structural_coverage_passed"] is True
    # Verify metadata is correctly enriched
    for cit in data["citations"]:
        assert cit["chunk_id"] is not None
        assert cit["source_file"] == "CSE202.pdf"


def test_real_model_supported_textbooks_question():
    """Scenario 2: Supported syllabus question (CSE202 textbooks)."""
    resp = client.post(
        "/api/v1/chat/answer",
        json={"query": "What are the prescribed textbooks for CSE202?", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evidence_state"] == EvidenceDecisionState.SUFFICIENT.value
    assert data["model_identifier"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert data["citation_validation_status"] in [
        CitationValidationStatus.VALID.value,
        CitationValidationStatus.REGENERATED_VALID.value,
    ]
    assert len(data["citations"]) >= 1
    assert "OBJECT ORIENTED PROGRAMMING" in data["answer"].upper()


def test_real_model_partially_supported_question():
    """Scenario 3: Genuine PARTIAL query with verified PEL132 syllabus support + missing fee aspect."""
    resp = client.post(
        "/api/v1/chat/answer",
        json={
            "query": "What are the prescribed textbooks for PEL132, and what is the processing fee in rupees for supplementary audio materials?",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evidence_state"] == EvidenceDecisionState.PARTIAL.value
    assert data["model_identifier"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert data["citation_validation_status"] in [
        CitationValidationStatus.VALID.value,
        CitationValidationStatus.REGENERATED_VALID.value,
    ]
    assert data["citation_validation"]["status"] == data["citation_validation_status"]
    assert len(data["citations"]) >= 1
    assert all(c["source_file"] == "PEL132.pdf" for c in data["citations"])
    assert data["missing_information"] is not None
    assert len(data["missing_information"]) >= 1


def test_real_model_partial_no_verified_support_abstains_deterministically():
    """Scenario 4: PARTIAL query with zero verified supporting chunks abstains deterministically without LLM."""
    resp = client.post(
        "/api/v1/chat/answer",
        json={"query": "What happens if a student misses an end-term examination due to medical illness?", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evidence_state"] == EvidenceDecisionState.PARTIAL.value
    assert data["model_identifier"] == "deterministic_abstention_engine"
    assert data["generation_latency_ms"] == 0.0
    assert data["citations"] == []
    assert data["citation_validation_status"] == CitationValidationStatus.BYPASSED.value
    assert data["citation_validation"]["status"] == CitationValidationStatus.BYPASSED.value


def test_real_model_insufficient_bypasses_llm():
    """Scenario 5: Out-of-corpus INSUFFICIENT query bypasses LLM completely."""
    resp = client.post(
        "/api/v1/chat/answer",
        json={
            "query": "What is the refund policy for hostel fee cancellation if a student leaves in the middle of the semester?",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evidence_state"] == EvidenceDecisionState.INSUFFICIENT.value
    assert data["model_identifier"] == "deterministic_abstention_engine"
    assert data["generation_latency_ms"] == 0.0
    assert data["citations"] == []
    assert data["citation_validation_status"] == CitationValidationStatus.BYPASSED.value


def test_real_model_conflicting_bypasses_llm_with_traceable_provenance():
    """Scenario 6: Attendance 75% vs 80% query bypasses LLM and classifies as verification_caveat with citations."""
    resp = client.post(
        "/api/v1/chat/answer",
        json={
            "query": "What is the official minimum percentage attendance required to appear in examinations: is it 75% or 80%?",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evidence_state"] == EvidenceDecisionState.CONFLICTING.value
    assert data["model_identifier"] == "deterministic_conflict_engine"
    assert data["generation_latency_ms"] == 0.0
    assert data["citation_validation_status"] == CitationValidationStatus.BYPASSED.value
    assert data["conflict_details"]["conflict_classification"] == "verification_caveat"
    assert data["conflict_details"]["competing_claims"] == []
    assert len(data["citations"]) >= 2
    assert len(data["source_metadata"]) >= 2
    assert "across official documents" not in data["answer"].lower()


def test_real_model_citation_correctness_and_provenance():
    """Scenario 7: Citation correctness test with metadata provenance mapping and independent latency telemetry."""
    resp = client.post(
        "/api/v1/chat/answer",
        json={"query": "Which unit in CSE205 covers heaps and hashing?", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evidence_state"] == EvidenceDecisionState.SUFFICIENT.value
    assert data["citation_validation_status"] in [
        CitationValidationStatus.VALID.value,
        CitationValidationStatus.REGENERATED_VALID.value,
    ]
    assert data["citation_validation"]["status"] == data["citation_validation_status"]
    assert len(data["citations"]) >= 1
    # Verify independent latency measurements (reranking dominates retrieval; not hardcoded 15/85)
    assert data["retrieval_latency_ms"] > 0.0
    assert data["reranking_latency_ms"] > 0.0
    # Verify that all citation tags [[cite:X]] in the answer exist in citations list
    import re
    cites_in_text = re.findall(r"\[\[cite:([a-zA-Z0-9_\-\.]+)\]\]", data["answer"])
    assert len(cites_in_text) >= 1
    valid_ids_set = {c["chunk_id"] for c in data["citations"]}
    for cid in cites_in_text:
        assert cid in valid_ids_set
    # Verify source provenance metadata exists
    assert len(data["source_metadata"]) >= 1
    assert data["source_metadata"][0]["authority_level"] is not None
