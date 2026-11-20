"""Mocked and contract tests for POST /api/v1/chat/answer endpoint (Phase 10.3)."""

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import httpx
import pytest

from app.main import app
from app.rag.evidence_gate import EvidenceDecision, EvidenceDecisionState
from app.rag.retrieval import RetrievedChunk
from app.schemas.chat import CitationValidationStatus
from app.services.llm_service import OllamaTimeoutError, OllamaUnavailableError, reset_llm_service_instances

client = TestClient(app)


@pytest.fixture(autouse=True)
def force_ollama_provider_for_mocked_chat_tests():
    """Ensure mocked chat tests isolate from environment variables and default to Ollama."""
    with patch("app.core.config.settings.llm_provider", "ollama"):
        reset_llm_service_instances()
        yield
        reset_llm_service_instances()


def test_empty_query_returns_422():
    resp = client.post("/api/v1/chat/answer", json={"query": "   "})
    assert resp.status_code == 422


def test_insufficient_evidence_bypasses_llm_completely():
    # When query is out of corpus or insufficient
    with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
        resp = client.post(
            "/api/v1/chat/answer",
            json={"query": "What is the fee refund percentage for 4th week admission withdrawal?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["evidence_state"] == "insufficient"
        assert data["model_identifier"] == "deterministic_abstention_engine"
        assert data["generation_latency_ms"] == 0.0
        assert data["citation_validation_status"] == "bypassed"
        assert "do not sufficiently support an answer" in data["answer"]
        # LLM generate must never have been called
        mock_gen.assert_not_called()


def test_conflicting_evidence_bypasses_llm_completely():
    # Force conflicting decision from evidence gate
    mock_decision = EvidenceDecision(
        decision=EvidenceDecisionState.CONFLICTING,
        reason_code="cross_source_contradiction",
        explanation="Contradiction between 2022 and 2024 regulations regarding credit limits.",
        heuristic_score=0.45,
        relevant_sources=["Academic_Regulations_Archive_2022.pdf", "Academic_Regulations_Amended_2024.pdf"],
        conflict_details={"rule": "semester_credit_limit", "source_1": 24, "source_2": 26},
    )

    with patch("app.rag.evidence_gate.EvidenceGate.evaluate", return_value=mock_decision):
        with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
            resp = client.post(
                "/api/v1/chat/answer",
                json={"query": "What is the maximum credit limit per semester?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["evidence_state"] == "conflicting"
            assert data["model_identifier"] == "deterministic_conflict_engine"
            assert data["generation_latency_ms"] == 0.0
            assert data["citation_validation_status"] == "bypassed"
            assert "contradiction was detected" in data["answer"]
            mock_gen.assert_not_called()


def test_ollama_unavailable_returns_503():
    with patch(
        "app.services.llm_service.OllamaLLMService.generate",
        side_effect=OllamaUnavailableError("Ollama offline", detail="Local LLM service is offline."),
    ):
        resp = client.post(
            "/api/v1/chat/answer",
            json={"query": "What are the prerequisites for CSE205?"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "Local LLM service is offline" in data["detail"]


def test_ollama_timeout_returns_504():
    with patch(
        "app.services.llm_service.OllamaLLMService.generate",
        side_effect=OllamaTimeoutError("Timeout", detail="LLM generation timed out after 60 seconds."),
    ):
        resp = client.post(
            "/api/v1/chat/answer",
            json={"query": "What are the prerequisites for CSE205?"},
        )
        assert resp.status_code == 504
        data = resp.json()
        assert "timed out" in data["detail"]


def test_controlled_regeneration_on_first_failure():
    # First response has no citations; second response has valid citation
    first_resp = "CSE205 has 4 credits and requires CSE101."
    # The actual chunk_id will be provided in retrieved chunks
    retrieved_chunk = RetrievedChunk(
        chunk_id="b84e8c8b020a1a9c",
        document_id="doc1",
        text="CSE205 has 4 credits.",
        metadata={"source_file": "CSE205.pdf", "category": "syllabus"},
        distance=0.1,
        similarity=0.9,
    )
    second_resp = "CSE205 has 4 credits and requires CSE101 [[cite:b84e8c8b020a1a9c]]."

    with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[retrieved_chunk]):
        with patch(
            "app.rag.evidence_gate.EvidenceGate.evaluate",
            return_value=EvidenceDecision(
                decision=EvidenceDecisionState.SUFFICIENT,
                reason_code="high_relevance",
                explanation="Directly supported",
                heuristic_score=0.9,
                supporting_chunk_ids=["b84e8c8b020a1a9c"],
            ),
        ):
            with patch(
                "app.services.llm_service.OllamaLLMService.generate",
                side_effect=[(first_resp, {}), (second_resp, {})],
            ) as mock_gen:
                resp = client.post(
                    "/api/v1/chat/answer",
                    json={"query": "What are the prerequisites for CSE205?"},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["citation_validation_status"] == "regenerated_valid"
                assert mock_gen.call_count == 2
                assert len(data["citations"]) == 1
                assert data["citations"][0]["chunk_id"] == "b84e8c8b020a1a9c"


def test_fallback_when_regeneration_still_fails_citation_validation():
    # Both initial and regenerated responses fail citation validation
    bad_resp = "CSE205 has 4 credits [[cite:invalid_fabricated_id]]."
    retrieved_chunk = RetrievedChunk(
        chunk_id="b84e8c8b020a1a9c",
        document_id="doc1",
        text="CSE205 has 4 credits.",
        metadata={"source_file": "CSE205.pdf"},
        distance=0.1,
        similarity=0.9,
    )

    with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[retrieved_chunk]):
        with patch(
            "app.rag.evidence_gate.EvidenceGate.evaluate",
            return_value=EvidenceDecision(
                decision=EvidenceDecisionState.SUFFICIENT,
                reason_code="high_relevance",
                explanation="Directly supported",
                heuristic_score=0.9,
                supporting_chunk_ids=["b84e8c8b020a1a9c"],
            ),
        ):
            with patch(
                "app.services.llm_service.OllamaLLMService.generate",
                side_effect=[(bad_resp, {}), (bad_resp, {})],
            ):
                resp = client.post(
                    "/api/v1/chat/answer",
                    json={"query": "What are the prerequisites for CSE205?"},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["citation_validation_status"] == "failed_rejected"
                assert data["citation_validation"]["status"] == "failed_rejected"
                assert "could not satisfy strict citation verification standards" in data["answer"]
                assert "invalid_fabricated_id" not in data["answer"]
                assert "invalid_fabricated_id" not in str(data["citation_validation"]["invalid_citations"])
                assert data["citations"] == []
                assert data["citation_validation"]["valid_citations_count"] == 0


def test_partial_empty_supporting_intersection_abstains_without_llm():
    """PARTIAL with empty supporting_chunk_ids intersection must abstain deterministically (no top-3 fallback)."""
    retrieved_chunk = RetrievedChunk(
        chunk_id="unverified_chunk_1",
        document_id="doc_faq",
        text="Unverified FAQ answer.",
        metadata={"source_file": "LPU_CSE_AIML_Student_FAQs.json", "verification_status": "needs_verification"},
        distance=0.15,
        similarity=0.85,
    )
    partial_decision = EvidenceDecision(
        decision=EvidenceDecisionState.PARTIAL,
        reason_code="PARTIAL_EVIDENCE_MISSING_DETAILS",
        explanation="No verified supporting chunks available.",
        heuristic_score=0.85,
        supporting_chunk_ids=[],
        missing_information=["Verified official policy missing."],
    )

    with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[retrieved_chunk]):
        with patch("app.rag.evidence_gate.EvidenceGate.evaluate", return_value=partial_decision):
            with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
                resp = client.post(
                    "/api/v1/chat/answer",
                    json={"query": "What happens if I miss an exam due to illness?"},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["evidence_state"] == "partial"
                assert data["model_identifier"] == "deterministic_abstention_engine"
                assert data["generation_latency_ms"] == 0.0
                assert data["citations"] == []
                assert data["source_metadata"] == []
                assert data["missing_information"] == ["Verified official policy missing."]
                assert data["citation_validation_status"] == "bypassed"
                mock_gen.assert_not_called()


def test_partial_excludes_unverified_and_missing_status_chunks_from_prompt_and_citations():
    """Only allowlisted verified supporting chunks may enter the LLM prompt and permitted citation set."""
    verified_chunk = RetrievedChunk(
        chunk_id="verified_pel_tb",
        document_id="doc_pel",
        text="PEL132 Text Book: Headway Intermediate Student's Book.",
        metadata={"source_file": "PEL132.pdf", "category": "syllabus", "course_code": "PEL132"},
        distance=0.12,
        similarity=0.88,
    )
    unverified_faq = RetrievedChunk(
        chunk_id="excluded_faq_unverified",
        document_id="doc_faq",
        text="Unverified FAQ claim about PEL132.",
        metadata={"source_file": "LPU_CSE_AIML_Student_FAQs.json", "verification_status": "needs_verification"},
        distance=0.10,
        similarity=0.90,
    )
    missing_prov_chunk = RetrievedChunk(
        chunk_id="excluded_missing_prov",
        document_id="doc_unknown",
        text="Unknown unregistered source text.",
        metadata={"source_file": "unregistered_notes.pdf"},
        distance=0.14,
        similarity=0.86,
    )

    # Even if supporting_chunk_ids tried to list unverified/missing chunks, RAGChatService must filter them out
    partial_decision = EvidenceDecision(
        decision=EvidenceDecisionState.PARTIAL,
        reason_code="PARTIAL_EVIDENCE_MISSING_DETAILS",
        explanation="Textbooks supported; audio fee missing.",
        heuristic_score=0.85,
        supporting_chunk_ids=["verified_pel_tb", "excluded_faq_unverified", "excluded_missing_prov"],
        missing_information=["fee amount in currency"],
    )

    captured_prompts: list[str] = []

    async def fake_generate(prompt: str, system_prompt: str = None, **kwargs):
        captured_prompts.append(prompt)
        if len(captured_prompts) == 1:
            # First attempt tries to cite an excluded chunk -> MUST fail validation!
            return "PEL132 uses Headway [[cite:excluded_faq_unverified]].", {}
        # Second attempt cites only the verified chunk -> passes regeneration
        return (
            "The prescribed textbook for PEL132 is Headway Intermediate Student's Book [[cite:verified_pel_tb]]. "
            "The processing fee in rupees is not available in the provided documents [[cite:verified_pel_tb]].",
            {},
        )

    with patch(
        "app.rag.retrieval.RetrievalService.retrieve",
        return_value=[unverified_faq, verified_chunk, missing_prov_chunk],
    ):
        with patch("app.rag.evidence_gate.EvidenceGate.evaluate", return_value=partial_decision):
            with patch("app.services.llm_service.OllamaLLMService.generate", side_effect=fake_generate):
                resp = client.post(
                    "/api/v1/chat/answer",
                    json={"query": "What are the textbooks for PEL132 and the audio fee in rupees?"},
                )
                assert resp.status_code == 200
                data = resp.json()
                # Verify prompt NEVER contained excluded chunks
                assert len(captured_prompts) == 2
                for p in captured_prompts:
                    assert "verified_pel_tb" in p
                    assert "excluded_faq_unverified" not in p
                    assert "excluded_missing_prov" not in p
                # First attempt citing excluded chunk failed -> regenerated_valid
                assert data["citation_validation_status"] == "regenerated_valid"
                assert data["citation_validation"]["status"] == "regenerated_valid"
                assert [c["chunk_id"] for c in data["citations"]] == ["verified_pel_tb"]


def test_conflict_actual_conflict_populates_citations_and_competing_claims():
    """Actual conflict bypasses LLM, populates deterministic citations/provenance, and labels competing policy claims."""
    chunk_a = RetrievedChunk(
        chunk_id="c_act_1",
        document_id="doc_a",
        text="Academic Policy 2025: Mandatory minimum attendance is 80%.",
        metadata={"source_file": "LPU_CSE_Academic_Regulations.docx", "category": "regulations"},
        distance=0.1,
        similarity=0.9,
    )
    chunk_b = RetrievedChunk(
        chunk_id="c_act_2",
        document_id="doc_b",
        text="Syllabus Notice: Mandatory minimum attendance is 75%.",
        metadata={"source_file": "CSE202.pdf", "category": "syllabus"},
        distance=0.12,
        similarity=0.88,
    )

    with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[chunk_a, chunk_b]):
        with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
            resp = client.post(
                "/api/v1/chat/answer",
                json={"query": "Is mandatory minimum attendance 75% or 80%?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["evidence_state"] == "conflicting"
            assert data["model_identifier"] == "deterministic_conflict_engine"
            assert data["generation_latency_ms"] == 0.0
            assert data["conflict_details"]["conflict_classification"] == "actual_conflict"
            assert "competing policy requirements" in data["answer"].lower()
            assert ".." not in data["answer"]
            assert {c["chunk_id"] for c in data["citations"]} == {"c_act_1", "c_act_2"}
            assert len(data["source_metadata"]) == 2
            mock_gen.assert_not_called()


def test_conflict_verification_caveat_does_not_label_as_competing_claims_or_official_docs():
    """Requirement + unverified FAQ warning must be classified as verification_caveat, not competing official policies."""
    reg_chunk = RetrievedChunk(
        chunk_id="c_reg_80",
        document_id="doc_reg",
        text="1. Attendance: Students should attend at least 80% in aggregate (75% after 5 percentage-point relaxation).",
        metadata={"source_file": "LPU_CSE_Academic_Regulations.docx", "category": "regulations"},
        distance=0.1,
        similarity=0.9,
    )
    faq_chunk = RetrievedChunk(
        chunk_id="c_faq_013",
        document_id="LPU_CSE_AIML_Student_FAQs_FAQ013",
        text="Question: Minimum attendance?\nAnswer: This FAQ dataset does not yet contain a verified rule. Do not rely on an unverified percentage.",
        metadata={
            "source_file": "LPU_CSE_AIML_Student_FAQs.json",
            "category": "Attendance",
            "verification_status": "needs_verification",
        },
        distance=0.12,
        similarity=0.88,
    )

    with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[reg_chunk, faq_chunk]):
        with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
            resp = client.post(
                "/api/v1/chat/answer",
                json={"query": "Is the attendance threshold 75% or 80%?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["evidence_state"] == "conflicting"
            assert data["model_identifier"] == "deterministic_conflict_engine"
            assert data["generation_latency_ms"] == 0.0
            assert data["conflict_details"]["conflict_classification"] == "verification_caveat"
            assert data["conflict_details"]["competing_claims"] == []
            assert "verification caveat" in data["answer"].lower()
            assert "across official documents" not in data["answer"].lower()
            assert "student_contributed_faq" in data["answer"]
            assert ".." not in data["answer"]
            assert {c["chunk_id"] for c in data["citations"]} == {"c_reg_80", "c_faq_013"}
            assert {s["source_file"] for s in data["source_metadata"]} == {
                "LPU_CSE_Academic_Regulations.docx",
                "LPU_CSE_AIML_Student_FAQs.json",
            }
            mock_gen.assert_not_called()


def test_conflict_untraceable_or_missing_source_returns_safe_untraceable_response():
    """Filtered single-source conflict without opposing/caveat chunk must not invent claims or provenance."""
    reg_only = RetrievedChunk(
        chunk_id="c_reg_only",
        document_id="doc_reg",
        text="1. Attendance: Students should attend at least 80% in aggregate.",
        metadata={"source_file": "LPU_CSE_Academic_Regulations.docx", "category": "regulations"},
        distance=0.1,
        similarity=0.9,
    )

    with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[reg_only]):
        with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
            resp = client.post(
                "/api/v1/chat/answer",
                json={"query": "Is the attendance threshold 75% or 80%?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["evidence_state"] == "conflicting"
            assert data["conflict_details"]["conflict_classification"] == "untraceable_conflict"
            assert data["conflict_details"]["competing_claims"] == []
            assert "could not be traced" in data["answer"].lower()
            assert "FAQ013" not in data["answer"]
            assert "LPU_CSE_AIML_Student_FAQs.json" not in str(data["source_metadata"])
            mock_gen.assert_not_called()


def test_latency_telemetry_independent_measurement_and_exception_reset():
    """Vector retrieval and reranking latencies are measured independently and reset cleanly on error."""
    import time
    from app.rag.retrieval import RetrievalService, get_last_retrieval_timings

    dummy_chunk = RetrievedChunk(
        chunk_id="c_lat_1",
        document_id="doc_lat",
        text="CSE202 covers OOP in C++.",
        metadata={"source_file": "CSE202.pdf"},
        distance=0.1,
        similarity=0.9,
    )

    svc = RetrievalService(vector_store=object(), embedding_service=object())

    def slow_vector_search(*args, **kwargs):
        time.sleep(0.015)
        return [dummy_chunk]

    class SlowReranker:
        def rerank(self, query, candidates, top_k=None):
            time.sleep(0.060)
            return candidates

    svc._vector_search = slow_vector_search
    svc._reranker = SlowReranker()

    svc.retrieve("test query", top_k=1, enable_rerank=True)
    timings = get_last_retrieval_timings()
    assert timings is not None
    vec_ms, rerank_ms = timings
    assert vec_ms >= 10.0
    assert rerank_ms >= 50.0
    # Must NOT match artificial 15%/85% ratio when actual ratio is ~20%/80% or arbitrary
    assert abs(vec_ms - round((vec_ms + rerank_ms) * 0.15, 2)) > 0.5

    # Exception during retrieve() must reset timing state so stale timings are never reused
    class FailingReranker:
        def rerank(self, query, candidates, top_k=None):
            raise RuntimeError("simulated reranker crash")

    svc._reranker = FailingReranker()
    with pytest.raises(RuntimeError, match="simulated reranker crash"):
        svc.retrieve("test query", top_k=1, enable_rerank=True)
    assert get_last_retrieval_timings() is None


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://localhost:5174"])
def test_cors_preflight_options_chat_answer_allows_frontend_origins(origin):
    """Verify OPTIONS preflight on /api/v1/chat/answer allows POST and Content-Type for 5173 & 5174."""
    resp = client.options(
        "/api/v1/chat/answer",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    assert resp.headers.get("access-control-allow-credentials") == "true"
    allowed_methods = [m.strip().upper() for m in resp.headers.get("access-control-allow-methods", "").split(",")]
    assert "POST" in allowed_methods
    allowed_headers = [h.strip().lower() for h in resp.headers.get("access-control-allow-headers", "").split(",")]
    assert "content-type" in allowed_headers


def test_cors_post_chat_answer_returns_allow_origin_for_5174():
    """Verify actual POST /api/v1/chat/answer response includes Access-Control-Allow-Origin for localhost:5174."""
    with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
        resp = client.post(
            "/api/v1/chat/answer",
            headers={"Origin": "http://localhost:5174", "Content-Type": "application/json"},
            json={"query": "What is the fee refund percentage for 4th week admission withdrawal?"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5174"
        assert resp.headers.get("access-control-allow-credentials") == "true"
        mock_gen.assert_not_called()


def test_cors_preflight_rejects_unlisted_origin_and_wildcard_disallowed():
    """Verify unlisted origins are blocked and Settings.cors_origins never emits wildcard '*'."""
    from app.core.config import Settings

    resp = client.options(
        "/api/v1/chat/answer",
        headers={
            "Origin": "http://untrusted.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers

    s = Settings(allowed_origins="*,http://localhost:5174")
    assert "*" not in s.cors_origins
    assert s.cors_origins == ["http://localhost:5174"]


@pytest.mark.parametrize(
    "utterance,expected_phrase",
    [
        ("hi", "Hello!"),
        ("hello", "Hello!"),
        ("hey", "Hello!"),
        ("Hey!", "Hello!"),
        ("Good morning", "Good morning!"),
        ("good afternoon!", "Good afternoon!"),
        ("Good evening.", "Good evening!"),
        ("thanks", "You're welcome!"),
        ("thank you!", "You're welcome!"),
        ("bye", "Goodbye!"),
        ("goodbye.", "Goodbye!"),
    ],
)
def test_conversational_intents_bypass_retrieval_evidence_gate_and_llm(utterance, expected_phrase):
    """Standalone greetings, gratitude, and farewells bypass retrieval, reranking, EvidenceGate, and LLM."""
    with patch("app.rag.retrieval.RetrievalService.retrieve") as mock_retrieve:
        with patch("app.rag.evidence_gate.EvidenceGate.evaluate") as mock_gate:
            with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
                resp = client.post("/api/v1/chat/answer", json={"query": utterance})
                assert resp.status_code == 200
                data = resp.json()
                assert expected_phrase in data["answer"]
                assert data["model_identifier"] == "deterministic_conversational_handler"
                assert data["evidence_state"] == "sufficient"
                assert data["retrieval_latency_ms"] == 0.0
                assert data["reranking_latency_ms"] == 0.0
                assert data["generation_latency_ms"] == 0.0
                assert data["citations"] == []
                assert data["source_metadata"] == []
                assert data["citation_validation_status"] == "bypassed"
                assert data["citation_validation"]["status"] == "bypassed"
                mock_retrieve.assert_not_called()
                mock_gate.assert_not_called()
                mock_gen.assert_not_called()


@pytest.mark.parametrize(
    "academic_query",
    [
        "Hi, what is CSE202?",
        "Thanks, explain inheritance",
        "Hello, what are the prescribed textbooks for CSE202?",
        "Good morning, what is the attendance requirement?",
        "Bye, wait what topics are in CSE205?",
    ],
)
def test_ambiguous_greeting_prefixed_academic_queries_enter_rag_pipeline(academic_query):
    """Academic questions prefixed with greetings or gratitude must NOT be treated as conversational intents."""
    retrieved_chunk = RetrievedChunk(
        chunk_id="b84e8c8b020a1a9c",
        document_id="doc_cse202",
        text="CSE202 Object Oriented Programming covers C++ basics and inheritance.",
        metadata={"source_file": "CSE202.pdf", "category": "syllabus", "course_code": "CSE202"},
        distance=0.1,
        similarity=0.9,
    )
    sufficient_decision = EvidenceDecision(
        decision=EvidenceDecisionState.SUFFICIENT,
        reason_code="SUFFICIENT_OFFICIAL_EVIDENCE",
        explanation="Directly supported",
        heuristic_score=0.9,
        supporting_chunk_ids=["b84e8c8b020a1a9c"],
    )

    with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[retrieved_chunk]) as mock_retrieve:
        with patch("app.rag.evidence_gate.EvidenceGate.evaluate", return_value=sufficient_decision) as mock_gate:
            with patch(
                "app.services.llm_service.OllamaLLMService.generate",
                return_value=("CSE202 covers Object Oriented Programming and inheritance [[cite:b84e8c8b020a1a9c]].", {}),
            ) as mock_gen:
                resp = client.post("/api/v1/chat/answer", json={"query": academic_query})
                assert resp.status_code == 200
                data = resp.json()
                assert data["model_identifier"] != "deterministic_conversational_handler"
                assert data["citation_validation_status"] == "valid"
                mock_retrieve.assert_called_once()
                mock_gate.assert_called_once()
                mock_gen.assert_called_once()


@pytest.mark.parametrize(
    "utterance,expected_intent",
    [
        ("goood morning", "greeting"),
        ("Goodddd mornigggg buddy", "greeting"),
        ("heyyyy", "greeting"),
        ("hiiii buddy", "greeting"),
        ("hey bro", "greeting"),
        ("hello friend", "greeting"),
        ("thanks buddy", "gratitude"),
        ("thanks a lot bro", "gratitude"),
        ("goodbye for now friend", "farewell"),
        ("bye mate", "farewell"),
    ],
)
def test_typo_tolerant_and_repeated_character_greetings(utterance, expected_intent):
    """Verifies repeated-character collapse, bounded edit distance, and casual addressees."""
    from app.services.rag_service import detect_conversational_intent

    result = detect_conversational_intent(utterance)
    assert result is not None
    intent, reply = result
    assert intent == expected_intent

    with patch("app.rag.retrieval.RetrievalService.retrieve") as mock_retrieve:
        with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
            resp = client.post("/api/v1/chat/answer", json={"query": utterance})
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_identifier"] == "deterministic_conversational_handler"
            assert data["evidence_state"] == "sufficient"
            assert data["citations"] == []
            assert data["citation_validation_status"] == "bypassed"
            mock_retrieve.assert_not_called()
            mock_gen.assert_not_called()


@pytest.mark.parametrize(
    "academic_query",
    [
        "Hi, what is CSE202?",
        "Thanks, explain inheritance",
        "AI and ML",
        "Bye, explain the syllabus",
        "what is cse",
    ],
)
def test_academic_query_guard_disqualifies_conversational_handler(academic_query):
    """Academic questions, even with typos, greetings, or farewells, must never be classified as greetings."""
    from app.services.rag_service import detect_conversational_intent

    assert detect_conversational_intent(academic_query) is None


@pytest.mark.parametrize(
    "query,expected_stripped",
    [
        ("Thanks, explain inheritance", "explain inheritance"),
        ("Hello, explain CSE202", "explain CSE202"),
        ("Good morning, what are the CSE205 topics?", "what are the CSE205 topics?"),
        ("Bye, what is attendance policy?", "what is attendance policy?"),
        ("Thanks for explaining inheritance", "Thanks for explaining inheritance"),
        ("Thanks to the teacher", "Thanks to the teacher"),
        ("Hello explain CSE202", "Hello explain CSE202"),
        ("Good morning what are the CSE205 topics?", "Good morning what are the CSE205 topics?"),
    ],
)
def test_conversational_prefix_stripping_rules(query, expected_stripped):
    """Punctuation delimiter is strictly required for prefix stripping; bare whitespace never strips."""
    from app.services.rag_service import strip_conversational_prefix

    assert strip_conversational_prefix(query) == expected_stripped


def test_conversational_prefix_stripping_for_rag():
    """RAG pipeline must strip conversational prefixes before retrieval and reranking while preserving user query."""
    retrieved_chunk = RetrievedChunk(
        chunk_id="b84e8c8b020a1a9c",
        document_id="doc_cse202",
        text="CSE202 Object Oriented Programming covers C++ basics and inheritance.",
        metadata={"source_file": "CSE202.pdf", "category": "syllabus", "course_code": "CSE202", "verification_status": "verified_syllabus_structure"},
        distance=0.1,
        similarity=0.9,
    )
    sufficient_decision = EvidenceDecision(
        decision=EvidenceDecisionState.SUFFICIENT,
        reason_code="SUFFICIENT_OFFICIAL_EVIDENCE",
        explanation="Directly supported",
        heuristic_score=0.9,
        supporting_chunk_ids=["b84e8c8b020a1a9c"],
    )

    with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[retrieved_chunk]) as mock_retrieve:
        with patch("app.rag.evidence_gate.EvidenceGate.evaluate", return_value=sufficient_decision) as mock_gate:
            with patch(
                "app.services.llm_service.OllamaLLMService.generate",
                return_value=("CSE202 covers inheritance [[cite:b84e8c8b020a1a9c]].", {}),
            ):
                # 1. Punctuation delimiter present -> prefix stripped for retrieval
                resp = client.post("/api/v1/chat/answer", json={"query": "Thanks, explain inheritance"})
                assert resp.status_code == 200
                data = resp.json()
                assert data["query"] == "Thanks, explain inheritance"
                assert mock_retrieve.call_args.kwargs["query"] == "explain inheritance"
                assert mock_gate.call_args.kwargs["query"] == "explain inheritance"

                # 2. Bare whitespace delimiter without punctuation -> query remains unchanged for retrieval
                resp2 = client.post("/api/v1/chat/answer", json={"query": "Thanks for explaining inheritance"})
                assert resp2.status_code == 200
                data2 = resp2.json()
                assert data2["query"] == "Thanks for explaining inheritance"
                assert mock_retrieve.call_args.kwargs["query"] == "Thanks for explaining inheritance"
                assert mock_gate.call_args.kwargs["query"] == "Thanks for explaining inheritance"


def test_conceptual_question_clarification_on_insufficient_evidence():
    """When EvidenceGate returns INSUFFICIENT for a conceptual query whose topic is listed in a syllabus chunk,

    the chatbot returns a deterministic clarification without fabricating explanations or citations.
    """
    syllabus_chunk = RetrievedChunk(
        chunk_id="be775cdccf11f750",
        document_id="doc_cse202",
        text="Unit IV: Inheritance : Inheritance Basics - derived class and base class, Types, Modes.",
        metadata={
            "source_file": "CSE202.pdf",
            "category": "syllabus",
            "course_code": "CSE202",
            "verification_status": "verified_syllabus_structure",
        },
        distance=0.3,
        similarity=0.7,
        reranker_score=-6.74,
    )
    insufficient_decision = EvidenceDecision(
        decision=EvidenceDecisionState.INSUFFICIENT,
        reason_code="INSUFFICIENT_RELEVANCE",
        explanation="Retrieved chunks do not match the query intent (Cross-encoder score: -6.74).",
        heuristic_score=0.1,
        missing_information=["Full topic coverage is absent from academic documents."],
    )

    with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[syllabus_chunk]) as mock_retrieve:
        with patch("app.rag.evidence_gate.EvidenceGate.evaluate", return_value=insufficient_decision):
            with patch("app.services.llm_service.OllamaLLMService.generate") as mock_gen:
                resp = client.post("/api/v1/chat/answer", json={"query": "Thanks, explain inheritance"})
                assert resp.status_code == 200
                data = resp.json()
                assert data["query"] == "Thanks, explain inheritance"
                assert data["evidence_state"] == "insufficient"
                assert data["model_identifier"] == "deterministic_abstention_engine"
                assert data["citations"] == []
                assert data["citation_validation_status"] == "bypassed"
                # Must contain the deterministic syllabus clarification
                assert "CSE202 (CSE202.pdf)" in data["answer"]
                assert "Inheritance" in data["answer"]
                assert "curriculum topic" in data["answer"]
                assert "instructional textbook explanations" in data["answer"]
                # Must not have called LLM generate
                mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 11: Chat API Gemini Provider & Fallback Tests
# ---------------------------------------------------------------------------

def test_chat_api_gemini_provider_end_to_end_success():
    """When LLM_PROVIDER=gemini, Chat API generates answers using Gemini and enforces citation validation."""
    from app.services.llm_service import GeminiLLMService

    retrieved_chunk = RetrievedChunk(
        chunk_id="b84e8c8b020a1a9c",
        document_id="doc_cse202",
        text="CSE202 Object Oriented Programming covers C++ basics and inheritance.",
        metadata={"source_file": "CSE202.pdf", "category": "syllabus", "course_code": "CSE202", "verification_status": "verified_syllabus_structure"},
        distance=0.1,
        similarity=0.9,
    )
    sufficient_decision = EvidenceDecision(
        decision=EvidenceDecisionState.SUFFICIENT,
        reason_code="SUFFICIENT_OFFICIAL_EVIDENCE",
        explanation="Directly supported",
        heuristic_score=0.9,
        supporting_chunk_ids=["b84e8c8b020a1a9c"],
    )

    mock_gemini_resp = MagicMock(spec=httpx.Response)
    mock_gemini_resp.status_code = 200
    mock_gemini_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "CSE202 covers Object Oriented Programming and inheritance [[cite:b84e8c8b020a1a9c]]."}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"totalTokenCount": 85},
    }

    with patch("app.core.config.settings.llm_provider", "gemini"):
        with patch("app.core.config.settings.gemini_api_key", "test_gemini_api_key"):
            # Reset Gemini singleton for testing
            GeminiLLMService._instance = None
            with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[retrieved_chunk]):
                with patch("app.rag.evidence_gate.EvidenceGate.evaluate", return_value=sufficient_decision):
                    with patch("httpx.AsyncClient.post", return_value=mock_gemini_resp):
                        resp = client.post("/api/v1/chat/answer", json={"query": "What does CSE202 cover?"})
                        assert resp.status_code == 200
                        data = resp.json()
                        assert data["evidence_state"] == "sufficient"
                        assert data["model_identifier"] == "gemini-3.1-flash-lite"
                        assert data["citation_validation_status"] == "valid"
                        assert len(data["citations"]) == 1
                        assert data["citations"][0]["chunk_id"] == "b84e8c8b020a1a9c"
            GeminiLLMService._instance = None


def test_chat_api_gemini_citation_validation_enforced_and_regenerated():
    """Gemini output must pass through the identical CitationValidator, triggering regeneration if invalid."""
    from app.services.llm_service import GeminiLLMService

    retrieved_chunk = RetrievedChunk(
        chunk_id="b84e8c8b020a1a9c",
        document_id="doc_cse202",
        text="CSE202 Object Oriented Programming covers C++ basics and inheritance.",
        metadata={"source_file": "CSE202.pdf", "category": "syllabus", "course_code": "CSE202", "verification_status": "verified_syllabus_structure"},
        distance=0.1,
        similarity=0.9,
    )
    sufficient_decision = EvidenceDecision(
        decision=EvidenceDecisionState.SUFFICIENT,
        reason_code="SUFFICIENT_OFFICIAL_EVIDENCE",
        explanation="Directly supported",
        heuristic_score=0.9,
        supporting_chunk_ids=["b84e8c8b020a1a9c"],
    )

    # First attempt: no citation; Second attempt (regeneration): valid citation
    resp_invalid = MagicMock(spec=httpx.Response)
    resp_invalid.status_code = 200
    resp_invalid.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "CSE202 covers inheritance without any citations."}]}}],
        "usageMetadata": {"totalTokenCount": 40},
    }

    resp_valid = MagicMock(spec=httpx.Response)
    resp_valid.status_code = 200
    resp_valid.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "CSE202 covers inheritance [[cite:b84e8c8b020a1a9c]]."}]}}],
        "usageMetadata": {"totalTokenCount": 50},
    }

    with patch("app.core.config.settings.llm_provider", "gemini"):
        with patch("app.core.config.settings.gemini_api_key", "test_gemini_api_key"):
            GeminiLLMService._instance = None
            with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[retrieved_chunk]):
                with patch("app.rag.evidence_gate.EvidenceGate.evaluate", return_value=sufficient_decision):
                    with patch("httpx.AsyncClient.post", side_effect=[resp_invalid, resp_valid]) as mock_post:
                        resp = client.post("/api/v1/chat/answer", json={"query": "What does CSE202 cover?"})
                        assert resp.status_code == 200
                        data = resp.json()
                        assert data["citation_validation_status"] == "regenerated_valid"
                        assert mock_post.call_count == 2
            GeminiLLMService._instance = None


def test_chat_api_gemini_fallback_to_ollama_on_api_error():
    """When Gemini returns HTTP 429 quota exhausted, system transparently falls back to local Ollama."""
    from app.services.llm_service import GeminiLLMService

    retrieved_chunk = RetrievedChunk(
        chunk_id="b84e8c8b020a1a9c",
        document_id="doc_cse202",
        text="CSE202 Object Oriented Programming covers C++ basics and inheritance.",
        metadata={"source_file": "CSE202.pdf", "category": "syllabus", "course_code": "CSE202", "verification_status": "verified_syllabus_structure"},
        distance=0.1,
        similarity=0.9,
    )
    sufficient_decision = EvidenceDecision(
        decision=EvidenceDecisionState.SUFFICIENT,
        reason_code="SUFFICIENT_OFFICIAL_EVIDENCE",
        explanation="Directly supported",
        heuristic_score=0.9,
        supporting_chunk_ids=["b84e8c8b020a1a9c"],
    )

    mock_gemini_429 = MagicMock(spec=httpx.Response)
    mock_gemini_429.status_code = 429
    mock_gemini_429.text = "RESOURCE_EXHAUSTED: Rate limit exceeded"

    with patch("app.core.config.settings.llm_provider", "gemini"):
        with patch("app.core.config.settings.gemini_api_key", "test_gemini_api_key"):
            with patch("app.core.config.settings.gemini_fallback_to_local", True):
                GeminiLLMService._instance = None
                with patch("app.rag.retrieval.RetrievalService.retrieve", return_value=[retrieved_chunk]):
                    with patch("app.rag.evidence_gate.EvidenceGate.evaluate", return_value=sufficient_decision):
                        with patch("httpx.AsyncClient.post", return_value=mock_gemini_429):
                            with patch(
                                "app.services.llm_service.OllamaLLMService.generate",
                                return_value=("Local Ollama fallback response [[cite:b84e8c8b020a1a9c]].", {"provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"}),
                            ) as mock_ollama_gen:
                                resp = client.post("/api/v1/chat/answer", json={"query": "What does CSE202 cover?"})
                                assert resp.status_code == 200
                                data = resp.json()
                                assert data["model_identifier"] == "qwen3:4b-instruct-2507-q4_K_M"
                                assert data["citation_validation_status"] == "valid"
                                assert "Local Ollama fallback response" in data["answer"]
                                mock_ollama_gen.assert_called_once()
                GeminiLLMService._instance = None
