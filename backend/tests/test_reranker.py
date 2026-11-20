"""Unit and integration tests for CrossEncoderReranker and Reranked Retrieval."""

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.rag.reranker import CrossEncoderReranker
from app.rag.retrieval import RetrievalService, RetrievedChunk
from app.services.retrieval_service import get_retrieval_service


@pytest.fixture(scope="module")
def reranker():
    """Provides a shared CrossEncoderReranker instance for tests."""
    return CrossEncoderReranker()


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def _make_chunk(chunk_id: str, text: str, source_file: str = "doc.pdf", similarity: float = 0.8) -> RetrievedChunk:
    """Helper to construct a mock RetrievedChunk."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=text,
        metadata={"source_file": source_file, "category": "syllabus", "custom_key": "custom_val"},
        distance=round(1.0 - similarity, 6),
        similarity=similarity,
    )


# ==============================================================================
# 1. Reranker Core Tests
# ==============================================================================


def test_reranker_ordering(reranker: CrossEncoderReranker):
    """Verify that cross-encoder correctly orders relevant document ahead of irrelevant document."""
    query = "What is the attendance requirement for examinations?"
    chunk_irrelevant = _make_chunk("c1", "The hostel fee for 2026-27 is non-refundable after semester commencement.")
    chunk_relevant = _make_chunk("c2", "Students must maintain a minimum of 75% attendance to sit for final examinations.")

    # Pass in irrelevant first
    candidates = [chunk_irrelevant, chunk_relevant]
    reranked = reranker.rerank(query=query, candidates=candidates)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"  # Relevant chunk promoted to #1
    assert reranked[1].chunk_id == "c1"
    assert reranked[0].reranker_score > reranked[1].reranker_score


def test_reranker_score_handling(reranker: CrossEncoderReranker):
    """Verify that reranker scores are populated, sorted descending, and finite floats."""
    query = "operating system course outcomes"
    c1 = _make_chunk("c1", "Unit 1 covers processes, threads, CPU scheduling and synchronization.")
    c2 = _make_chunk("c2", "Examination seating arrangements will be posted on university notice board.")

    reranked = reranker.rerank(query=query, candidates=[c1, c2])
    assert all(isinstance(c.reranker_score, float) for c in reranked)
    assert reranked[0].reranker_score >= reranked[1].reranker_score


def test_reranker_empty_candidates(reranker: CrossEncoderReranker):
    """Verify that an empty candidate list returns empty without error."""
    result = reranker.rerank(query="any query", candidates=[])
    assert result == []


def test_reranker_metadata_preservation(reranker: CrossEncoderReranker):
    """Verify that metadata, document_id, distance, and similarity are strictly preserved."""
    query = "What is graph theory in MTH401?"
    chunk = _make_chunk("c10", "Graph theory covers trees, cycles, Eulerian paths and planar graphs.", similarity=0.88)
    reranked = reranker.rerank(query=query, candidates=[chunk])

    assert len(reranked) == 1
    r = reranked[0]
    assert r.chunk_id == "c10"
    assert r.document_id == "doc_c10"
    assert r.distance == 0.12
    assert r.similarity == 0.88
    assert r.metadata["source_file"] == "doc.pdf"
    assert r.metadata["custom_key"] == "custom_val"


def test_original_rank_and_new_rank(reranker: CrossEncoderReranker):
    """Verify that original vector search rank and reranked rank are accurately recorded."""
    query = "passing marks for CSE205"
    c1 = _make_chunk("c1", "Hostel room allocation is strictly based on registration timestamp.")
    c2 = _make_chunk("c2", "The minimum passing marks for CSE205 Data Structures is 40% aggregate.")

    candidates = [c1, c2]  # c1 is original rank 1, c2 is original rank 2
    reranked = reranker.rerank(query=query, candidates=candidates)

    promoted_chunk = next(c for c in reranked if c.chunk_id == "c2")
    demoted_chunk = next(c for c in reranked if c.chunk_id == "c1")

    assert promoted_chunk.original_rank == 2
    assert promoted_chunk.reranked_rank == 1

    assert demoted_chunk.original_rank == 1
    assert demoted_chunk.reranked_rank == 2


def test_reranker_invalid_inputs(reranker: CrossEncoderReranker):
    """Verify that invalid inputs raise ValueError."""
    c = _make_chunk("c1", "sample text")

    with pytest.raises(ValueError, match="empty or whitespace"):
        reranker.rerank(query="", candidates=[c])

    with pytest.raises(ValueError, match="empty or whitespace"):
        reranker.rerank(query="   \t\n ", candidates=[c])

    with pytest.raises(ValueError, match="positive integer"):
        reranker.rerank(query="valid query", candidates=[c], top_k=0)

    with pytest.raises(ValueError, match="positive integer"):
        reranker.rerank(query="valid query", candidates=[c], top_k=-3)


# ==============================================================================
# 2. RetrievalService Integration Tests
# ==============================================================================


def test_retrieval_service_rerank_disabled():
    """Verify that retrieval with enable_rerank=False preserves baseline vector rankings."""
    service = get_retrieval_service()
    results = service.retrieve("attendance policy", top_k=3, enable_rerank=False)

    assert len(results) > 0
    for idx, r in enumerate(results, 1):
        assert r.original_rank == idx
        assert r.reranker_score is None
        assert r.reranked_rank is None


def test_retrieval_service_rerank_enabled():
    """Verify that retrieval with enable_rerank=True executes cross-encoder reranking."""
    service = get_retrieval_service()
    results = service.retrieve("attendance policy", top_k=3, enable_rerank=True, candidate_k=6)

    assert len(results) == 3
    for idx, r in enumerate(results, 1):
        assert r.original_rank is not None
        assert r.reranker_score is not None
        assert r.reranked_rank == idx


# ==============================================================================
# 3. API Compatibility Tests
# ==============================================================================


def test_api_search_with_rerank_disabled(client: TestClient):
    """Verify API default behavior (reranking disabled)."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "attendance policy for examinations", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["rerank_enabled"] is False
    assert len(data["results"]) == 3
    for item in data["results"]:
        assert item["reranker_score"] is None
        assert item["reranked_rank"] is None
        assert item["original_rank"] is not None


def test_api_search_with_rerank_enabled(client: TestClient):
    """Verify API behavior when enable_rerank is explicitly True."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "attendance policy for examinations",
            "top_k": 3,
            "enable_rerank": True,
            "candidate_k": 8,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["rerank_enabled"] is True
    assert data["candidate_k"] == 8
    assert len(data["results"]) == 3

    for idx, item in enumerate(data["results"], 1):
        assert item["reranker_score"] is not None
        assert item["original_rank"] is not None
        assert item["reranked_rank"] == idx


def test_api_search_rerank_filtered(client: TestClient):
    """Verify that reranking works smoothly with metadata filters."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "passing criteria",
            "course_code": "CSE306",
            "enable_rerank": True,
            "top_k": 2,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["rerank_enabled"] is True
    for item in data["results"]:
        assert item["course_code"] == "CSE306"
        assert item["reranker_score"] is not None
