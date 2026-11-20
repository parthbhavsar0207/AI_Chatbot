"""API tests for the FastAPI retrieval and semantic search endpoint."""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.services.retrieval_service import (
    RetrievalServiceManager,
    get_retrieval_service,
)


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance with lifespan events properly triggered."""
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# 1. Successful Searches & Filtering
# ==============================================================================


def test_search_unfiltered_success(client: TestClient):
    """Verify that POST /api/v1/retrieval/search succeeds with valid query and default top_k."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "What is the attendance policy for examinations?"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "What is the attendance policy for examinations?"
    assert data["top_k"] == 5
    assert data["total_results"] > 0
    assert len(data["results"]) == data["total_results"]
    assert data["retrieval_time_ms"] > 0.0

    first_item = data["results"][0]
    assert "chunk_id" in first_item
    assert "document_id" in first_item
    assert "text" in first_item
    assert len(first_item["text"]) > 0
    assert "source_file" in first_item
    assert "category" in first_item
    assert "similarity" in first_item
    assert "distance" in first_item
    assert "metadata" in first_item
    assert isinstance(first_item["metadata"], dict)
    assert first_item["similarity"] <= 1.0


def test_search_custom_top_k(client: TestClient):
    """Verify that specifying a custom top_k returns exactly min(top_k, total_available) items."""
    k = 2
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "attendance", "top_k": k},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["top_k"] == k
    assert len(data["results"]) == k
    assert data["total_results"] == k


def test_search_category_filter(client: TestClient):
    """Verify that filtering by category returns chunks belonging only to that category."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "attendance criteria",
            "category": "regulations",
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] > 0
    for item in data["results"]:
        assert item["category"].lower() == "regulations"


def test_search_course_code_filter(client: TestClient):
    """Verify that filtering by course_code returns chunks belonging only to that course."""
    course = "CSE306"
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "course syllabus and outcomes",
            "course_code": course,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] > 0
    for item in data["results"]:
        assert item["course_code"] == course


def test_search_combined_filters(client: TestClient):
    """Verify combined category and course_code filtering."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "course objectives",
            "category": "syllabus",
            "course_code": "CSE202",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] > 0
    for item in data["results"]:
        assert item["category"].lower() == "syllabus"
        assert item["course_code"] == "CSE202"


def test_search_source_file_filter(client: TestClient):
    """Verify filtering by specific source_file."""
    source_file = "lpu_cse_aiml_upcoming_end_term_examinations_2026-27.docx"
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "examination timings",
            "source_file": source_file,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] > 0
    for item in data["results"]:
        assert item["source_file"] == source_file


def test_search_empty_result_set(client: TestClient):
    """Verify that a valid filter with no matching chunks returns 200 with an empty results list."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "examination timings",
            "source_file": "non_existent_document_sample.pdf",
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 0
    assert data["results"] == []


# ==============================================================================
# 2. Input Validation and Error Handling (422)
# ==============================================================================


def test_search_empty_query_rejected(client: TestClient):
    """Verify that an empty string query returns HTTP 422."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": ""},
    )
    assert response.status_code == 422


def test_search_whitespace_query_rejected(client: TestClient):
    """Verify that a whitespace-only query returns HTTP 422."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "    \t\n  "},
    )
    assert response.status_code == 422


def test_search_missing_query_rejected(client: TestClient):
    """Verify that omitting the query field returns HTTP 422."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"top_k": 5},
    )
    assert response.status_code == 422


def test_search_invalid_top_k_zero(client: TestClient):
    """Verify that top_k = 0 returns HTTP 422."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "attendance", "top_k": 0},
    )
    assert response.status_code == 422


def test_search_invalid_top_k_negative(client: TestClient):
    """Verify that negative top_k returns HTTP 422."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "attendance", "top_k": -5},
    )
    assert response.status_code == 422


def test_search_invalid_top_k_exceeds_max(client: TestClient):
    """Verify that top_k exceeding the maximum threshold (50) returns HTTP 422."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "attendance", "top_k": 100},
    )
    assert response.status_code == 422


def test_search_unsupported_category_rejected(client: TestClient):
    """Verify that an unsupported category returns HTTP 422."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "attendance", "category": "non_existent_category_xyz"},
    )
    assert response.status_code == 422
    assert "Unsupported category" in response.text


def test_search_invalid_course_code_rejected(client: TestClient):
    """Verify that an invalid course code returns HTTP 422."""
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "passing criteria", "course_code": "BIO999"},
    )
    assert response.status_code == 422
    assert "Invalid course code" in response.text


def test_search_malformed_json_body(client: TestClient):
    """Verify that invalid request body format returns HTTP 422."""
    response = client.post(
        "/api/v1/retrieval/search",
        content="not a json string",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ==============================================================================
# 3. Service Failure Handling (500)
# ==============================================================================


def test_search_service_failure_returns_500(client: TestClient):
    """Verify that unexpected service errors yield a sanitized HTTP 500 without leaking stack traces."""
    mock_service = MagicMock()
    mock_service.retrieve.side_effect = RuntimeError("Fatal low-level ChromaDB disk I/O error at /internal/path")

    # Override the retrieval service dependency
    app.dependency_overrides[get_retrieval_service] = lambda: mock_service
    try:
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "What is the attendance policy?"},
        )
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert data["detail"] == "An internal error occurred while processing the retrieval request."
        # Verify sensitive internals/paths are not exposed
        assert "/internal/path" not in response.text
        assert "Fatal low-level" not in response.text
    finally:
        app.dependency_overrides.clear()


# ==============================================================================
# 4. Lifecycle & Singleton Verification
# ==============================================================================


def test_singleton_retrieval_service_identity():
    """Verify that get_retrieval_service returns the same singleton across invocations."""
    service_a = get_retrieval_service()
    service_b = get_retrieval_service()
    assert service_a is service_b
    assert service_a.vector_store is service_b.vector_store
    assert service_a.embedding_service is service_b.embedding_service


# ==============================================================================
# 5. OpenAPI Documentation Verification
# ==============================================================================


def test_openapi_documentation():
    """Verify that OpenAPI schema includes /api/v1/retrieval/search with proper documentation."""
    openapi_schema = app.openapi()
    assert "paths" in openapi_schema
    assert "/api/v1/retrieval/search" in openapi_schema["paths"]

    search_endpoint = openapi_schema["paths"]["/api/v1/retrieval/search"]["post"]
    assert search_endpoint["summary"] == "Semantic Retrieval Search"
    assert "Retrieval" in search_endpoint["tags"]
    assert "200" in search_endpoint["responses"]
    assert "422" in search_endpoint["responses"]
    assert "500" in search_endpoint["responses"]
