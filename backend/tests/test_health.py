"""Tests for the service health check endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    """Verify that GET /health responds with 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "service" in payload
    assert "version" in payload
    assert "timestamp" in payload


def test_root_endpoint_returns_200():
    """Verify that GET / responds with 200 and links."""
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert "docs" in payload
    assert "health" in payload
