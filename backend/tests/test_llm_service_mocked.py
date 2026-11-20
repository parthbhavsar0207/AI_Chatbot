"""Unit tests for OllamaLLMService and GeminiLLMService using mocked HTTP responses (Phases 10.3 & 11)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.services.llm_service import (
    BaseLLMService,
    GeminiAPIError,
    GeminiAuthError,
    GeminiLLMService,
    GeminiMalformedResponseError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    ModelUnavailableError,
    OllamaLLMService,
    OllamaMalformedResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    get_llm_service,
)


@pytest.mark.anyio
async def test_ollama_unavailable_raises_clean_error():
    service = OllamaLLMService(base_url="http://127.0.0.1:9999", timeout_seconds=1.0)
    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(OllamaUnavailableError) as exc_info:
            await service.generate("Test prompt")
        assert "Cannot connect to local Ollama service" in str(exc_info.value)
        assert exc_info.value.detail == "Local LLM service is offline or unreachable at 127.0.0.1:11434."


@pytest.mark.anyio
async def test_model_404_raises_model_unavailable_error():
    service = OllamaLLMService(primary_model="non_existent_model_tag")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 404
    mock_resp.text = "model 'non_existent_model_tag' not found"

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(ModelUnavailableError) as exc_info:
            await service.generate("Test prompt")
        assert "not found in the Ollama registry" in exc_info.value.detail


@pytest.mark.anyio
async def test_timeout_raises_ollama_timeout_error():
    service = OllamaLLMService(timeout_seconds=0.1)
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timed out")):
        with pytest.raises(OllamaTimeoutError) as exc_info:
            await service.generate("Test prompt")
        assert "took longer than the configured timeout limit" in exc_info.value.detail


@pytest.mark.anyio
async def test_malformed_json_raises_malformed_error():
    service = OllamaLLMService()
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = "NOT JSON"
    mock_resp.json.side_effect = ValueError("Invalid JSON")

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(OllamaMalformedResponseError) as exc_info:
            await service.generate("Test prompt")
        assert "malformed response body" in exc_info.value.detail


@pytest.mark.anyio
async def test_successful_generation_parses_response_and_metrics():
    service = OllamaLLMService()
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "Answer text [[cite:chunk_123]].",
        "total_duration": 1500000000,
        "load_duration": 10000000,
        "prompt_eval_count": 50,
        "prompt_eval_duration": 400000000,
        "eval_count": 30,
        "eval_duration": 1000000000,
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        text, metrics = await service.generate("Test prompt")
        assert text == "Answer text [[cite:chunk_123]]."
        assert metrics["eval_count"] == 30
        assert metrics["total_duration_ms"] == 1500.0


@pytest.mark.anyio
async def test_single_concurrency_semaphore():
    service = OllamaLLMService(max_concurrency=1)
    active_requests = 0
    max_active = 0

    async def fake_post(*args, **kwargs):
        nonlocal active_requests, max_active
        active_requests += 1
        max_active = max(max_active, active_requests)
        await asyncio.sleep(0.05)
        active_requests -= 1
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"response": "done"}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        tasks = [service.generate(f"Prompt {i}") for i in range(3)]
        await asyncio.gather(*tasks)

    # Max concurrent executions inside the critical section must not exceed 1
    assert max_active == 1


# ---------------------------------------------------------------------------
# Phase 11: Google Gemini LLM Service Unit Tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_gemini_missing_api_key_raises_auth_error_without_fallback():
    with patch("app.core.config.settings.gemini_api_key", None):
        service = GeminiLLMService(api_key=None, fallback_service=None)
        with pytest.raises(GeminiAuthError) as exc_info:
            await service.generate("Test prompt")
        assert "GEMINI_API_KEY is not configured" in str(exc_info.value)
        assert "missing" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_successful_generation_parses_response_and_usage_metrics():
    service = GeminiLLMService(api_key="mock_key_123", fallback_service=None)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Gemini answer with citation [[cite:chunk_001]]."}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 45,
            "totalTokenCount": 165,
        },
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp) as mock_post:
        text, metrics = await service.generate("Grounded prompt", system_prompt="System prompt")
        assert text == "Gemini answer with citation [[cite:chunk_001]]."
        assert metrics["provider"] == "gemini"
        assert metrics["model"] == "gemini-3.1-flash-lite"
        assert metrics["total_token_count"] == 165
        assert mock_post.call_count == 1
        # API key must be in x-goog-api-key header
        headers = mock_post.call_args.kwargs.get("headers", {})
        assert headers.get("x-goog-api-key") == "mock_key_123"


@pytest.mark.anyio
async def test_gemini_malformed_json_raises_clean_error():
    service = GeminiLLMService(api_key="mock_key_123", fallback_service=None)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = "INVALID JSON"
    mock_resp.json.side_effect = ValueError("JSONDecodeError")

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(GeminiMalformedResponseError) as exc_info:
            await service.generate("Test prompt")
        assert "unparseable response body" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_zero_candidates_raises_clean_error():
    service = GeminiLLMService(api_key="mock_key_123", fallback_service=None)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = '{"candidates": []}'
    mock_resp.json.return_value = {"candidates": []}

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(GeminiMalformedResponseError) as exc_info:
            await service.generate("Test prompt")
        assert "empty candidate list" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_429_rate_limit_raises_rate_limit_error():
    service = GeminiLLMService(api_key="mock_key_123", fallback_service=None)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.text = "RESOURCE_EXHAUSTED: Quota exceeded"

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(GeminiRateLimitError) as exc_info:
            await service.generate("Test prompt")
        assert "free tier rate limit was reached" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_401_invalid_key_raises_auth_error():
    service = GeminiLLMService(api_key="invalid_key", fallback_service=None)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.text = "API_KEY_INVALID"

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(GeminiAuthError) as exc_info:
            await service.generate("Test prompt")
        assert "client error 401" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_500_server_error_raises_api_error():
    service = GeminiLLMService(api_key="mock_key_123", fallback_service=None)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.text = "Internal error"

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(GeminiAPIError) as exc_info:
            await service.generate("Test prompt")
        assert "server error (500)" in exc_info.value.message


@pytest.mark.anyio
async def test_gemini_timeout_raises_timeout_error():
    service = GeminiLLMService(api_key="mock_key_123", timeout_seconds=0.1, fallback_service=None)
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timed out")):
        with pytest.raises(GeminiTimeoutError) as exc_info:
            await service.generate("Test prompt")
        assert "timed out" in str(exc_info.value)
        assert "timeout limit" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_connect_error_raises_unavailable_error():
    service = GeminiLLMService(api_key="mock_key_123", fallback_service=None)
    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Failed to connect")):
        with pytest.raises(GeminiUnavailableError) as exc_info:
            await service.generate("Test prompt")
        assert "Network connection error" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_fallback_to_local_ollama_on_429():
    mock_fallback = AsyncMock(spec=BaseLLMService)
    mock_fallback.primary_model = "qwen3:4b-instruct-2507-q4_K_M"
    mock_fallback.generate.return_value = (
        "Local fallback answer [[cite:chunk_002]].",
        {"provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
    )

    service = GeminiLLMService(api_key="mock_key_123", fallback_service=mock_fallback)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.text = "Quota exceeded"

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        text, metrics = await service.generate("Test prompt")
        assert text == "Local fallback answer [[cite:chunk_002]]."
        assert metrics["fallback_from"] == "gemini"
        assert metrics["fallback_reason"] == "rate_limit_429"
        mock_fallback.generate.assert_called_once()


@pytest.mark.anyio
async def test_gemini_fallback_to_local_ollama_on_timeout():
    mock_fallback = AsyncMock(spec=BaseLLMService)
    mock_fallback.primary_model = "qwen3:4b-instruct-2507-q4_K_M"
    mock_fallback.generate.return_value = (
        "Fallback answer after timeout [[cite:chunk_003]].",
        {"provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
    )

    service = GeminiLLMService(api_key="mock_key_123", fallback_service=mock_fallback)
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timed out")):
        text, metrics = await service.generate("Test prompt")
        assert text == "Fallback answer after timeout [[cite:chunk_003]]."
        assert metrics["fallback_from"] == "gemini"
        assert metrics["fallback_reason"] == "timeout"
        mock_fallback.generate.assert_called_once()


@pytest.mark.anyio
async def test_gemini_fallback_to_local_ollama_on_missing_api_key():
    mock_fallback = AsyncMock(spec=BaseLLMService)
    mock_fallback.primary_model = "qwen3:4b-instruct-2507-q4_K_M"
    mock_fallback.generate.return_value = (
        "Fallback answer for missing key [[cite:chunk_004]].",
        {"provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
    )

    with patch("app.core.config.settings.gemini_api_key", None):
        service = GeminiLLMService(api_key=None, fallback_service=mock_fallback)
        text, metrics = await service.generate("Test prompt")
        assert text == "Fallback answer for missing key [[cite:chunk_004]]."
        assert metrics["fallback_from"] == "gemini"
        assert metrics["fallback_reason"] == "missing_api_key"
        mock_fallback.generate.assert_called_once()


def test_provider_factory_selection():
    from app.services.llm_service import reset_llm_service_instances

    reset_llm_service_instances()
    with patch("app.core.config.settings.llm_provider", "ollama"):
        srv = get_llm_service()
        assert isinstance(srv, OllamaLLMService)

    with patch("app.core.config.settings.llm_provider", "gemini"):
        with patch("app.core.config.settings.gemini_fallback_to_local", False):
            reset_llm_service_instances()
            srv_gemini = get_llm_service()
            assert isinstance(srv_gemini, GeminiLLMService)
            assert srv_gemini.primary_model == "gemini-3.1-flash-lite"
            reset_llm_service_instances()


@pytest.mark.anyio
async def test_gemini_503_transient_retry_success():
    service = GeminiLLMService(api_key="mock_key_123", fallback_service=None)

    resp_503 = MagicMock(spec=httpx.Response)
    resp_503.status_code = 503
    resp_503.text = '{"error": {"code": 503, "message": "High demand", "status": "UNAVAILABLE"}}'

    resp_200 = MagicMock(spec=httpx.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Success after retry [[cite:chunk_001]]."}],
                    "role": "model",
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 15,
            "totalTokenCount": 35,
        },
    }

    with patch("httpx.AsyncClient.post", side_effect=[resp_503, resp_200]) as mock_post:
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            text, metrics = await service.generate("Test prompt")
            assert text == "Success after retry [[cite:chunk_001]]."
            assert metrics["provider"] == "gemini"
            assert metrics["retries"] == 1
            assert mock_post.call_count == 2
            mock_sleep.assert_called_once_with(2.0)


@pytest.mark.anyio
async def test_gemini_503_repeated_failures_raises_api_error_without_fallback():
    service = GeminiLLMService(api_key="mock_key_123", fallback_service=None)

    resp_503 = MagicMock(spec=httpx.Response)
    resp_503.status_code = 503
    resp_503.text = '{"error": {"code": 503, "message": "Overloaded", "status": "UNAVAILABLE"}}'

    with patch("httpx.AsyncClient.post", side_effect=[resp_503, resp_503, resp_503]) as mock_post:
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(GeminiAPIError) as exc_info:
                await service.generate("Test prompt")
            assert "server error (503) after 3 attempts" in str(exc_info.value)
            assert mock_post.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(2.0)
            mock_sleep.assert_any_call(4.0)


@pytest.mark.anyio
async def test_gemini_503_repeated_failures_triggers_ollama_fallback():
    mock_fallback = AsyncMock(spec=BaseLLMService)
    mock_fallback.primary_model = "qwen3:4b-instruct-2507-q4_K_M"
    mock_fallback.generate.return_value = (
        "Fallback answer after 503 exhaustion [[cite:chunk_005]].",
        {"provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
    )

    service = GeminiLLMService(api_key="mock_key_123", fallback_service=mock_fallback)

    resp_503 = MagicMock(spec=httpx.Response)
    resp_503.status_code = 503
    resp_503.text = '{"error": {"code": 503, "message": "High demand", "status": "UNAVAILABLE"}}'

    with patch("httpx.AsyncClient.post", side_effect=[resp_503, resp_503, resp_503]) as mock_post:
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            text, metrics = await service.generate("Test prompt")
            assert text == "Fallback answer after 503 exhaustion [[cite:chunk_005]]."
            assert metrics["fallback_from"] == "gemini"
            assert metrics["fallback_reason"] == "server_error_503"
            assert mock_post.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(2.0)
            mock_sleep.assert_any_call(4.0)
            mock_fallback.generate.assert_called_once()


@pytest.mark.anyio
async def test_gemini_auth_and_rate_limit_do_not_retry():
    service = GeminiLLMService(api_key="mock_key_123", fallback_service=None)

    resp_401 = MagicMock(spec=httpx.Response)
    resp_401.status_code = 401
    resp_401.text = "Unauthorized"

    with patch("httpx.AsyncClient.post", return_value=resp_401) as mock_post:
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(GeminiAuthError):
                await service.generate("Test prompt")
            assert mock_post.call_count == 1
            assert mock_sleep.call_count == 0

    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    resp_429.text = "Rate limited"

    with patch("httpx.AsyncClient.post", return_value=resp_429) as mock_post:
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(GeminiRateLimitError):
                await service.generate("Test prompt")
            assert mock_post.call_count == 1
            assert mock_sleep.call_count == 0

