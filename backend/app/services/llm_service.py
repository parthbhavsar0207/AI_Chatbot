"""Isolated asynchronous LLM service supporting local Ollama and Google Gemini (Phases 10.3 & 11).

Provides non-blocking HTTP communication enforcing:
- Async I/O via httpx.AsyncClient (never blocks FastAPI event loop).
- Minimal BaseLLMService interface implemented by OllamaLLMService and GeminiLLMService.
- Single-request concurrency limit via asyncio.Semaphore to protect GPU VRAM (Ollama)
  and rate limits (Gemini free tier).
- Configurable timeouts and optional local Ollama fallback for Gemini API failures.
- Structured, sanitized domain exceptions for all error conditions (API keys never logged).
- Zero external framework dependencies (no LangChain, no LlamaIndex, no PyTorch, no extra SDKs).
"""

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any, Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Sanitized Domain Exceptions
# ---------------------------------------------------------------------------

class LLMServiceError(Exception):
    """Base exception for all LLM service failures."""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or message


# Ollama-specific exceptions
class OllamaUnavailableError(LLMServiceError):
    """Raised when the local Ollama daemon cannot be reached."""
    pass


class ModelUnavailableError(LLMServiceError):
    """Raised when the requested model is not found in the Ollama registry."""
    pass


class OllamaTimeoutError(LLMServiceError):
    """Raised when an inference request exceeds the configured timeout."""
    pass


class OllamaMalformedResponseError(LLMServiceError):
    """Raised when Ollama returns an unexpected or invalid JSON response."""
    pass


# Gemini-specific exceptions (Phase 11)
class GeminiAPIError(LLMServiceError):
    """Raised when Google Gemini API encounters a server or protocol error."""
    pass


class GeminiUnavailableError(LLMServiceError):
    """Raised when Google Gemini API endpoint cannot be reached."""
    pass


class GeminiAuthError(LLMServiceError):
    """Raised when Gemini API key is missing or invalid."""
    pass


class GeminiRateLimitError(LLMServiceError):
    """Raised when Gemini API rate limit or quota is exceeded (HTTP 429)."""
    pass


class GeminiTimeoutError(LLMServiceError):
    """Raised when a Gemini API inference request exceeds the timeout limit."""
    pass


class GeminiMalformedResponseError(LLMServiceError):
    """Raised when Gemini returns an unexpected or invalid JSON response."""
    pass


# ---------------------------------------------------------------------------
# Common LLM Service Base Interface
# ---------------------------------------------------------------------------

class BaseLLMService(ABC):
    """Abstract base class for grounded LLM inference providers."""

    @property
    @abstractmethod
    def primary_model(self) -> str:
        """Primary model identifier for provenance and telemetry."""
        ...

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Checks connectivity and service health."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Executes non-streaming completion.

        Returns:
            tuple[str, dict[str, Any]]: Generated answer text and provider telemetry metrics.
        """
        ...


# ---------------------------------------------------------------------------
# Local Ollama Implementation
# ---------------------------------------------------------------------------

class OllamaLLMService(BaseLLMService):
    """Asynchronous client for local Ollama LLM inference."""

    _instance: Optional["OllamaLLMService"] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        primary_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        num_ctx: Optional[int] = None,
        temperature: Optional[float] = None,
        keep_alive: Optional[str] = None,
        max_concurrency: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._primary_model = primary_model or settings.llm_primary_model
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self.num_ctx = num_ctx or settings.llm_num_ctx
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.keep_alive = keep_alive or settings.llm_keep_alive
        self.max_concurrency = max_concurrency or settings.llm_max_concurrency

        # Strict concurrency limiter to guard RTX 4050 6GB VRAM
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    @property
    def primary_model(self) -> str:
        return self._primary_model

    @classmethod
    def get_instance(cls) -> "OllamaLLMService":
        """Returns the singleton LLM service instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def check_health(self) -> dict[str, Any]:
        """Checks connectivity to Ollama and verifies installed models."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    raise OllamaUnavailableError(
                        f"Ollama server returned status {resp.status_code}",
                        detail="Local LLM runtime is returning an unexpected status.",
                    )
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                primary_available = any(self.primary_model in m for m in models)
                return {
                    "status": "healthy",
                    "available_models": models,
                    "primary_model": self.primary_model,
                    "primary_model_ready": primary_available,
                }
        except httpx.ConnectError as exc:
            logger.error("Failed to connect to Ollama daemon at %s: %s", self.base_url, exc)
            raise OllamaUnavailableError(
                "Unable to connect to Ollama local service.",
                detail="Local LLM service is offline or unreachable at 127.0.0.1:11434.",
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Timeout checking Ollama health at %s", self.base_url)
            raise OllamaTimeoutError(
                "Ollama health check timed out.",
                detail="Local LLM service did not respond within health check timeout.",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error during Ollama health check: %s", exc)
            raise OllamaUnavailableError(
                "Ollama service health check failed.",
                detail="An internal communication error occurred with the local LLM runtime.",
            ) from exc

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Executes non-streaming completion with local Ollama model.

        Args:
            prompt: User-facing or RAG-contextualized prompt text.
            system_prompt: Optional system instruction prompt.
            model: Optional model override; defaults to self.primary_model.
            temperature: Optional sampling temperature override; defaults to 0.0.
            num_ctx: Optional context length override; defaults to 4096.

        Returns:
            tuple[str, dict[str, Any]]: Generated text string and server-side performance metrics.

        Raises:
            OllamaUnavailableError: If Ollama daemon is offline or connection refused.
            ModelUnavailableError: If the specified model tag does not exist.
            OllamaTimeoutError: If generation exceeds configured request timeout.
            OllamaMalformedResponseError: If response JSON cannot be decoded or is invalid.
        """
        target_model = model or self.primary_model
        eff_temperature = temperature if temperature is not None else self.temperature
        eff_num_ctx = num_ctx or self.num_ctx

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": eff_num_ctx,
                "temperature": eff_temperature,
                "seed": 42,
            },
            "keep_alive": self.keep_alive,
        }
        if system_prompt:
            payload["system"] = system_prompt

        # Acquire concurrency semaphore to enforce single active generation
        async with self._semaphore:
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resp = await client.post(url, json=payload)

                    if resp.status_code == 404:
                        logger.error("Requested model '%s' not found on Ollama server.", target_model)
                        raise ModelUnavailableError(
                            f"Model '{target_model}' is not available.",
                            detail=f"The requested local model '{target_model}' was not found in the Ollama registry.",
                        )
                    elif resp.status_code != 200:
                        logger.error("Ollama generate failed with status %d: %s", resp.status_code, resp.text)
                        raise OllamaUnavailableError(
                            f"Ollama generation failed with status code {resp.status_code}.",
                            detail="Local LLM runtime reported an internal execution error.",
                        )

                    try:
                        data = resp.json()
                    except Exception as exc:
                        logger.error("Malformed JSON response from Ollama: %s", resp.text)
                        raise OllamaMalformedResponseError(
                            "Invalid response received from local LLM runtime.",
                            detail="Ollama returned a malformed response body.",
                        ) from exc

                    text = data.get("response", "").strip()
                    metrics = {
                        "model": target_model,
                        "total_duration_ms": round(data.get("total_duration", 0) / 1e6, 2),
                        "load_duration_ms": round(data.get("load_duration", 0) / 1e6, 2),
                        "prompt_eval_count": data.get("prompt_eval_count", 0),
                        "prompt_eval_duration_ms": round(data.get("prompt_eval_duration", 0) / 1e6, 2),
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration_ms": round(data.get("eval_duration", 0) / 1e6, 2),
                    }
                    return text, metrics

            except httpx.ConnectError as exc:
                logger.error("Connection error to Ollama at %s: %s", url, exc)
                raise OllamaUnavailableError(
                    "Cannot connect to local Ollama service.",
                    detail="Local LLM service is offline or unreachable at 127.0.0.1:11434.",
                ) from exc
            except httpx.TimeoutException as exc:
                logger.error("Ollama generation timed out after %.1f seconds.", self.timeout_seconds)
                raise OllamaTimeoutError(
                    f"LLM generation timed out after {self.timeout_seconds} seconds.",
                    detail="Local LLM inference took longer than the configured timeout limit.",
                ) from exc
            except (OllamaUnavailableError, ModelUnavailableError, OllamaTimeoutError, OllamaMalformedResponseError):
                raise
            except Exception as exc:
                logger.exception("Unexpected error in OllamaLLMService.generate: %s", exc)
                raise OllamaUnavailableError(
                    "Unexpected LLM generation failure.",
                    detail="An unexpected error occurred during local LLM generation.",
                ) from exc


# ---------------------------------------------------------------------------
# Google Gemini Implementation (Phase 11 - Free Tier)
# ---------------------------------------------------------------------------

class GeminiLLMService(BaseLLMService):
    """Asynchronous client for Google Gemini REST API (Free Tier)."""

    MAX_503_ATTEMPTS: int = 3
    BACKOFF_503_SECONDS: tuple[float, ...] = (2.0, 4.0)

    _UNSET_KEY = object()
    _instance: Optional["GeminiLLMService"] = None

    def __init__(
        self,
        api_key: Any = _UNSET_KEY,
        primary_model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_concurrency: Optional[int] = None,
        fallback_service: Optional[BaseLLMService] = None,
    ):
        if api_key is self._UNSET_KEY:
            self.api_key = settings.gemini_api_key
        else:
            self.api_key = api_key
        self._primary_model = primary_model or settings.gemini_model
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.gemini_timeout_seconds
        self.max_concurrency = max_concurrency or settings.gemini_max_concurrency
        self.fallback_service = fallback_service

        # Concurrency limiter to strictly respect 15 RPM free-tier bounds
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    @property
    def primary_model(self) -> str:
        return self._primary_model

    @classmethod
    def get_instance(cls, fallback_service: Optional[BaseLLMService] = None) -> "GeminiLLMService":
        """Returns the singleton Gemini LLM service instance."""
        if cls._instance is None:
            cls._instance = cls(fallback_service=fallback_service)
        elif fallback_service is not None and cls._instance.fallback_service is None:
            cls._instance.fallback_service = fallback_service
        return cls._instance

    async def check_health(self) -> dict[str, Any]:
        """Checks connectivity and credential status with Google Gemini API."""
        if not self.api_key:
            return {
                "status": "degraded",
                "provider": "gemini",
                "primary_model": self.primary_model,
                "detail": "GEMINI_API_KEY is not configured.",
            }

        url = f"{self.base_url}/models/{self.primary_model}"
        headers = {"x-goog-api-key": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    return {
                        "status": "healthy",
                        "provider": "gemini",
                        "primary_model": self.primary_model,
                        "primary_model_ready": True,
                    }
                elif resp.status_code in (400, 401, 403):
                    return {
                        "status": "degraded",
                        "provider": "gemini",
                        "primary_model": self.primary_model,
                        "detail": f"Gemini API authentication failed (status {resp.status_code}).",
                    }
                return {
                    "status": "degraded",
                    "provider": "gemini",
                    "primary_model": self.primary_model,
                    "detail": f"Gemini API returned status {resp.status_code}.",
                }
        except Exception as exc:
            logger.warning("Gemini health check encountered error: %s", exc)
            return {
                "status": "degraded",
                "provider": "gemini",
                "primary_model": self.primary_model,
                "detail": "Cannot connect to Google Gemini API endpoint.",
            }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Executes non-streaming completion with Google Gemini generateContent REST API.

        Strictly enforces:
        - Stateless API invocation (no persistent remote sessions).
        - No Google Search grounding or paid services (tools omitted).
        - Deterministic zero temperature for grounded academic generation.
        - Programmatic fallback to local Ollama if configured and available.
        """
        target_model = model or self._primary_model

        # Helper to execute local Ollama fallback safely
        async def _execute_fallback(reason: str, exc: Exception) -> tuple[str, dict[str, Any]]:
            if self.fallback_service is not None:
                logger.warning(
                    "Gemini API generation failed (%s). Executing automatic fallback to local %s.",
                    reason,
                    self.fallback_service.primary_model,
                )
                text, metrics = await self.fallback_service.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    num_ctx=num_ctx,
                )
                metrics["fallback_from"] = "gemini"
                metrics["fallback_reason"] = reason
                return text, metrics
            raise exc

        if not self.api_key:
            err = GeminiAuthError(
                "GEMINI_API_KEY is not configured.",
                detail="Google Gemini API key is missing from backend environment configuration.",
            )
            return await _execute_fallback("missing_api_key", err)

        url = f"{self.base_url}/models/{target_model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        # Stateless request payload without search grounding or external tools
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 2048,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        async with self._semaphore:
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    for attempt in range(self.MAX_503_ATTEMPTS):
                        resp = await client.post(url, headers=headers, json=payload)

                        if resp.status_code == 200:
                            try:
                                data = resp.json()
                            except Exception as exc:
                                logger.error("Malformed JSON response from Gemini API: %s", resp.text)
                                err = GeminiMalformedResponseError(
                                    "Invalid JSON response from Gemini API.",
                                    detail="Gemini API returned an unparseable response body.",
                                )
                                return await _execute_fallback("malformed_json", err)

                            candidates = data.get("candidates", [])
                            if not candidates:
                                logger.error("Gemini API returned zero candidates: %s", resp.text)
                                err = GeminiMalformedResponseError(
                                    "No candidates returned from Gemini API.",
                                    detail="Gemini returned an empty candidate list.",
                                )
                                return await _execute_fallback("zero_candidates", err)

                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(p.get("text", "") for p in parts).strip()
                            usage = data.get("usageMetadata", {})
                            metrics = {
                                "provider": "gemini",
                                "model": target_model,
                                "prompt_token_count": usage.get("promptTokenCount", 0),
                                "candidates_token_count": usage.get("candidatesTokenCount", 0),
                                "total_token_count": usage.get("totalTokenCount", 0),
                            }
                            if attempt > 0:
                                metrics["retries"] = attempt
                            return text, metrics

                        elif resp.status_code == 503:
                            if attempt < self.MAX_503_ATTEMPTS - 1:
                                wait_seconds = self.BACKOFF_503_SECONDS[attempt]
                                logger.warning(
                                    "Gemini API returned 503 (high demand/service unavailable) on attempt %d/%d. Retrying in %.1fs...",
                                    attempt + 1,
                                    self.MAX_503_ATTEMPTS,
                                    wait_seconds,
                                )
                                await asyncio.sleep(wait_seconds)
                                continue

                            logger.error(
                                "Gemini API server error (503) after %d attempts: %s",
                                self.MAX_503_ATTEMPTS,
                                resp.text,
                            )
                            err = GeminiAPIError(
                                f"Gemini API server error (503) after {self.MAX_503_ATTEMPTS} attempts.",
                                detail="Google Gemini service is temporarily overloaded or unavailable.",
                            )
                            return await _execute_fallback("server_error_503", err)

                        elif resp.status_code == 429:
                            logger.warning("Gemini API rate limit or quota exceeded (429) for model %s.", target_model)
                            err = GeminiRateLimitError(
                                "Gemini API rate limit or quota exceeded (429).",
                                detail="Gemini free tier rate limit was reached. Please wait before retrying.",
                            )
                            return await _execute_fallback("rate_limit_429", err)

                        elif resp.status_code in (400, 401, 403):
                            logger.error("Gemini API client or authentication error (%d).", resp.status_code)
                            err = GeminiAuthError(
                                f"Gemini API authentication failed (status {resp.status_code}).",
                                detail=f"Gemini API returned client error {resp.status_code}. Verify GEMINI_API_KEY.",
                            )
                            return await _execute_fallback(f"auth_error_{resp.status_code}", err)

                        else:
                            logger.error("Gemini API server error (%d): %s", resp.status_code, resp.text)
                            err = GeminiAPIError(
                                f"Gemini API server error ({resp.status_code}).",
                                detail="Google Gemini service encountered an unexpected error.",
                            )
                            return await _execute_fallback(f"server_error_{resp.status_code}", err)

            except httpx.ConnectError as exc:
                logger.error("Connection error to Google Gemini API: %s", exc)
                err = GeminiUnavailableError(
                    "Cannot connect to Google Gemini API.",
                    detail="Network connection error to Google Gemini API endpoint.",
                )
                return await _execute_fallback("connect_error", err)

            except httpx.TimeoutException as exc:
                logger.error("Gemini API request timed out after %.1f seconds.", self.timeout_seconds)
                err = GeminiTimeoutError(
                    f"Gemini API generation timed out after {self.timeout_seconds} seconds.",
                    detail="Google Gemini API inference took longer than the configured timeout limit.",
                )
                return await _execute_fallback("timeout", err)

            except (GeminiRateLimitError, GeminiAuthError, GeminiTimeoutError, GeminiAPIError, GeminiUnavailableError, GeminiMalformedResponseError):
                raise

            except Exception as exc:
                logger.exception("Unexpected error in GeminiLLMService.generate: %s", exc)
                err = GeminiAPIError(
                    "Unexpected Gemini generation failure.",
                    detail="An unexpected error occurred during Google Gemini API generation.",
                )
                return await _execute_fallback("unexpected_error", err)


def get_llm_service() -> BaseLLMService:
    """Dependency provider for FastAPI or service layer.

    Inspects `settings.llm_provider` to supply either OllamaLLMService or GeminiLLMService.
    Defaults to local Ollama runtime for 100% backward compatibility and offline operation.
    """
    provider = (settings.llm_provider or "ollama").strip().lower()
    if provider == "gemini":
        fallback = OllamaLLMService.get_instance() if settings.gemini_fallback_to_local else None
        return GeminiLLMService.get_instance(fallback_service=fallback)
    return OllamaLLMService.get_instance()


def reset_llm_service_instances() -> None:
    """Resets singleton instances for isolated test environments."""
    OllamaLLMService._instance = None
    GeminiLLMService._instance = None
