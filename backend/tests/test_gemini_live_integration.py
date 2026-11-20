"""Opt-in live integration tests for Google Gemini API (Phase 11).

These tests are skipped by default to guarantee ZERO live external API calls
during standard local regression testing and CI.
To execute them explicitly, set RUN_LIVE_GEMINI_TESTS=1 in the environment.
"""

import os
import pytest
from app.core.config import settings
from app.services.llm_service import (
    GeminiLLMService,
    GeminiTimeoutError,
    GeminiUnavailableError,
    reset_llm_service_instances,
)


pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_GEMINI_TESTS"),
    reason="Opt-in live Gemini integration test. Run with RUN_LIVE_GEMINI_TESTS=1.",
)


@pytest.mark.anyio
async def test_live_gemini_direct_generation():
    """Verify live Gemini API connectivity and generation when explicitly opted-in."""
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY is not set in environment or settings.")

    reset_llm_service_instances()
    service = GeminiLLMService()
    prompt = "Answer in one sentence: What is polymorphism in object oriented programming? [[cite:chunk_001]]"
    system_prompt = "You are a concise academic assistant. Always preserve citations."

    try:
        answer, metrics = await service.generate(prompt=prompt, system_prompt=system_prompt)
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert metrics["provider"] == "gemini"
        assert metrics["model"] == settings.gemini_model
        assert "total_token_count" in metrics
    except (GeminiTimeoutError, GeminiUnavailableError) as err:
        pytest.skip(f"Live Gemini API endpoint unreachable or timed out: {err}")
    finally:
        reset_llm_service_instances()
