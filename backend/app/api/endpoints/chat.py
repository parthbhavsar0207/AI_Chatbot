"""FastAPI router for grounded RAG chat answer generation (Phase 10.3)."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.chat import ChatAnswerRequest, ChatAnswerResponse
from app.schemas.retrieval import ErrorResponse
from app.services.llm_service import (
    GeminiAPIError,
    GeminiAuthError,
    GeminiMalformedResponseError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    LLMServiceError,
    ModelUnavailableError,
    OllamaMalformedResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.services.rag_service import RAGChatService, get_rag_chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post(
    "/answer",
    response_model=ChatAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Grounded Academic Q&A with Citation Validation",
    description=(
        "Executes a complete decision-conditioned RAG generation pipeline: "
        "retrieves chunks via FastEmbed, reranks via cross-encoder, evaluates evidence sufficiency, "
        "and generates a cited answer using the local Qwen3 model. "
        "All citations are programmatically validated against retrieved evidence chunks."
    ),
    responses={
        200: {
            "description": "Successful grounded response with verified citations or deterministic abstention.",
            "model": ChatAnswerResponse,
        },
        422: {
            "description": "Validation error (e.g. empty query or invalid parameter).",
            "model": ErrorResponse,
        },
        503: {
            "description": "Local LLM service unavailable or unreachable.",
            "model": ErrorResponse,
        },
        504: {
            "description": "Local LLM generation request timed out.",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error during RAG orchestration.",
            "model": ErrorResponse,
        },
    },
)
async def generate_chat_answer(
    request: ChatAnswerRequest,
    service: RAGChatService = Depends(get_rag_chat_service),
) -> ChatAnswerResponse:
    """Answers a student question grounded strictly in official university documents."""
    try:
        response = await service.answer_question(request)
        return response
    except ValueError as ve:
        logger.warning("Chat request validation failed: %s", ve)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except (OllamaUnavailableError, ModelUnavailableError, GeminiUnavailableError, GeminiAuthError, GeminiRateLimitError) as oue:
        logger.error("LLM service unavailable / rate limit: %s", oue)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=oue.detail,
        )
    except (OllamaTimeoutError, GeminiTimeoutError) as ote:
        logger.error("LLM request timed out: %s", ote)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=ote.detail,
        )
    except (OllamaMalformedResponseError, GeminiMalformedResponseError, GeminiAPIError) as mre:
        logger.error("LLM response / API error: %s", mre)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=mre.detail,
        )
    except LLMServiceError as lse:
        logger.error("General LLM service error: %s", lse)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=lse.detail,
        )
    except Exception as exc:
        logger.exception("Unexpected error during chat answer generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while generating the answer.",
        )
