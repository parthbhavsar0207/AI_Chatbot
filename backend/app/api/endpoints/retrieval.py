"""FastAPI endpoint router for semantic retrieval and search operations."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.rag.retrieval import RetrievalService
from app.schemas.retrieval import ErrorResponse, SearchRequest, SearchResponse
from app.services.retrieval_service import execute_search, get_retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/retrieval", tags=["Retrieval"])


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Retrieval Search",
    description=(
        "Executes semantic similarity search over the LPU CSE-AIML academic document collection "
        "using BAAI/bge-small-en-v1.5 embeddings and ChromaDB persistent vector storage. "
        "Supports optional metadata filtering by document category, course code, and source file, "
        "as well as cross-encoder reranking via ms-marco-MiniLM-L-6-v2."
    ),
    responses={
        200: {
            "description": "Successful retrieval response with ranked chunks.",
            "model": SearchResponse,
        },
        422: {
            "description": "Validation error (e.g., empty query, invalid top_k, unsupported category, invalid course code).",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal retrieval service failure.",
            "model": ErrorResponse,
        },
    },
)
async def search_documents(
    request: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """Executes semantic search against the indexed academic knowledge base."""
    try:
        results, elapsed_ms, decision = execute_search(request, service)
    except ValueError as ve:
        logger.warning("Retrieval request validation failed: %s", ve)
        raise HTTPException(
            status_code=422,
            detail=str(ve),
        )
    except Exception as exc:
        logger.exception("Unexpected error during retrieval search: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the retrieval request.",
        )

    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        total_results=len(results),
        retrieval_time_ms=elapsed_ms,
        rerank_enabled=request.enable_rerank,
        candidate_k=request.candidate_k if request.enable_rerank else None,
        evidence_decision=decision,
        results=results,
    )
