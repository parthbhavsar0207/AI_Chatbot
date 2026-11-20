"""Retrieval evaluation and benchmark service for RAG corpus.

Calculates Hit Rate@3, Recall@5, Precision@5, and Mean Reciprocal Rank (MRR),
supporting overall, category-wise, and filtered vs. unfiltered comparisons.
"""

import logging
from typing import Any, Optional
from pydantic import BaseModel, Field

from app.rag.retrieval import RetrievalService

logger = logging.getLogger(__name__)


# ==============================================================================
# Pure Evaluation Metric Functions
# ==============================================================================


def compute_hit_rate(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str] | set[str],
    k: int = 3,
) -> float:
    """Returns 1.0 if at least one expected chunk appears in the top-k results, else 0.0."""
    if not expected_chunk_ids:
        return 0.0
    expected_set = set(expected_chunk_ids)
    top_k_ids = retrieved_chunk_ids[:k]
    return 1.0 if any(cid in expected_set for cid in top_k_ids) else 0.0


def compute_recall(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str] | set[str],
    k: int = 5,
) -> float:
    """Calculates the proportion of expected chunks that appear in the top-k retrieved results."""
    if not expected_chunk_ids:
        return 0.0
    expected_set = set(expected_chunk_ids)
    top_k_set = set(retrieved_chunk_ids[:k])
    hits = len(top_k_set & expected_set)
    return round(hits / len(expected_set), 6)


def compute_precision(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str] | set[str],
    k: int = 5,
) -> float:
    """Calculates the proportion of top-k retrieved chunks that are relevant."""
    top_k_ids = retrieved_chunk_ids[:k]
    if not top_k_ids:
        return 0.0
    if not expected_chunk_ids:
        return 0.0
    expected_set = set(expected_chunk_ids)
    hits = sum(1 for cid in top_k_ids if cid in expected_set)
    return round(hits / len(top_k_ids), 6)


def compute_reciprocal_rank(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str] | set[str],
    max_k: int = 10,
) -> float:
    """Calculates reciprocal rank (1/rank) of the first relevant chunk within top max_k results."""
    if not expected_chunk_ids:
        return 0.0
    expected_set = set(expected_chunk_ids)
    for rank_idx, cid in enumerate(retrieved_chunk_ids[:max_k], 1):
        if cid in expected_set:
            return round(1.0 / rank_idx, 6)
    return 0.0


# ==============================================================================
# Evaluation Schemas
# ==============================================================================


class QueryEvaluationResult(BaseModel):
    """Detailed evaluation result for a single query."""

    query_id: str
    query: str
    domain: str
    course_code: Optional[str] = None
    is_supported: bool
    expected_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_categories: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    retrieved_similarities: list[float] = Field(default_factory=list)
    hit_at_3: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    first_relevant_rank: Optional[int] = None
    outranked_by_irrelevant_category: bool = False
    top_similarity: float = 0.0


class MetricSummary(BaseModel):
    """Aggregated retrieval benchmark metrics."""

    total_queries: int
    supported_queries: int
    unsupported_queries: int
    mean_hit_rate_at_3: float
    mean_recall_at_5: float
    mean_precision_at_5: float
    mean_mrr: float


class CategoryBreakdown(BaseModel):
    """Metrics aggregated for a specific domain/category."""

    domain: str
    query_count: int
    mean_hit_rate_at_3: float
    mean_recall_at_5: float
    mean_precision_at_5: float
    mean_mrr: float


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report containing overall, category, and diagnostic records."""

    mode: str
    rerank_enabled: bool = False
    overall_metrics: MetricSummary
    category_metrics: list[CategoryBreakdown]
    failed_queries: list[QueryEvaluationResult] = Field(default_factory=list)
    outranked_queries: list[QueryEvaluationResult] = Field(default_factory=list)
    unsupported_queries: list[QueryEvaluationResult] = Field(default_factory=list)
    all_query_results: list[QueryEvaluationResult] = Field(default_factory=list)


# ==============================================================================
# RetrievalEvaluator Class
# ==============================================================================


class RetrievalEvaluator:
    """Evaluates RetrievalService against ground-truth evaluation datasets."""

    def __init__(self, retrieval_service: RetrievalService):
        self.retriever = retrieval_service

    def evaluate_queries(
        self,
        queries: list[dict[str, Any]],
        mode: str = "unfiltered",
        top_k: int = 5,
        enable_rerank: bool = False,
        candidate_k: Optional[int] = None,
    ) -> EvaluationReport:
        """Runs evaluation over a list of query dicts in the specified filter mode.

        Args:
            queries: List of evaluation query dicts matching retrieval_eval_dataset schema.
            mode: 'unfiltered', 'category_filtered', or 'course_filtered'.
            top_k: Number of chunks to retrieve per query (default: 5).
            enable_rerank: Whether to enable cross-encoder reranking.
            candidate_k: Initial candidate count retrieved before reranking.

        Returns:
            EvaluationReport containing detailed and aggregated metrics.
        """
        results: list[QueryEvaluationResult] = []

        for q_item in queries:
            qid = q_item["query_id"]
            q_text = q_item["query"]
            domain = q_item["domain"]
            course = q_item.get("course_code")
            is_supported = q_item.get("is_supported", True)
            expected_ids = q_item.get("expected_chunk_ids", [])

            # Apply filters according to mode
            category_filter = None
            course_filter = None
            if mode == "category_filtered" and is_supported:
                category_filter = domain
            elif mode == "course_filtered" and is_supported and course:
                course_filter = course

            retrieved = self.retriever.retrieve(
                query=q_text,
                top_k=top_k,
                category=category_filter,
                course_code=course_filter,
                enable_rerank=enable_rerank,
                candidate_k=candidate_k,
            )

            retrieved_ids = [r.chunk_id for r in retrieved]
            retrieved_cats = [r.metadata.get("category", "") for r in retrieved]
            retrieved_srcs = [r.metadata.get("source_file", "") for r in retrieved]
            retrieved_sims = [r.similarity for r in retrieved]
            top_sim = retrieved_sims[0] if retrieved_sims else 0.0

            if is_supported:
                h3 = compute_hit_rate(retrieved_ids, expected_ids, k=3)
                r5 = compute_recall(retrieved_ids, expected_ids, k=5)
                p5 = compute_precision(retrieved_ids, expected_ids, k=5)
                mrr_val = compute_reciprocal_rank(retrieved_ids, expected_ids, max_k=10)

                # Check first relevant rank
                expected_set = set(expected_ids)
                first_rank = None
                for idx, cid in enumerate(retrieved_ids, 1):
                    if cid in expected_set:
                        first_rank = idx
                        break

                # Check if outranked by an irrelevant category
                outranked = False
                if first_rank and first_rank > 1:
                    higher_rank_cats = [
                        c.lower() for c in retrieved_cats[: first_rank - 1]
                    ]
                    # If any higher rank chunk does not match query domain
                    if any(domain.lower() not in c for c in higher_rank_cats):
                        outranked = True
            else:
                # For unsupported questions, expected is empty
                h3 = 0.0
                r5 = 0.0
                p5 = 0.0
                mrr_val = 0.0
                first_rank = None
                outranked = False

            results.append(
                QueryEvaluationResult(
                    query_id=qid,
                    query=q_text,
                    domain=domain,
                    course_code=course,
                    is_supported=is_supported,
                    expected_chunk_ids=expected_ids,
                    retrieved_chunk_ids=retrieved_ids,
                    retrieved_categories=retrieved_cats,
                    retrieved_sources=retrieved_srcs,
                    retrieved_similarities=retrieved_sims,
                    hit_at_3=h3,
                    recall_at_5=r5,
                    precision_at_5=p5,
                    mrr=mrr_val,
                    first_relevant_rank=first_rank,
                    outranked_by_irrelevant_category=outranked,
                    top_similarity=top_sim,
                )
            )

        # Aggregate metrics for supported queries
        supported_results = [r for r in results if r.is_supported]
        unsupported_results = [r for r in results if not r.is_supported]

        n_supp = len(supported_results)
        mean_h3 = round(sum(r.hit_at_3 for r in supported_results) / n_supp, 4) if n_supp else 0.0
        mean_r5 = round(sum(r.recall_at_5 for r in supported_results) / n_supp, 4) if n_supp else 0.0
        mean_p5 = round(sum(r.precision_at_5 for r in supported_results) / n_supp, 4) if n_supp else 0.0
        mean_mrr = round(sum(r.mrr for r in supported_results) / n_supp, 4) if n_supp else 0.0

        overall = MetricSummary(
            total_queries=len(results),
            supported_queries=n_supp,
            unsupported_queries=len(unsupported_results),
            mean_hit_rate_at_3=mean_h3,
            mean_recall_at_5=mean_r5,
            mean_precision_at_5=mean_p5,
            mean_mrr=mean_mrr,
        )

        # Category breakdowns
        domains = sorted(set(r.domain for r in supported_results))
        cat_summaries: list[CategoryBreakdown] = []
        for d in domains:
            d_results = [r for r in supported_results if r.domain == d]
            d_n = len(d_results)
            cat_summaries.append(
                CategoryBreakdown(
                    domain=d,
                    query_count=d_n,
                    mean_hit_rate_at_3=round(sum(r.hit_at_3 for r in d_results) / d_n, 4),
                    mean_recall_at_5=round(sum(r.recall_at_5 for r in d_results) / d_n, 4),
                    mean_precision_at_5=round(sum(r.precision_at_5 for r in d_results) / d_n, 4),
                    mean_mrr=round(sum(r.mrr for r in d_results) / d_n, 4),
                )
            )

        failed = [r for r in supported_results if r.hit_at_3 == 0.0]
        outranked = [r for r in supported_results if r.outranked_by_irrelevant_category]

        return EvaluationReport(
            mode=mode,
            rerank_enabled=enable_rerank,
            overall_metrics=overall,
            category_metrics=cat_summaries,
            failed_queries=failed,
            outranked_queries=outranked,
            unsupported_queries=unsupported_results,
            all_query_results=results,
        )
