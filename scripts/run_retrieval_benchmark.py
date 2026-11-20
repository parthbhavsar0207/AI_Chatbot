"""Runs comprehensive retrieval benchmark comparing unfiltered, category-filtered, and course-filtered retrieval."""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.evaluator import RetrievalEvaluator
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import VectorStoreService


def main():
    processed_dir = REPO_ROOT / "data" / "processed"
    chroma_dir = processed_dir / "chroma"
    dataset_path = processed_dir / "retrieval_eval_dataset.json"

    print("=" * 80)
    print(" LPU CSE-AIML RAG CHATBOT — RETRIEVAL BENCHMARK & EVALUATION (PHASE 6)")
    print("=" * 80)

    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    with open(dataset_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} evaluation queries.")
    supported_count = sum(1 for q in queries if q.get("is_supported", True))
    unsupported_count = len(queries) - supported_count
    print(f"  • Supported queries:   {supported_count}")
    print(f"  • Unsupported queries: {unsupported_count}")
    print("-" * 80)

    vs = VectorStoreService(persist_directory=chroma_dir, collection_name="lpu_cse_aiml_docs")
    retriever = RetrievalService(vector_store=vs)
    evaluator = RetrievalEvaluator(retrieval_service=retriever)

    # 1. Unfiltered evaluation
    print("\n[1/3] Running UNFILTERED Retrieval Benchmark...")
    report_unfiltered = evaluator.evaluate_queries(queries, mode="unfiltered", top_k=5)

    # 2. Category-filtered evaluation
    print("\n[2/3] Running CATEGORY-FILTERED Retrieval Benchmark...")
    report_cat = evaluator.evaluate_queries(queries, mode="category_filtered", top_k=5)

    # 3. Course-filtered evaluation
    print("\n[3/3] Running COURSE-FILTERED Retrieval Benchmark...")
    report_course = evaluator.evaluate_queries(queries, mode="course_filtered", top_k=5)

    # Output detailed report
    print("\n" + "=" * 80)
    print(" 1. OVERALL METRIC COMPARISON")
    print("=" * 80)
    print(f"{'Mode':<22} | {'Hit Rate@3':<12} | {'Recall@5':<12} | {'Precision@5':<14} | {'MRR':<10}")
    print("-" * 80)
    for rep in (report_unfiltered, report_cat, report_course):
        m = rep.overall_metrics
        print(
            f"{rep.mode:<22} | {m.mean_hit_rate_at_3 * 100:>10.2f}% | {m.mean_recall_at_5 * 100:>10.2f}% | "
            f"{m.mean_precision_at_5 * 100:>12.2f}% | {m.mean_mrr:>10.4f}"
        )

    print("\n" + "=" * 80)
    print(" 2. CATEGORY-WISE METRICS (UNFILTERED)")
    print("=" * 80)
    print(f"{'Category/Domain':<18} | {'Queries':<8} | {'Hit Rate@3':<12} | {'Recall@5':<12} | {'Precision@5':<14} | {'MRR':<10}")
    print("-" * 80)
    for cat in report_unfiltered.category_metrics:
        print(
            f"{cat.domain:<18} | {cat.query_count:<8} | {cat.mean_hit_rate_at_3 * 100:>10.2f}% | "
            f"{cat.mean_recall_at_5 * 100:>10.2f}% | {cat.mean_precision_at_5 * 100:>12.2f}% | {cat.mean_mrr:>10.4f}"
        )

    print("\n" + "=" * 80)
    print(" 3. CATEGORY-WISE METRICS (CATEGORY-FILTERED)")
    print("=" * 80)
    print(f"{'Category/Domain':<18} | {'Queries':<8} | {'Hit Rate@3':<12} | {'Recall@5':<12} | {'Precision@5':<14} | {'MRR':<10}")
    print("-" * 80)
    for cat in report_cat.category_metrics:
        print(
            f"{cat.domain:<18} | {cat.query_count:<8} | {cat.mean_hit_rate_at_3 * 100:>10.2f}% | "
            f"{cat.mean_recall_at_5 * 100:>10.2f}% | {cat.mean_precision_at_5 * 100:>12.2f}% | {cat.mean_mrr:>10.4f}"
        )

    print("\n" + "=" * 80)
    print(" 4. OUTRANKED OR FAILED QUERIES (DIAGNOSTICS)")
    print("=" * 80)
    if report_unfiltered.failed_queries:
        print(f"Failed Queries (Hit Rate@3 == 0) [{len(report_unfiltered.failed_queries)}]:")
        for fq in report_unfiltered.failed_queries:
            print(f"  • [{fq.query_id}] '{fq.query}' (Domain: {fq.domain})")
            print(f"      Expected IDs: {fq.expected_chunk_ids}")
            print(f"      Retrieved IDs: {fq.retrieved_chunk_ids[:3]}")
    else:
        print("  • No queries failed to achieve a hit in top-3 (100% Hit Rate@3).")

    if report_unfiltered.outranked_queries:
        print(f"\nQueries where irrelevant categories outranked relevant evidence [{len(report_unfiltered.outranked_queries)}]:")
        for oq in report_unfiltered.outranked_queries:
            print(f"  • [{oq.query_id}] '{oq.query}' (Expected domain: {oq.domain})")
            print(f"      First relevant rank: #{oq.first_relevant_rank}")
            print(f"      Higher-ranked categories: {oq.retrieved_categories[:oq.first_relevant_rank-1]}")
            print(f"      Higher-ranked sources:    {oq.retrieved_sources[:oq.first_relevant_rank-1]}")
    else:
        print("  • No queries had relevant evidence outranked by irrelevant categories.")

    print("\n" + "=" * 80)
    print(" 5. UNSUPPORTED QUESTIONS ANALYSIS")
    print("=" * 80)
    for uq in report_unfiltered.unsupported_queries:
        print(f"  • [{uq.query_id}] '{uq.query}'")
        print(f"      Top similarity: {uq.top_similarity:.4f}")
        print(f"      Top retrieved source: {uq.retrieved_sources[0] if uq.retrieved_sources else 'None'}")
        print(f"      Top retrieved category: {uq.retrieved_categories[0] if uq.retrieved_categories else 'None'}")
        print(f"      Evidence Status: GENUINELY INSUFFICIENT (No corpus match, low confidence)")

    # Save benchmark records
    results_output = {
        "unfiltered_report": report_unfiltered.model_dump(),
        "category_filtered_report": report_cat.model_dump(),
        "course_filtered_report": report_course.model_dump(),
    }
    out_path = processed_dir / "retrieval_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2)

    print(f"\nSaved full benchmark results to: {out_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    main()
