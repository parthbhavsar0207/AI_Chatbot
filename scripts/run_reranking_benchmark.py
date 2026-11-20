"""Benchmark comparing Baseline Vector Retrieval vs. Cross-Encoder Reranked Retrieval.

Evaluates on the 26-query dataset across Hit Rate@3, Recall@5, Precision@5, and MRR.
Reports overall, category-wise, improved, degraded, and unchanged queries.
"""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.evaluator import RetrievalEvaluator
from app.rag.reranker import CrossEncoderReranker
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import VectorStoreService


def run_benchmark():
    processed_dir = REPO_ROOT / "data" / "processed"
    chroma_dir = processed_dir / "chroma"
    dataset_path = processed_dir / "retrieval_eval_dataset.json"

    print("=" * 80)
    print(" LPU CSE-AIML RAG CHATBOT — CROSS-ENCODER RERANKING BENCHMARK (PHASE 8)")
    print(" Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (ONNX Runtime CPU)")
    print("=" * 80)

    if not dataset_path.exists():
        print(f"Error: Dataset not found at: {dataset_path}", file=sys.stderr)
        return 1

    with open(dataset_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} evaluation queries.")
    supported_queries = [q for q in queries if q.get("is_supported", True)]
    unsupported_queries = [q for q in queries if not q.get("is_supported", True)]
    print(f"  • Supported queries:   {len(supported_queries)}")
    print(f"  • Unsupported queries: {len(unsupported_queries)}")
    print("-" * 80)

    vs = VectorStoreService(persist_directory=chroma_dir, collection_name="lpu_cse_aiml_docs")
    reranker = CrossEncoderReranker()
    retriever = RetrievalService(vector_store=vs, reranker=reranker)
    evaluator = RetrievalEvaluator(retrieval_service=retriever)

    # 1. Baseline Unfiltered Vector Retrieval
    print("\n[1/4] Running BASELINE Vector Retrieval (Unfiltered, top_k=5)...")
    baseline_unfiltered = evaluator.evaluate_queries(queries, mode="unfiltered", top_k=5, enable_rerank=False)

    # 2. Reranked Unfiltered Retrieval
    print("[2/4] Running RERANKED Retrieval (Unfiltered, candidate_k=10, top_k=5)...")
    reranked_unfiltered = evaluator.evaluate_queries(queries, mode="unfiltered", top_k=5, enable_rerank=True, candidate_k=10)

    # 3. Baseline Category-Filtered Retrieval
    print("[3/4] Running BASELINE Category-Filtered Retrieval (top_k=5)...")
    baseline_cat = evaluator.evaluate_queries(queries, mode="category_filtered", top_k=5, enable_rerank=False)

    # 4. Reranked Category-Filtered Retrieval
    print("[4/4] Running RERANKED Category-Filtered Retrieval (candidate_k=10, top_k=5)...")
    reranked_cat = evaluator.evaluate_queries(queries, mode="category_filtered", top_k=5, enable_rerank=True, candidate_k=10)

    # =========================================================================
    # Comparison Tables
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 1. OVERALL METRIC COMPARISON (BASELINE vs. RERANKED)")
    print("=" * 80)
    print(f"{'Pipeline Mode':<30} | {'Hit Rate@3':<12} | {'Recall@5':<12} | {'Precision@5':<14} | {'MRR':<10}")
    print("-" * 80)

    modes = [
        ("Baseline (Unfiltered)", baseline_unfiltered.overall_metrics),
        ("Reranked (Unfiltered)", reranked_unfiltered.overall_metrics),
        ("Baseline (Category-Filtered)", baseline_cat.overall_metrics),
        ("Reranked (Category-Filtered)", reranked_cat.overall_metrics),
    ]

    for label, m in modes:
        print(
            f"{label:<30} | {m.mean_hit_rate_at_3 * 100:>10.2f}% | {m.mean_recall_at_5 * 100:>10.2f}% | "
            f"{m.mean_precision_at_5 * 100:>12.2f}% | {m.mean_mrr:>10.4f}"
        )

    # Category Breakdown for Unfiltered
    print("\n" + "=" * 80)
    print(" 2. CATEGORY-WISE METRIC COMPARISON (UNFILTERED)")
    print("=" * 80)
    print(f"{'Domain':<14} | {'Pipeline':<12} | {'Queries':<8} | {'Hit Rate@3':<12} | {'Recall@5':<12} | {'MRR':<10}")
    print("-" * 80)

    base_cats = {c.domain: c for c in baseline_unfiltered.category_metrics}
    rerank_cats = {c.domain: c for c in reranked_unfiltered.category_metrics}

    for dom in sorted(base_cats.keys()):
        b = base_cats[dom]
        r = rerank_cats[dom]
        print(
            f"{dom:<14} | {'Baseline':<12} | {b.query_count:<8} | {b.mean_hit_rate_at_3 * 100:>10.2f}% | "
            f"{b.mean_recall_at_5 * 100:>10.2f}% | {b.mean_mrr:>10.4f}"
        )
        print(
            f"{'':<14} | {'Reranked':<12} | {r.query_count:<8} | {r.mean_hit_rate_at_3 * 100:>10.2f}% | "
            f"{r.mean_recall_at_5 * 100:>10.2f}% | {r.mean_mrr:>10.4f}"
        )
        print("-" * 80)

    # Detailed Query-by-Query Analysis (Unfiltered)
    base_results = {r.query_id: r for r in baseline_unfiltered.all_query_results if r.is_supported}
    rerank_results = {r.query_id: r for r in reranked_unfiltered.all_query_results if r.is_supported}

    improved = []
    degraded = []
    unchanged = []

    for qid in sorted(base_results.keys()):
        b_res = base_results[qid]
        r_res = rerank_results[qid]

        b_rank = b_res.first_relevant_rank or 999
        r_rank = r_res.first_relevant_rank or 999

        diff_mrr = r_res.mrr - b_res.mrr

        item = {
            "query_id": qid,
            "query": b_res.query,
            "domain": b_res.domain,
            "baseline_rank": b_res.first_relevant_rank,
            "reranked_rank": r_res.first_relevant_rank,
            "baseline_mrr": b_res.mrr,
            "reranked_mrr": r_res.mrr,
            "baseline_h3": b_res.hit_at_3,
            "reranked_h3": r_res.hit_at_3,
        }

        if diff_mrr > 0.0 or (r_rank < b_rank):
            improved.append(item)
        elif diff_mrr < 0.0 or (r_rank > b_rank):
            degraded.append(item)
        else:
            unchanged.append(item)

    print("\n" + "=" * 80)
    print(f" 3. QUERY-LEVEL RANKING DIAGNOSTICS ({len(improved)} Improved, {len(degraded)} Degraded, {len(unchanged)} Unchanged)")
    print("=" * 80)

    print(f"\n[A] IMPROVED QUERIES ({len(improved)}):")
    if improved:
        for it in improved:
            print(f"  • [{it['query_id']}] '{it['query']}' (Domain: {it['domain']})")
            print(f"      Rank: {it['baseline_rank']} -> {it['reranked_rank']} | MRR: {it['baseline_mrr']:.4f} -> {it['reranked_mrr']:.4f}")
    else:
        print("  None.")

    print(f"\n[B] DEGRADED QUERIES ({len(degraded)}):")
    if degraded:
        for it in degraded:
            print(f"  • [{it['query_id']}] '{it['query']}' (Domain: {it['domain']})")
            print(f"      Rank: {it['baseline_rank']} -> {it['reranked_rank']} | MRR: {it['baseline_mrr']:.4f} -> {it['reranked_mrr']:.4f}")
    else:
        print("  None. No queries degraded.")

    print(f"\n[C] UNCHANGED QUERIES ({len(unchanged)}):")
    for it in unchanged:
        print(f"  • [{it['query_id']}] Rank: {it['baseline_rank']} | MRR: {it['baseline_mrr']:.4f} | '{it['query']}'")

    # Unsupported queries analysis
    print("\n" + "=" * 80)
    print(" 4. UNSUPPORTED QUESTIONS ANALYSIS (CROSS-ENCODER CONFIDENCE)")
    print("=" * 80)
    for q_item in unsupported_queries:
        qid = q_item["query_id"]
        q_text = q_item["query"]
        retrieved = retriever.retrieve(q_text, top_k=3, enable_rerank=True, candidate_k=10)
        top = retrieved[0] if retrieved else None
        print(f"  • [{qid}] '{q_text}'")
        if top:
            print(f"      Top Vector Similarity: {top.similarity:.4f}")
            print(f"      Top Cross-Encoder Score: {top.reranker_score:.4f}")
            print(f"      Top Source: {top.metadata.get('source_file')}")
            print(f"      Top Category: {top.metadata.get('category')}")
        print()

    # Save benchmark artifacts
    benchmark_data = {
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "unfiltered": {
            "baseline": baseline_unfiltered.model_dump(),
            "reranked": reranked_unfiltered.model_dump(),
        },
        "category_filtered": {
            "baseline": baseline_cat.model_dump(),
            "reranked": reranked_cat.model_dump(),
        },
        "diagnostics": {
            "improved": improved,
            "degraded": degraded,
            "unchanged_count": len(unchanged),
        },
    }

    out_file = processed_dir / "reranking_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Saved benchmark results to: {out_file}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
