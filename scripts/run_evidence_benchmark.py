"""Benchmark script evaluating the Evidence Sufficiency and Abstention Engine across 50 labeled queries.

Calculates:
- Decision Accuracy
- Precision and Recall for Insufficient-Evidence Detection
- False Acceptance Rate (FAR) and False Rejection Rate (FRR)
- 4x4 Confusion Matrix across (sufficient, partial, insufficient, conflicting)
- Per-State Classification Metrics
"""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.evidence_gate import EvidenceDecisionState, EvidenceGate
from app.rag.reranker import CrossEncoderReranker
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import VectorStoreService

STATES = [
    EvidenceDecisionState.SUFFICIENT.value,
    EvidenceDecisionState.PARTIAL.value,
    EvidenceDecisionState.INSUFFICIENT.value,
    EvidenceDecisionState.CONFLICTING.value,
]


def run_evidence_benchmark():
    processed_dir = REPO_ROOT / "data" / "processed"
    chroma_dir = processed_dir / "chroma"
    dataset_path = processed_dir / "evidence_eval_dataset.json"

    print("=" * 80)
    print(" LPU CSE-AIML RAG CHATBOT — EVIDENCE SUFFICIENCY & ABSTENTION BENCHMARK")
    print("=" * 80)

    if not dataset_path.exists():
        print(f"Error: Dataset not found at: {dataset_path}", file=sys.stderr)
        return 1

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} evaluation queries.")
    state_counts = {}
    for d in dataset:
        st = d["expected_decision"]
        state_counts[st] = state_counts.get(st, 0) + 1
    for st in STATES:
        print(f"  • {st:<12}: {state_counts.get(st, 0)}")
    print("-" * 80)

    vs = VectorStoreService(persist_directory=chroma_dir, collection_name="lpu_cse_aiml_docs")
    reranker = CrossEncoderReranker()
    gate = EvidenceGate()
    retriever = RetrievalService(vector_store=vs, reranker=reranker, evidence_gate=gate)

    results = []
    matrix = {true_st: {pred_st: 0 for pred_st in STATES} for true_st in STATES}

    for item in dataset:
        qid = item["query_id"]
        q_text = item["query"]
        expected_st = item["expected_decision"]

        # Run retrieval with cross-encoder rerank
        chunks = retriever.retrieve(q_text, top_k=5, enable_rerank=True, candidate_k=10)
        decision = gate.evaluate(query=q_text, retrieved_chunks=chunks)
        pred_st = decision.decision.value

        matrix[expected_st][pred_st] += 1

        results.append({
            "query_id": qid,
            "query": q_text,
            "domain": item.get("domain", "general"),
            "expected_decision": expected_st,
            "predicted_decision": pred_st,
            "is_correct": expected_st == pred_st,
            "reason_code": decision.reason_code,
            "confidence_score": decision.confidence_score,
            "missing_information": decision.missing_information,
            "conflict_details": decision.conflict_details,
            "top_similarity": chunks[0].similarity if chunks else 0.0,
            "top_reranker_score": chunks[0].reranker_score if chunks else -999.0,
            "supporting_chunk_ids": decision.supporting_chunk_ids,
        })

    # =========================================================================
    # Metrics Calculation
    # =========================================================================
    total_queries = len(results)
    correct_count = sum(1 for r in results if r["is_correct"])
    accuracy = correct_count / total_queries

    # Insufficient-evidence metrics (binary detection)
    # Target = 'insufficient'
    tp_ins = matrix["insufficient"]["insufficient"]
    fp_ins = sum(matrix[other]["insufficient"] for other in STATES if other != "insufficient")
    fn_ins = sum(matrix["insufficient"][other] for other in STATES if other != "insufficient")
    tn_ins = sum(matrix[t][p] for t in STATES if t != "insufficient" for p in STATES if p != "insufficient")

    prec_ins = tp_ins / (tp_ins + fp_ins) if (tp_ins + fp_ins) > 0 else 0.0
    rec_ins = tp_ins / (tp_ins + fn_ins) if (tp_ins + fn_ins) > 0 else 0.0
    f1_ins = (2 * prec_ins * rec_ins) / (prec_ins + rec_ins) if (prec_ins + rec_ins) > 0 else 0.0

    # False Acceptance Rate (FAR): truly insufficient classified as sufficient
    unsupported_total = sum(matrix["insufficient"].values())
    false_accepts = matrix["insufficient"]["sufficient"]
    far = false_accepts / unsupported_total if unsupported_total > 0 else 0.0

    # False Rejection Rate (FRR): truly sufficient classified as insufficient
    supported_total = sum(matrix["sufficient"].values())
    false_rejects = matrix["sufficient"]["insufficient"]
    frr = false_rejects / supported_total if supported_total > 0 else 0.0

    # =========================================================================
    # Terminal Display
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 1. OVERALL DECISION METRICS")
    print("=" * 80)
    print(f"Total Test Queries:                  {total_queries}")
    print(f"Correct Decisions:                   {correct_count} / {total_queries}")
    print(f"Overall Decision Accuracy:           {accuracy * 100:.2f}%")
    print(f"Insufficient-Evidence Precision:     {prec_ins * 100:.2f}%")
    print(f"Insufficient-Evidence Recall:        {rec_ins * 100:.2f}%")
    print(f"Insufficient-Evidence F1-Score:      {f1_ins:.4f}")
    print(f"False Acceptance Rate (FAR):         {far * 100:.2f}% ({false_accepts} / {unsupported_total})")
    print(f"False Rejection Rate (FRR):          {frr * 100:.2f}% ({false_rejects} / {supported_total})")

    print("\n" + "=" * 80)
    print(" 2. 4x4 CONFUSION MATRIX")
    print("=" * 80)
    header = f"{'True \\ Pred':<14} | " + " | ".join(f"{s:<12}" for s in STATES) + " | Total"
    print(header)
    print("-" * 80)
    for true_st in STATES:
        row_vals = [f"{matrix[true_st][p]:<12}" for p in STATES]
        row_tot = sum(matrix[true_st].values())
        print(f"{true_st:<14} | " + " | ".join(row_vals) + f" | {row_tot}")

    print("\n" + "=" * 80)
    print(" 3. PER-STATE PERFORMANCE BREAKDOWN")
    print("=" * 80)
    print(f"{'State':<14} | {'Support':<8} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<10}")
    print("-" * 80)
    for st in STATES:
        support = sum(matrix[st].values())
        tp = matrix[st][st]
        fp = sum(matrix[other][st] for other in STATES if other != st)
        fn = sum(matrix[st][other] for other in STATES if other != st)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        print(f"{st:<14} | {support:<8} | {p * 100:>10.2f}% | {r * 100:>10.2f}% | {f1:>10.4f}")

    # Inspect any mismatches
    mismatches = [r for r in results if not r["is_correct"]]
    print("\n" + "=" * 80)
    print(f" 4. MISCLASSIFIED QUERIES (N={len(mismatches)})")
    print("=" * 80)
    if mismatches:
        for m in mismatches:
            print(f"  • [{m['query_id']}] '{m['query']}'")
            print(f"      Expected: {m['expected_decision']} | Predicted: {m['predicted_decision']} (Reason: {m['reason_code']})")
            print(f"      Top Score: {m['top_reranker_score']:+.2f} | Sim: {m['top_similarity']:.4f}")
    else:
        print("  None! Perfect 100% classification across all 50 test cases.")

    # Save artifact
    output_artifact = {
        "dataset_size": total_queries,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "insufficient_precision": round(prec_ins, 4),
            "insufficient_recall": round(rec_ins, 4),
            "insufficient_f1": round(f1_ins, 4),
            "false_acceptance_rate": round(far, 4),
            "false_rejection_rate": round(frr, 4),
        },
        "confusion_matrix": matrix,
        "results": results,
    }

    out_file = processed_dir / "evidence_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_artifact, f, indent=2)
    print(f"\nSaved benchmark results to: {out_file}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_evidence_benchmark())
