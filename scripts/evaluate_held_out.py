"""Independent Evaluation of EvidenceGate on Held-Out Evaluation Dataset (Phase 9.1).

Runs 32 completely held-out test queries without tuning any rules or thresholds.
Calculates:
- Overall accuracy
- Per-state precision and recall
- Insufficient-evidence false acceptance rate (FAR)
- Supported-query false rejection rate (FRR)
- 4x4 Confusion matrix
- Detailed failure analysis
"""

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

from app.rag.evidence_gate import EvidenceDecisionState, EvidenceGate
from app.rag.retrieval import RetrievalService
from app.schemas.retrieval import SearchRequest
from app.services.retrieval_service import execute_search, get_retrieval_service


def run_held_out_benchmark(
    dataset_path: str = "data/evaluation/held_out_evidence_dataset.json",
    output_path: str = "data/processed/held_out_benchmark_results.json",
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Held-out dataset not found at {dataset_path}")

    with open(dataset_file, encoding="utf-8") as f:
        dataset = json.load(f)

    service: RetrievalService = get_retrieval_service()
    gate: EvidenceGate = service.evidence_gate

    results = []
    confusion_matrix = {
        state.value: {s.value: 0 for s in EvidenceDecisionState}
        for state in EvidenceDecisionState
    }

    correct_count = 0
    total_count = len(dataset)

    print(f"\n==================================================================")
    print(f"Executing Independent Held-Out Benchmark ({total_count} queries)")
    print(f"==================================================================")

    for item in dataset:
        qid = item["query_id"]
        query = item["query"]
        expected = item["expected_decision"]
        category = item.get("category", "unspecified")

        req = SearchRequest(
            query=query,
            top_k=5,
            enable_rerank=True,
            candidate_k=10,
            evaluate_evidence=True,
        )

        t0 = time.perf_counter()
        retrieved_items, elapsed_ms, decision = execute_search(req, service)
        t1 = time.perf_counter()

        pred_decision = decision.decision.value if decision else "insufficient"
        is_correct = (pred_decision == expected)
        if is_correct:
            correct_count += 1

        confusion_matrix[expected][pred_decision] += 1

        top_chunk = retrieved_items[0] if retrieved_items else None
        record = {
            "query_id": qid,
            "query": query,
            "category": category,
            "expected_decision": expected,
            "predicted_decision": pred_decision,
            "is_correct": is_correct,
            "reason_code": decision.reason_code if decision else None,
            "confidence_score": decision.confidence_score if decision else 0.0,
            "missing_information": decision.missing_information if decision else None,
            "conflict_details": decision.conflict_details if decision else None,
            "top_similarity": top_chunk.similarity if top_chunk else 0.0,
            "top_reranker_score": top_chunk.reranker_score if top_chunk else None,
            "supporting_chunk_ids": decision.supporting_chunk_ids if decision else [],
            "relevant_sources": decision.relevant_sources if decision else [],
        }
        results.append(record)

        mark = "[PASS]" if is_correct else "[FAIL]"
        print(f"{mark} {qid} ({category}): Expected={expected}, Got={pred_decision} (Conf={record['confidence_score']:.2f}, Rerank={record['top_reranker_score']})")

    # Metrics calculation
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Per-state precision and recall
    per_state_metrics = {}
    for state in EvidenceDecisionState:
        s_val = state.value
        tp = confusion_matrix[s_val][s_val]
        fn = sum(confusion_matrix[s_val][pred] for pred in confusion_matrix[s_val] if pred != s_val)
        fp = sum(confusion_matrix[act][s_val] for act in confusion_matrix if act != s_val)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_state_metrics[s_val] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Insufficient-evidence False Acceptance Rate:
    # Actual insufficient queries classified as sufficient or partial
    insufficient_actual = sum(confusion_matrix["insufficient"].values())
    insufficient_false_accepts = (
        confusion_matrix["insufficient"]["sufficient"] + confusion_matrix["insufficient"]["partial"]
    )
    far = (insufficient_false_accepts / insufficient_actual) if insufficient_actual > 0 else 0.0

    # Supported-query False Rejection Rate:
    # Actual sufficient queries classified as insufficient
    sufficient_actual = sum(confusion_matrix["sufficient"].values())
    sufficient_false_rejections = confusion_matrix["sufficient"]["insufficient"]
    frr = (sufficient_false_rejections / sufficient_actual) if sufficient_actual > 0 else 0.0

    summary = {
        "dataset_size": total_count,
        "correct_count": correct_count,
        "overall_accuracy": round(accuracy, 4),
        "false_acceptance_rate": round(far, 4),
        "false_rejection_rate": round(frr, 4),
        "per_state_metrics": per_state_metrics,
        "confusion_matrix": confusion_matrix,
        "failures": [r for r in results if not r["is_correct"]],
        "results": results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n------------------------------------------------------------------")
    print(f"Overall Accuracy: {accuracy * 100:.2f}% ({correct_count}/{total_count})")
    print(f"False Acceptance Rate (FAR): {far * 100:.2f}% ({insufficient_false_accepts}/{insufficient_actual})")
    print(f"False Rejection Rate (FRR):  {frr * 100:.2f}% ({sufficient_false_rejections}/{sufficient_actual})")
    print("------------------------------------------------------------------")
    print("Per-State Breakdown:")
    for s_val, m in per_state_metrics.items():
        print(f"  {s_val.upper():12s} - Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1']:.4f}")
    print("------------------------------------------------------------------")
    print(f"Confusion Matrix (Actual Rows x Predicted Cols):")
    states = [s.value for s in EvidenceDecisionState]
    header = f"{'Actual':12s} | " + " | ".join(f"{s[:6]:6s}" for s in states)
    print(header)
    for act in states:
        row = f"{act:12s} | " + " | ".join(f"{confusion_matrix[act][p]:6d}" for p in states)
        print(row)
    print("------------------------------------------------------------------")
    if summary["failures"]:
        print(f"\nFailures Encountered ({len(summary['failures'])}):")
        for fail in summary["failures"]:
            print(f"  * {fail['query_id']}: Expected={fail['expected_decision']}, Predicted={fail['predicted_decision']}")
            print(f"    Query: {fail['query']}")
            print(f"    Reranker score: {fail['top_reranker_score']}, Sim: {fail['top_similarity']:.4f}")
            print(f"    Reason code: {fail['reason_code']}")
    else:
        print("\nZero Failures! All held-out queries correctly classified!")
    print("------------------------------------------------------------------\n")

    return summary


if __name__ == "__main__":
    run_held_out_benchmark()
