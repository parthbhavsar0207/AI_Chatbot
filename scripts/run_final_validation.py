"""Independent Final Evaluation of EvidenceGate on Fresh 50-Query Dataset (Phase 9.3).

Executes 50 completely fresh, unseen queries against the unmodified Phase 9.2 Evidence Gate.
Calculates:
- Overall accuracy
- Per-state precision, recall, and F1
- 4x4 Confusion matrix
- False Acceptance Rate (FAR) with explicit formula
- Unsupported-Claim Acceptance Rate (distinguishing PARTIAL from SUFFICIENT)
- False Rejection Rate (FRR) with explicit formula
- Detailed failure analysis and root causes
- Provenance validation
"""

import json
from pathlib import Path
import time
from typing import Any

from app.rag.evidence_gate import EvidenceDecisionState, EvidenceGate
from app.rag.retrieval import RetrievalService
from app.schemas.retrieval import SearchRequest
from app.services.retrieval_service import execute_search, get_retrieval_service


def run_final_validation(
    dataset_path: str = "data/evaluation/final_validation_dataset.json",
    output_path: str = "data/processed/final_validation_benchmark_results.json",
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

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

    print(f"\n================================================================================")
    print(f" PHASE 9.3 FINAL INDEPENDENT VALIDATION BENCHMARK ({total_count} QUERIES)")
    print(f"================================================================================")

    provenance_check_passed = True

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

        retrieved_items, elapsed_ms, decision = execute_search(req, service)

        pred_decision = decision.decision.value if decision else "insufficient"
        is_correct = (pred_decision == expected)
        if is_correct:
            correct_count += 1

        confusion_matrix[expected][pred_decision] += 1

        # Provenance check: verify schedule is never marked official
        if decision and "lpu_cse_aiml_upcoming_end_term_examinations_2026-27.docx" in decision.source_authorities:
            auth = decision.source_authorities["lpu_cse_aiml_upcoming_end_term_examinations_2026-27.docx"]
            if auth != "curated_schedule_snapshot":
                provenance_check_passed = False

        top_chunk = retrieved_items[0] if retrieved_items else None
        record = {
            "query_id": qid,
            "query": query,
            "category": category,
            "expected_decision": expected,
            "predicted_decision": pred_decision,
            "is_correct": is_correct,
            "reason_code": decision.reason_code if decision else None,
            "heuristic_score": decision.heuristic_score if decision else 0.0,
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
        print(f"{mark} {qid} ({category:12s}): Exp={expected:12s} | Got={pred_decision:12s} | Score={record['heuristic_score']:.2f} | Rerank={record['top_reranker_score']}")

    # Metrics calculation
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Per-state precision, recall, F1
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

    # Formulas:
    # 1. Broad False Acceptance Rate (FAR):
    #    Fraction of actually insufficient queries that are falsely accepted as either SUFFICIENT or PARTIAL
    actual_insufficient_total = sum(confusion_matrix["insufficient"].values())
    insufficient_to_sufficient = confusion_matrix["insufficient"]["sufficient"]
    insufficient_to_partial = confusion_matrix["insufficient"]["partial"]
    broad_far = (
        (insufficient_to_sufficient + insufficient_to_partial) / actual_insufficient_total
        if actual_insufficient_total > 0
        else 0.0
    )

    # 2. Unsupported-Claim Acceptance Rate (Strict / Catastrophic FAR):
    #    Fraction of actually insufficient queries that are accepted as SUFFICIENT (hallucination risk)
    strict_unsupported_claim_acceptance_rate = (
        insufficient_to_sufficient / actual_insufficient_total
        if actual_insufficient_total > 0
        else 0.0
    )

    # 3. False Rejection Rate (FRR):
    #    Fraction of actually sufficient queries that are rejected as INSUFFICIENT
    actual_sufficient_total = sum(confusion_matrix["sufficient"].values())
    sufficient_to_insufficient = confusion_matrix["sufficient"]["insufficient"]
    frr = (
        sufficient_to_insufficient / actual_sufficient_total
        if actual_sufficient_total > 0
        else 0.0
    )

    summary = {
        "dataset_size": total_count,
        "correct_count": correct_count,
        "overall_accuracy": round(accuracy, 4),
        "provenance_check_passed": provenance_check_passed,
        "metrics": {
            "broad_false_acceptance_rate": round(broad_far, 4),
            "unsupported_claim_acceptance_rate": round(strict_unsupported_claim_acceptance_rate, 4),
            "false_rejection_rate": round(frr, 4),
            "formulas": {
                "broad_false_acceptance_rate": "(insufficient_to_sufficient + insufficient_to_partial) / total_actual_insufficient",
                "unsupported_claim_acceptance_rate": "insufficient_to_sufficient / total_actual_insufficient",
                "false_rejection_rate": "sufficient_to_insufficient / total_actual_sufficient",
            },
        },
        "per_state_metrics": per_state_metrics,
        "confusion_matrix": confusion_matrix,
        "failures": [r for r in results if not r["is_correct"]],
        "results": results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================================================================================")
    print(f" BENCHMARK SUMMARY RESULTS")
    print(f"================================================================================")
    print(f"Total Queries:                                {total_count}")
    print(f"Correct Decisions:                            {correct_count} / {total_count}")
    print(f"Overall Decision Accuracy:                    {accuracy * 100:.2f}%")
    print(f"Unsupported-Claim Acceptance Rate (to SUFF):  {strict_unsupported_claim_acceptance_rate * 100:.2f}% ({insufficient_to_sufficient}/{actual_insufficient_total})")
    print(f"Broad False Acceptance Rate (to PART/SUFF):   {broad_far * 100:.2f}% ({(insufficient_to_sufficient + insufficient_to_partial)}/{actual_insufficient_total})")
    print(f"False Rejection Rate (to INSUF):              {frr * 100:.2f}% ({sufficient_to_insufficient}/{actual_sufficient_total})")
    print(f"Provenance Validation Passed:                 {provenance_check_passed}")
    print("--------------------------------------------------------------------------------")
    print("Per-State Breakdown:")
    for s_val, m in per_state_metrics.items():
        print(f"  {s_val.upper():12s} - Precision: {m['precision'] * 100:.2f}%, Recall: {m['recall'] * 100:.2f}%, F1: {m['f1']:.4f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
    print("--------------------------------------------------------------------------------")
    print("4x4 Confusion Matrix (Actual Rows x Predicted Cols):")
    states = [s.value for s in EvidenceDecisionState]
    header = f"{'Actual':12s} | " + " | ".join(f"{s[:6]:6s}" for s in states) + " | Total"
    print(header)
    print("-" * len(header))
    for act in states:
        total_act = sum(confusion_matrix[act].values())
        row = f"{act:12s} | " + " | ".join(f"{confusion_matrix[act][p]:6d}" for p in states) + f" | {total_act:5d}"
        print(row)
    print("--------------------------------------------------------------------------------")
    if summary["failures"]:
        print(f"Failures Encountered ({len(summary['failures'])}):")
        for fail in summary["failures"]:
            print(f"  * [{fail['query_id']}] {fail['query']}")
            print(f"      Expected: {fail['expected_decision']} | Predicted: {fail['predicted_decision']} (Reason: {fail['reason_code']})")
            print(f"      Top Reranker Score: {fail['top_reranker_score']} | Sim: {fail['top_similarity']:.4f}")
    else:
        print("PERFECT 100% ACCURACY! Zero misclassifications encountered.")
    print("================================================================================\n")

    return summary


if __name__ == "__main__":
    run_final_validation()
