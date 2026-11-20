"""Evaluates semantic retrieval across representative queries on production ChromaDB."""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.retrieval import RetrievalService
from app.rag.vector_store import VectorStoreService


def main():
    chroma_dir = REPO_ROOT / "data" / "processed" / "chroma"
    vs = VectorStoreService(persist_directory=chroma_dir, collection_name="lpu_cse_aiml_docs")
    retriever = RetrievalService(vector_store=vs)

    queries = [
        {
            "query": "What are the units in CSE306?",
            "domain": "Syllabus",
            "expected_info": "Units / Course content for CSE306 (Operating Systems)",
        },
        {
            "query": "What is the attendance requirement at LPU?",
            "domain": "Regulations / FAQ",
            "expected_info": "75% minimum attendance requirement and relaxation rules",
        },
        {
            "query": "When is the INT335 examination scheduled?",
            "domain": "Examination Notice",
            "expected_info": "Date and timing for INT335 examination",
        },
        {
            "query": "What is the reappear examination policy?",
            "domain": "Regulations / FAQ",
            "expected_info": "Policy, eligibility, or procedure for reappearing / backlogs",
        },
        {
            "query": "What are the course outcomes of CSE276?",
            "domain": "Syllabus",
            "expected_info": "Course outcomes for CSE276 (Computer Networks)",
        },
        {
            "query": "What is the exam pattern for MTH401?",
            "domain": "Syllabus / Exam Notice",
            "expected_info": "Evaluation / Examination details for MTH401 (Discrete Mathematics)",
        },
    ]

    print("=" * 80)
    print(" LPU CSE-AIML RAG CHATBOT — RETRIEVAL VALIDATION BENCHMARK (PHASE 5B)")
    print(f" Production Collection: lpu_cse_aiml_docs ({vs.count()} records)")
    print("=" * 80)

    summary_records = []

    for idx, q_item in enumerate(queries, 1):
        q = q_item["query"]
        domain = q_item["domain"]
        expected = q_item["expected_info"]

        resp = retriever.retrieve_with_response(q, top_k=3)

        print(f"\nQuery {idx}: '{q}'")
        print(f"Domain: {domain} | Expected: {expected}")
        print(f"Retrieval Time: {resp.retrieval_time_seconds * 1000:.2f} ms")
        print("-" * 80)

        top_results = []
        for rank, chunk in enumerate(resp.results, 1):
            src = chunk.metadata.get("source_file", "unknown")
            cat = chunk.metadata.get("category", "unknown")
            course = chunk.metadata.get("course_code") or "N/A"
            title = chunk.metadata.get("title") or "N/A"
            text_preview = " ".join(chunk.text.split())[:180] + "..."

            print(f"  [{rank}] Chunk ID: {chunk.chunk_id}")
            print(f"      Source:     {src} (Category: {cat}, Course: {course})")
            print(f"      Title:      {title}")
            print(f"      Distance:   {chunk.distance:.4f} | Cosine Similarity: {chunk.similarity:.4f}")
            print(f"      Text:       {text_preview}\n")

            top_results.append({
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "source": src,
                "category": cat,
                "course_code": course,
                "distance": chunk.distance,
                "similarity": chunk.similarity,
                "text_snippet": text_preview,
            })

        # Evaluate if expected info is in the retrieved results
        all_text = " ".join(c.text.lower() for c in resp.results)
        q_lower = q.lower()
        success = False
        if "cse306" in q_lower and ("cse306" in all_text or "unit" in all_text):
            success = True
        elif "attendance" in q_lower and "attendance" in all_text:
            success = True
        elif "int335" in q_lower and "int335" in all_text:
            success = True
        elif "reappear" in q_lower and ("reappear" in all_text or "backlog" in all_text):
            success = True
        elif "cse276" in q_lower and ("cse276" in all_text or "outcome" in all_text or "network" in all_text):
            success = True
        elif "mth401" in q_lower and ("mth401" in all_text or "discrete" in all_text):
            success = True

        status_str = "SUCCESS (Expected information found in top-3)" if success else "PARTIAL/INSUFFICIENT"
        print(f"  => Evaluation: {status_str}")

        summary_records.append({
            "query": q,
            "domain": domain,
            "expected": expected,
            "success": success,
            "top_chunk": top_results[0] if top_results else None,
            "all_results": top_results,
        })

    # Save benchmark results
    out_file = REPO_ROOT / "data" / "processed" / "retrieval_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\nSaved evaluation benchmark to: {out_file}")


if __name__ == "__main__":
    main()
