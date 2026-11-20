"""Real-Model Live Integration Test Suite for Phase 10.3.

Executes end-to-end integration tests using the live Ollama service (127.0.0.1:11434)
and actual ChromaDB indexed chunks from data/processed/chroma.

Collects and records:
- User query
- Actual retrieved chunk IDs
- Actual reranked chunk IDs
- Evidence gate decision and state
- Exact context supplied to LLM
- Generated answer text
- Programmatic citation validation report
- Resource metrics: Physical RAM (used, available), Process RAM (working set, private), GPU VRAM (used, total)
- Latency breakdown: retrieval, reranking, generation, total
"""

import asyncio
import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.abspath("backend"))

from app.rag.citation_validator import CitationValidator
from app.rag.evidence_gate import EvidenceDecisionState
from app.schemas.chat import ChatAnswerRequest
from app.services.rag_service import RAGChatService


# --- Windows Memory Helpers ---
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_system_ram() -> Dict[str, float]:
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    total_gb = stat.ullTotalPhys / (1024**3)
    avail_gb = stat.ullAvailPhys / (1024**3)
    used_gb = total_gb - avail_gb
    return {
        "total_gb": round(total_gb, 2),
        "used_gb": round(used_gb, 2),
        "avail_gb": round(avail_gb, 2),
        "load_pct": stat.dwMemoryLoad,
    }


def get_process_ram() -> Dict[str, float]:
    pmc = PROCESS_MEMORY_COUNTERS_EX()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
    GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
    return {
        "working_set_mb": round(pmc.WorkingSetSize / (1024**2), 2),
        "private_mb": round(pmc.PrivateUsage / (1024**2), 2),
    }


def get_gpu_vram() -> Dict[str, float]:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = res.stdout.strip().split(",")
        return {
            "vram_used_mib": float(parts[0].strip()),
            "vram_total_mib": float(parts[1].strip()),
        }
    except Exception as e:
        return {"vram_used_mib": 0.0, "vram_total_mib": 0.0, "error": str(e)}


REAL_MODEL_TEST_CASES = [
    {
        "id": "test_1_supported_syllabus",
        "name": "Supported Syllabus Question",
        "query": "What are the prerequisites and credit count for CSE205?",
        "course_code": "CSE205",
        "category": "syllabus",
        "expected_state": EvidenceDecisionState.SUFFICIENT,
    },
    {
        "id": "test_2_supported_regulation",
        "name": "Supported Regulation Question",
        "query": "What is the minimum attendance required and what happens if attendance falls below 75%?",
        "category": "regulations",
        "expected_state": EvidenceDecisionState.SUFFICIENT,
    },
    {
        "id": "test_3_partial_missing_detail",
        "name": "Partially Supported Multi-Aspect Question",
        "query": "What are the grading components for CSE306 and what is the minimum required score in the End-Term examination alone?",
        "course_code": "CSE306",
        "expected_state": EvidenceDecisionState.PARTIAL,
    },
    {
        "id": "test_4_multiaspect_question",
        "name": "Multi-Aspect Question",
        "query": "What are the attendance debarment thresholds and within how many days must medical regularization be submitted?",
        "category": "regulations",
        "expected_state": EvidenceDecisionState.SUFFICIENT,
    },
    {
        "id": "test_5_citation_correctness",
        "name": "Citation Correctness Test",
        "query": "What is the course objective and prerequisite for CSE202 Object Oriented Programming?",
        "course_code": "CSE202",
        "expected_state": EvidenceDecisionState.SUFFICIENT,
    },
    {
        "id": "test_6_unsupported_abstention",
        "name": "Unsupported Question (Deterministic Abstention)",
        "query": "What is the fee refund percentage if a student withdraws their admission in the 4th week of classes?",
        "expected_state": EvidenceDecisionState.INSUFFICIENT,
    },
]


async def run_tests():
    print("=" * 80)
    print("PHASE 10.3 REAL-MODEL LIVE INTEGRATION TESTS")
    print("Runtime: Ollama (127.0.0.1:11434) | Model: qwen3:4b-instruct-2507-q4_K_M")
    print("=" * 80)

    service = RAGChatService()

    # Pre-check Ollama health
    print("\nChecking Ollama service connectivity...")
    health = await service.llm_service.check_health()
    print(f"Ollama Health: {health['status']} | Available models: {health['available_models']}")
    if not health.get("primary_model_ready"):
        print(f"ERROR: Primary model {service.llm_service.primary_model} is NOT available on Ollama!")
        sys.exit(1)

    all_results = []

    for idx, tc in enumerate(REAL_MODEL_TEST_CASES, 1):
        t_id = tc["id"]
        t_name = tc["name"]
        query = tc["query"]
        print(f"\n{'='*80}")
        print(f"RUNNING TEST {idx}/{len(REAL_MODEL_TEST_CASES)}: [{t_name}]")
        print(f"Query: {query}")
        print(f"{'='*80}")

        # Baseline resources before request
        sys_ram_pre = get_system_ram()
        proc_ram_pre = get_process_ram()
        gpu_pre = get_gpu_vram()

        req = ChatAnswerRequest(
            query=query,
            top_k=5,
            category=tc.get("category"),
            course_code=tc.get("course_code"),
        )

        t0 = time.perf_counter()
        resp = await service.answer_question(req)
        wall_time = time.perf_counter() - t0

        # Resources after request
        sys_ram_post = get_system_ram()
        proc_ram_post = get_process_ram()
        gpu_post = get_gpu_vram()

        # Log details
        print(f"Evidence State: {resp.evidence_state.value} (Expected: {tc['expected_state'].value})")
        print(f"Model Identifier: {resp.model_identifier}")
        print(f"Latencies: Total={resp.total_latency_ms}ms | Retrieval={resp.retrieval_latency_ms}ms | Rerank={resp.reranking_latency_ms}ms | Gen={resp.generation_latency_ms}ms")
        print(f"Citation Validation Status: {resp.citation_validation_status.value}")
        print(f"Valid Citations Count: {resp.citation_validation.valid_citations_count}")
        print(f"Invalid Citations: {resp.citation_validation.invalid_citations}")
        print(f"Structural Coverage Passed: {resp.citation_validation.structural_coverage_passed}")
        print(f"Missing Information: {resp.missing_information}")
        print(f"\nGenerated Answer:\n{resp.answer}\n")
        print(f"Citations Metadata ({len(resp.citations)} items):")
        for c in resp.citations:
            print(f"  - Chunk: {c.chunk_id} | File: {c.source_file} | Title: {c.title}")
        print(f"Source Authenticity Metadata ({len(resp.source_metadata)} items):")
        for s in resp.source_metadata:
            print(f"  - File: {s.source_file} | Auth: {s.authority_level} | Status: {s.verification_status}")

        print(f"\nResource Metrics:")
        print(f"  - Physical System RAM: {sys_ram_post['used_gb']} GB used / {sys_ram_post['total_gb']} GB total (Available: {sys_ram_post['avail_gb']} GB)")
        print(f"  - Python Process RAM: Working Set = {proc_ram_post['working_set_mb']} MB, Private = {proc_ram_post['private_mb']} MB")
        print(f"  - Dedicated GPU VRAM: {gpu_post['vram_used_mib']} MiB used / {gpu_post['vram_total_mib']} MiB total")

        record = {
            "test_id": t_id,
            "test_name": t_name,
            "query": query,
            "expected_state": tc["expected_state"].value,
            "actual_state": resp.evidence_state.value,
            "model_identifier": resp.model_identifier,
            "answer": resp.answer,
            "citation_validation_status": resp.citation_validation_status.value,
            "valid_citations_count": resp.citation_validation.valid_citations_count,
            "invalid_citations": resp.citation_validation.invalid_citations,
            "structural_coverage_passed": resp.citation_validation.structural_coverage_passed,
            "citations": [c.model_dump() for c in resp.citations],
            "source_metadata": [s.model_dump() for s in resp.source_metadata],
            "missing_information": resp.missing_information,
            "conflict_details": resp.conflict_details,
            "latency": {
                "retrieval_ms": resp.retrieval_latency_ms,
                "rerank_ms": resp.reranking_latency_ms,
                "generation_ms": resp.generation_latency_ms,
                "total_ms": resp.total_latency_ms,
                "wall_time_s": round(wall_time, 3),
            },
            "resources": {
                "system_ram_gb": sys_ram_post,
                "process_ram_mb": proc_ram_post,
                "gpu_vram_mib": gpu_post,
            },
        }
        all_results.append(record)

    output_path = "scripts/real_model_integration_results_phase10_3.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Successfully saved real-model integration results to {output_path}")


if __name__ == "__main__":
    asyncio.run(run_tests())
