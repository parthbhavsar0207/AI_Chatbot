"""Phase 10.2 — Local LLM Benchmark Script.

Benchmarks candidate local LLMs (Qwen3-4B vs Qwen2.5-3B) on:
1. Standardized 4K context across 6 test prompts.
2. Latency & throughput: TTFT, TPS, Load Time, Total Generation Time.
3. System & GPU resources: Peak dedicated VRAM, Peak System RAM, CPU/GPU utilization, layer offloading.
4. Coexistence with FastEmbed + ChromaDB + Cross-Encoder retrieval pipeline.
5. Qualitative evaluation across factual correctness, faithfulness, citation adherence, abstention, and edge cases.
"""

import sys
import os
import json
import time
import ctypes
import threading
import subprocess
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath("backend"))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Constants
OLLAMA_API_BASE = "http://127.0.0.1:11434"
MODELS_TO_BENCHMARK = [
    {
        "role": "Baseline Candidate",
        "tag": "qwen2.5:3b-instruct-q4_K_M",
        "name": "Qwen2.5-3B-Instruct (Q4_K_M)",
    },
    {
        "role": "Primary Candidate",
        "tag": "qwen3:4b-instruct-2507-q4_K_M",
        "name": "Qwen3-4B-Instruct-2507 (Q4_K_M)",
    },
]

# Standardized Prompts
BENCHMARK_PROMPTS = [
    {
        "id": "prompt_1_simple_factual",
        "category": "Simple factual question",
        "description": "Direct, single-hop factual query grounded in course syllabus details.",
        "context": (
            "[Document: CSE205_Data_Structures_Syllabus.txt]\n"
            "Course Title: Data Structures and Algorithms\n"
            "Course Code: CSE205\n"
            "Credits: 4 (L: 3, T: 1, P: 0)\n"
            "Prerequisite: Programming in C/C++ (CSE101 or CSE202) with minimum passing grade 'E'.\n"
            "Department: School of Computer Science and Engineering\n"
            "Course Objective: To introduce algorithmic thinking, time and space complexity, and linear/non-linear data structures."
        ),
        "prompt": "What are the prerequisites and credit count for CSE205?",
        "instruction": "Answer directly and concisely using only the provided context.",
    },
    {
        "id": "prompt_2_multipara_university",
        "category": "Multi-paragraph university-context question",
        "description": "Multi-paragraph policy explanation spanning attendance thresholds, debarment consequences, and medical regularization.",
        "context": (
            "[Document: Academic_Regulations_2024.pdf | Section 4: Attendance Regulations]\n"
            "Every registered student is required to maintain a minimum attendance of 75% in each registered course (theory and practical separately) to be eligible to appear in the End-Term Examination (ETE).\n\n"
            "If a student's attendance in a course falls below 75% but remains at or above 65%, condonation up to 10% may be granted strictly on medical grounds or for authorized university representation (e.g., sports, cultural, academic conferences), subject to approval by the Competent Authority.\n\n"
            "Students with final attendance below 65% (or below 75% without authorized condonation) shall be officially Debarred ('D' Grade) from taking the End-Term Examination in that specific course. A debarred student must re-register for the course as a backlog in subsequent terms.\n\n"
            "Medical Leave Regularization Process: A student seeking condonation on medical grounds must submit a formal Medical Regularization Application accompanied by an authentic medical certificate and hospital discharge summary to the Department Head / Proctor within 3 working days of returning to campus. Late submissions will be summarily rejected."
        ),
        "prompt": "Explain the attendance requirement policy, what happens if attendance falls below 75%, and how a student can apply for medical duty leaves.",
        "instruction": "Provide a comprehensive, structured response covering all points mentioned in the query based strictly on the provided context.",
    },
    {
        "id": "prompt_3_multiaspect_missing_detail",
        "category": "Multi-aspect question with one missing detail",
        "description": "Multi-aspect question where grading components and passing percentage are supplied, but minimum end-term score is missing.",
        "context": (
            "[Document: CSE306_Operating_Systems_Assessment_Scheme.txt]\n"
            "Course: CSE306 - Operating Systems\n"
            "Evaluation Structure:\n"
            "1. Continuous Assessment (CA): 25% (Best 3 out of 4 quizzes, programming assignments, and class participation)\n"
            "2. Mid-Term Examination (MTE): 25% (Covers Units 1 to 3, duration 90 minutes)\n"
            "3. End-Term Examination (ETE): 50% (Comprehensive, covers Units 1 to 6, duration 180 minutes)\n\n"
            "Overall Passing Criteria:\n"
            "A student must secure a minimum aggregate score of 40% across all continuous and term-end evaluation components combined to qualify for a passing grade ('E' or higher)."
        ),
        "prompt": "What are the grading components for CSE306, what is the overall passing percentage, and what is the minimum required score in the End-Term examination alone?",
        "instruction": "Answer each part of the question. If specific information for any aspect is not provided in the context, explicitly state that it is not available in the provided documents.",
    },
    {
        "id": "prompt_4_strict_instruction_following",
        "category": "Strict instruction-following question",
        "description": "Strict structural constraint: exactly 3 bullet points, each starting with an action verb, no intro or outro text.",
        "context": (
            "[Document: Hostel_Rules_and_Leave_Procedure_2024.txt]\n"
            "Standard hostel entry deadline is 8:30 PM for all residential blocks.\n"
            "Any resident planning to stay out overnight or travel home during weekends/vacation must complete the online gate-pass authorization workflow. "
            "First, the resident must apply for an overnight leave request on the University Management System (UMS) portal at least 24 hours prior to departure. "
            "Second, the student must obtain digital approval from their registered parent or legal guardian via the automated IVR/SMS verification system. "
            "Finally, the student must scan their biometric student ID and display the approved digital gate-pass QR code at the campus main security gate before exiting."
        ),
        "prompt": "What steps must a residential student follow when requesting an overnight hostel leave?",
        "instruction": (
            "STRICT CONSTRAINT: Provide your response in exactly three bullet points. "
            "Each bullet point must begin with an active action verb (e.g. 'Submit...', 'Obtain...', 'Scan...'). "
            "Do NOT include any introductory header, greeting, or concluding text."
        ),
    },
    {
        "id": "prompt_5_citation_requirement",
        "category": "Question requiring citations to supplied evidence",
        "description": "Explicit citation requirement citing exact source document brackets for every factual claim.",
        "context": (
            "[Doc 1: Academic_Regulations_2024.pdf]\n"
            "Clause 7.2: Grade Improvement Examination. A student who has obtained a grade 'D' or 'E' in any course may apply to appear in the Improvement Examination in subsequent regular terms to improve their grade. Students who obtained grade 'C' or above are ineligible for grade improvement.\n\n"
            "[Doc 2: Examination_Policy_FAQ.pdf]\n"
            "Q14: What is the procedure and timeline for answer script re-evaluation?\n"
            "Ans: A student dissatisfied with their ETE grade may apply for Answer Script Re-evaluation through the UMS portal within 10 calendar days from the date of official result declaration. The prescribed non-refundable re-evaluation fee is Rs. 500 per course."
        ),
        "prompt": "Can a student appear for an improvement exam if they received a 'C' grade, and what is the fee and deadline to apply for re-evaluation?",
        "instruction": "Answer the question factually and cite the exact source document in brackets (e.g. [Doc 1: Academic_Regulations_2024.pdf] or [Doc 2: Examination_Policy_FAQ.pdf]) for each factual statement.",
    },
    {
        "id": "prompt_6_unsupported_abstention",
        "category": "Unsupported question with explicit abstention instruction",
        "description": "Corpus contains only library access rules; user asks about admission fee refund timeline. Must abstain completely.",
        "context": (
            "[Document: Central_Library_Rules_and_Timings.txt]\n"
            "The Central Library remains open on all working days from 8:00 AM to 10:00 PM.\n"
            "Undergraduate students are eligible to borrow up to 4 books simultaneously for a maximum duration of 14 days.\n"
            "A late return fee of Rs. 5 per day per volume is levied on overdue books.\n"
            "Digital library access to IEEE Xplore, ScienceDirect, and ACM Digital Library is accessible via university campus IP authentication."
        ),
        "prompt": "What is the fee refund percentage if a student withdraws their admission in the 4th week of classes?",
        "instruction": (
            "Answer strictly based on the provided context. If the provided context does not contain sufficient information to answer the question, "
            "state clearly: 'I do not have sufficient information in the provided documents to answer this question.' Do not speculate or use outside knowledge."
        ),
    },
    {
        "id": "prompt_7_conflicting_evidence",
        "category": "Conflicting evidence handling",
        "description": "Two documents state conflicting rules for maximum credit registration limits. Model must detect and explain the conflict.",
        "context": (
            "[Document: Academic_Regulations_Archive_2022.pdf]\n"
            "Clause 3.1: The maximum permissible credit registration for a student in any regular semester is strictly capped at 24 credits. No exceptions or overloads are permitted under any circumstances.\n\n"
            "[Document: Academic_Regulations_Amended_2024.pdf]\n"
            "Clause 3.1 (Amended): Normal semester registration limit is 26 credits. However, students with a CGPA of 8.50 or higher may apply for credit overload up to 28 credits with prior written approval from the Academic Dean."
        ),
        "prompt": "What is the maximum number of credits a student can register for in a semester?",
        "instruction": "Evaluate the provided documents. If there is a contradiction or discrepancy between the documents, explicitly identify and explain the conflict.",
    },
]

# Memory Monitoring via Windows API
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

def get_system_ram_mb() -> Dict[str, float]:
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    total_mb = stat.ullTotalPhys / (1024 * 1024)
    used_mb = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 * 1024)
    return {
        "used_mb": round(used_mb, 1),
        "total_mb": round(total_mb, 1),
        "percent": stat.dwMemoryLoad,
    }

def get_gpu_metrics() -> Dict[str, Any]:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = res.stdout.strip()
        parts = [p.strip() for p in line.split(",")]
        return {
            "vram_used_mib": float(parts[0]),
            "vram_total_mib": float(parts[1]),
            "gpu_util_pct": float(parts[2]),
            "vram_util_pct": float(parts[3]),
        }
    except Exception as e:
        return {
            "vram_used_mib": 0.0,
            "vram_total_mib": 0.0,
            "gpu_util_pct": 0.0,
            "vram_util_pct": 0.0,
            "error": str(e),
        }

def get_loaded_models_ps() -> List[Dict[str, Any]]:
    try:
        req = urllib.request.Request(f"{OLLAMA_API_BASE}/api/ps")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("models", [])
    except Exception as e:
        return []

def call_ollama_generate_stream(
    model_tag: str,
    system_prompt: str,
    user_prompt: str,
    num_ctx: int = 4096,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Execute streaming inference call to accurately measure TTFT and token rate."""
    url = f"{OLLAMA_API_BASE}/api/generate"
    payload = {
        "model": model_tag,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "seed": 42,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    t_first_token = None
    response_tokens = []
    final_metadata = {}

    # Monitor peak resources during generation
    peak_vram = 0.0
    peak_ram = 0.0
    stop_monitor = threading.Event()
    resource_samples = []

    def monitor_resources():
        while not stop_monitor.is_set():
            gpu = get_gpu_metrics()
            ram = get_system_ram_mb()
            resource_samples.append((gpu["vram_used_mib"], ram["used_mb"], gpu["gpu_util_pct"]))
            time.sleep(0.05)

    mon_thread = threading.Thread(target=monitor_resources, daemon=True)
    mon_thread.start()

    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                now = time.perf_counter()
                if t_first_token is None and chunk.get("response"):
                    t_first_token = now
                response_tokens.append(chunk.get("response", ""))
                if chunk.get("done", False):
                    final_metadata = chunk
                    break
    finally:
        t_end = time.perf_counter()
        stop_monitor.set()
        mon_thread.join(timeout=1.0)

    # Compute resource peaks
    if resource_samples:
        peak_vram = max(s[0] for s in resource_samples)
        peak_ram = max(s[1] for s in resource_samples)
        avg_gpu_util = sum(s[2] for s in resource_samples) / len(resource_samples)
    else:
        current_gpu = get_gpu_metrics()
        peak_vram = current_gpu["vram_used_mib"]
        peak_ram = get_system_ram_mb()["used_mb"]
        avg_gpu_util = current_gpu["gpu_util_pct"]

    total_client_duration_s = t_end - t_start
    ttft_s = (t_first_token - t_start) if t_first_token else total_client_duration_s
    full_text = "".join(response_tokens).strip()

    # Ollama internal timing metadata (converted from nanoseconds)
    load_duration_s = final_metadata.get("load_duration", 0) / 1e9
    prompt_eval_duration_s = final_metadata.get("prompt_eval_duration", 0) / 1e9
    eval_duration_s = final_metadata.get("eval_duration", 0) / 1e9
    prompt_eval_count = final_metadata.get("prompt_eval_count", 0)
    eval_count = final_metadata.get("eval_count", 0)

    # Tokens per second during generation phase
    gen_tps = (eval_count / eval_duration_s) if eval_duration_s > 0 else 0.0

    return {
        "model": model_tag,
        "text": full_text,
        "total_client_time_s": round(total_client_duration_s, 3),
        "ttft_s": round(ttft_s, 3),
        "load_duration_s": round(load_duration_s, 3),
        "prompt_eval_duration_s": round(prompt_eval_duration_s, 3),
        "eval_duration_s": round(eval_duration_s, 3),
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "generation_tps": round(gen_tps, 2),
        "peak_vram_mib": round(peak_vram, 1),
        "peak_system_ram_mb": round(peak_ram, 1),
        "avg_gpu_util_pct": round(avg_gpu_util, 1),
    }

def run_coexistence_test(model_tag: str) -> Dict[str, Any]:
    """Test concurrent execution of FastEmbed + ChromaDB + Cross-Encoder while LLM infers."""
    print(f"\n[Coexistence] Testing concurrent execution with {model_tag}...")

    # Pre-test baseline resources
    ram_before = get_system_ram_mb()
    gpu_before = get_gpu_metrics()

    from fastembed import TextEmbedding
    import chromadb
    from app.rag.reranker import CrossEncoderReranker

    from app.rag.vector_store import VectorStoreService

    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    reranker = CrossEncoderReranker()
    vector_service = VectorStoreService()
    collection = vector_service.collection

    long_context_prompt = (
        "Generate a detailed, step-by-step 400-word explanation of how Operating Systems manage virtual memory, "
        "paging, page tables, TLB hits and misses, and page fault resolution routines."
    )

    llm_result = {}
    llm_error = None
    pipeline_result = {}
    pipeline_error = None

    def run_llm_worker():
        nonlocal llm_result, llm_error
        try:
            llm_result = call_ollama_generate_stream(
                model_tag=model_tag,
                system_prompt="You are an expert operating systems professor. Be thorough and detailed.",
                user_prompt=long_context_prompt,
                num_ctx=4096,
                temperature=0.2,
            )
        except Exception as e:
            llm_error = str(e)

    def run_pipeline_worker():
        nonlocal pipeline_result, pipeline_error
        time.sleep(0.3)  # Wait for LLM inference to be underway
        t0 = time.perf_counter()
        try:
            # 1. FastEmbed embedding generation
            queries = [
                "What is the attendance debarment criteria?",
                "How to apply for medical leave?",
                "What are the prerequisites for CSE205?",
                "What is the grading scheme for CSE306?",
            ]
            embeddings = list(embed_model.embed(queries))
            embed_duration = time.perf_counter() - t0

            # 2. ChromaDB vector search
            t1 = time.perf_counter()
            query_results = collection.query(
                query_embeddings=[embeddings[0].tolist()],
                n_results=5,
            )
            chroma_duration = time.perf_counter() - t1

            # 3. Cross-Encoder reranking
            t2 = time.perf_counter()
            rerank_scores = reranker.score_pairs(queries[0], query_results["documents"][0])
            rerank_duration = time.perf_counter() - t2

            pipeline_result = {
                "embed_duration_s": round(embed_duration, 4),
                "chroma_duration_s": round(chroma_duration, 4),
                "rerank_duration_s": round(rerank_duration, 4),
                "total_pipeline_s": round(time.perf_counter() - t0, 4),
                "retrieved_count": len(query_results["documents"][0]),
                "rerank_scores_sample": [round(float(s), 4) for s in rerank_scores[:3]],
            }
        except Exception as e:
            pipeline_error = str(e)

    t_llm = threading.Thread(target=run_llm_worker)
    t_pipe = threading.Thread(target=run_pipeline_worker)

    t_llm.start()
    t_pipe.start()

    t_pipe.join(timeout=30.0)
    t_llm.join(timeout=60.0)

    ram_after = get_system_ram_mb()
    gpu_after = get_gpu_metrics()

    ps_info = get_loaded_models_ps()

    return {
        "model_tag": model_tag,
        "llm_success": llm_error is None,
        "llm_error": llm_error,
        "llm_eval_count": llm_result.get("eval_count", 0),
        "llm_tps": llm_result.get("generation_tps", 0.0),
        "pipeline_success": pipeline_error is None,
        "pipeline_error": pipeline_error,
        "pipeline_metrics": pipeline_result,
        "ram_before_mb": ram_before["used_mb"],
        "ram_peak_mb": llm_result.get("peak_system_ram_mb", ram_after["used_mb"]),
        "vram_before_mib": gpu_before["vram_used_mib"],
        "vram_peak_mib": llm_result.get("peak_vram_mib", gpu_after["vram_used_mib"]),
        "vram_after_mib": gpu_after["vram_used_mib"],
        "active_models_in_ollama": ps_info,
    }


def main():
    print("=" * 80)
    print("PHASE 10.2 LOCAL LLM BENCHMARK SUITE")
    print("Models: Qwen2.5-3B (Baseline) vs Qwen3-4B (Primary)")
    print("=" * 80)

    # Verify Ollama server
    try:
        req = urllib.request.Request(f"{OLLAMA_API_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags_data = json.loads(resp.read().decode("utf-8"))
            installed_tags = [m["name"] for m in tags_data.get("models", [])]
            print(f"Ollama server responsive. Installed models: {installed_tags}")
    except Exception as e:
        print(f"Error connecting to Ollama at {OLLAMA_API_BASE}: {e}")
        sys.exit(1)

    system_prompt = (
        "You are an official academic assistant for LPU students. Answer questions using only the provided context. "
        "Follow all formatting and constraint instructions strictly. If the information is not present or cannot be "
        "determined from the context, abstain as instructed."
    )

    all_benchmark_results = {}

    for model_spec in MODELS_TO_BENCHMARK:
        model_tag = model_spec["tag"]
        model_name = model_spec["name"]
        print(f"\n{'#'*80}")
        print(f"BENCHMARKING MODEL: {model_name} ({model_tag})")
        print(f"{'#'*80}")

        model_results = []

        # Warm-up / load check
        print("\nExecuting warm-up query to check layer offloading and initial model load...")
        warmup = call_ollama_generate_stream(
            model_tag=model_tag,
            system_prompt=system_prompt,
            user_prompt="Context: [Course: CSE101]. Question: Name the course.",
            num_ctx=4096,
            temperature=0.0,
        )
        print(f"Warm-up complete. Initial Load Duration: {warmup['load_duration_s']}s, Peak VRAM: {warmup['peak_vram_mib']} MiB")

        # Inspect layer placement via /api/ps
        ps_models = get_loaded_models_ps()
        print(f"Layer offload status (/api/ps): {json.dumps(ps_models, indent=2)}")

        for p_idx, prompt_def in enumerate(BENCHMARK_PROMPTS, 1):
            p_id = prompt_def["id"]
            p_cat = prompt_def["category"]
            p_text = f"Context:\n{prompt_def['context']}\n\nQuestion: {prompt_def['prompt']}\n\nInstructions: {prompt_def['instruction']}"

            print(f"\n--- Running Test {p_idx}/6: [{p_cat}] ---")
            t_start = time.perf_counter()
            res = call_ollama_generate_stream(
                model_tag=model_tag,
                system_prompt=system_prompt,
                user_prompt=p_text,
                num_ctx=4096,
                temperature=0.0,
            )
            res["test_id"] = p_id
            res["test_category"] = p_cat
            res["description"] = prompt_def["description"]
            model_results.append(res)

            print(f"TTFT: {res['ttft_s']}s | Gen TPS: {res['generation_tps']} tok/s | Tokens: {res['eval_count']} | Total Time: {res['total_client_time_s']}s")
            print(f"Peak VRAM: {res['peak_vram_mib']} MiB | Peak RAM: {res['peak_system_ram_mb']} MB")
            print(f"Response:\n{res['text']}\n")

        # Coexistence Test
        coex_result = run_coexistence_test(model_tag)
        print(f"[Coexistence Summary] LLM Success: {coex_result['llm_success']} | Pipeline Success: {coex_result['pipeline_success']}")
        print(f"Total Pipeline Latency: {coex_result['pipeline_metrics'].get('total_pipeline_s')}s | Peak VRAM: {coex_result['vram_peak_mib']} MiB")

        all_benchmark_results[model_tag] = {
            "name": model_name,
            "role": model_spec["role"],
            "warmup": warmup,
            "layer_placement": ps_models,
            "tests": model_results,
            "coexistence": coex_result,
        }

    # Save to JSON
    output_path = "scripts/benchmark_results_phase10_2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_results, f, indent=2)
    print(f"\n[+] Saved full benchmark results to {output_path}")

if __name__ == "__main__":
    main()
