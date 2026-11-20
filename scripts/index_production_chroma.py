"""Production ChromaDB indexing script for LPU CSE-AIML RAG Chatbot.

Validates Phase 4 artifacts and indexes all 65 pre-embedded chunks into the
persistent ChromaDB collection 'lpu_cse_aiml_docs'.
"""

import json
import logging
from pathlib import Path
import sys
import time
import numpy as np

# Ensure backend directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.schema import Chunk
from app.rag.vector_store import VectorStoreService, EXPECTED_VECTOR_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("production_indexer")


def main() -> int:
    processed_dir = REPO_ROOT / "data" / "processed"
    chunks_path = processed_dir / "chunks.json"
    embeddings_path = processed_dir / "embeddings.npy"
    manifest_path = processed_dir / "embedding_manifest.json"
    chroma_dir = processed_dir / "chroma"

    print("=" * 70)
    print(" LPU CSE-AIML RAG CHATBOT — PRODUCTION CHROMADB INDEXING (PHASE 5B)")
    print("=" * 70)
    print(f" Chunks file:     {chunks_path}")
    print(f" Embeddings file: {embeddings_path}")
    print(f" Manifest file:   {manifest_path}")
    print(f" Chroma dir:      {chroma_dir}")
    print("-" * 70)

    # 1. Load artifacts
    for p in (chunks_path, embeddings_path, manifest_path):
        if not p.exists():
            print(f"ERROR: Artifact not found: {p}", file=sys.stderr)
            return 1

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    embeddings = np.load(embeddings_path)

    # 2. Validation
    print("Validating input artifacts...")
    chunks = [Chunk(**c) for c in chunks_data]
    num_chunks = len(chunks)
    num_embeddings = embeddings.shape[0]

    print(f"  • Chunks loaded:      {num_chunks}")
    print(f"  • Embeddings shape:   {embeddings.shape}")
    print(f"  • Embeddings dtype:   {embeddings.dtype}")

    if num_chunks != 65:
        raise ValueError(f"Expected exactly 65 chunks, found {num_chunks}")
    if num_embeddings != 65:
        raise ValueError(f"Expected exactly 65 embeddings, found {num_embeddings}")
    if embeddings.shape[1] != EXPECTED_VECTOR_DIM:
        raise ValueError(f"Expected dimension {EXPECTED_VECTOR_DIM}, found {embeddings.shape[1]}")

    # Absence of invalid values (NaN, Inf)
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings contain non-finite values (NaN or Inf)")

    # L2 normalization check
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(f"Embeddings are not L2-normalized. Norm range: [{norms.min()}, {norms.max()}]")

    # ID alignment and uniqueness
    chunk_ids = [c.chunk_id for c in chunks]
    unique_ids = set(chunk_ids)
    if len(unique_ids) != 65:
        raise ValueError(f"Chunk IDs are not unique! Found {len(unique_ids)} unique out of {num_chunks}")

    manifest_ids = manifest_data.get("chunk_ids", [])
    if chunk_ids != manifest_ids:
        raise ValueError("1-to-1 chunk ID order mismatch between chunks.json and embedding_manifest.json")

    print("  • All artifact integrity checks PASSED (1:1 alignment, normalized, finite, unique).")
    print("-" * 70)

    # 3. Production Indexing
    print(f"Indexing {num_chunks} chunks into persistent ChromaDB collection 'lpu_cse_aiml_docs'...")
    vector_store = VectorStoreService(
        persist_directory=chroma_dir,
        collection_name="lpu_cse_aiml_docs",
    )

    start_time = time.perf_counter()
    summary = vector_store.index_chunks(chunks, embeddings, batch_size=32)
    indexing_time = time.perf_counter() - start_time

    count_first = vector_store.count()
    print(f"  • Indexing time:      {indexing_time:.4f} seconds")
    print(f"  • Records in store:   {count_first}")

    if count_first != 65:
        raise ValueError(f"Expected 65 records in collection after indexing, got {count_first}")

    # 4. Idempotence verification
    print("Testing idempotence: Re-running upsert on identical records...")
    start_idem = time.perf_counter()
    summary_idem = vector_store.index_chunks(chunks, embeddings, batch_size=32)
    idem_time = time.perf_counter() - start_idem

    count_after_idem = vector_store.count()
    print(f"  • Idempotence time:   {idem_time:.4f} seconds")
    print(f"  • Records after rerun:{count_after_idem}")

    if count_after_idem != 65:
        raise ValueError(f"Idempotency failed! Record count changed from 65 to {count_after_idem}")

    print("  • Idempotence verified: Count remained exactly 65 without duplicates.")
    print("=" * 70)
    print(" PRODUCTION CHROMADB INDEXING SUMMARY")
    print("=" * 70)
    print(f" Collection Name:        lpu_cse_aiml_docs")
    print(f" Persist Directory:      {chroma_dir}")
    print(f" Total Records Indexed:  {count_after_idem}")
    print(f" Initial Indexing Time:  {indexing_time:.4f}s")
    print(f" Idempotent Upsert Time: {idem_time:.4f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
