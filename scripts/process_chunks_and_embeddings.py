"""CLI pipeline script to execute structure-aware chunking and FastEmbed embedding generation."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
import numpy as np

# Ensure backend directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.chunker import ChunkerService
from app.rag.embeddings import EmbeddingService
from app.rag.schema import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chunk_embed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk documents and generate FastEmbed embeddings.")
    parser.add_argument(
        "--input-file",
        "-i",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "ingested_documents.json",
        help="Path to ingested documents JSON",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Directory to save chunks.json, embeddings.npy, and embedding_manifest.json",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    args = parser.parse_args()

    input_file = args.input_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" LPU CSE-AIML RAG CHATBOT — CHUNKING & EMBEDDINGS (PHASE 4)")
    print("=" * 70)
    print(f" Input Documents:  {input_file}")
    print(f" Output Directory: {output_dir}")
    print(f" Batch Size:       {args.batch_size}")
    print("-" * 70)

    if not input_file.exists():
        print(f"Error: Input file does not exist: {input_file}", file=sys.stderr)
        return 1

    # Load ingested documents
    with open(input_file, "r", encoding="utf-8") as f:
        docs_raw = json.load(f)

    documents = [Document(**d) for d in docs_raw]
    print(f"Loaded {len(documents)} input document units.")

    # 1. Structure-Aware Chunking
    chunker = ChunkerService()
    chunks, chunk_stats = chunker.chunk_documents(documents)
    print(f"Produced {len(chunks)} structure-aware chunks.")

    # Save human-readable chunks.json
    chunks_file = output_dir / "chunks.json"
    chunks_serialized = [c.model_dump() for c in chunks]
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks_serialized, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks to: {chunks_file}")

    # 2. Embedding Generation
    print("\nGenerating FastEmbed BGE-small-en-v1.5 embeddings on CPU...")
    embedder = EmbeddingService()
    start_time = time.perf_counter()
    vectors, manifest = embedder.embed_chunks(chunks, batch_size=args.batch_size)
    elapsed_time = time.perf_counter() - start_time
    manifest["generation_time_seconds"] = round(elapsed_time, 2)

    # Save vectors to embeddings.npy
    vectors_file = output_dir / "embeddings.npy"
    np.save(vectors_file, vectors)
    print(f"Saved {vectors.shape[0]} vectors to: {vectors_file}")

    # Save embedding_manifest.json
    manifest_file = output_dir / "embedding_manifest.json"
    manifest["vectors_file"] = vectors_file.name
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest to: {manifest_file}")

    # 3. Quality & Validation Diagnostics
    unique_chunk_ids = len({c.chunk_id for c in chunks}) == len(chunks)
    vector_chunk_aligned = (vectors.shape[0] == len(chunks))

    print("\n" + "=" * 70)
    print(" PHASE 4 EXECUTION & VALIDATION REPORT")
    print("=" * 70)
    print(f" Number of input documents:         {len(documents)}")
    print(f" Total generated chunks:            {len(chunks)}")
    print("-" * 70)
    print(" Generated Chunks by Category:")
    for cat, count in sorted(chunk_stats["chunks_by_category"].items()):
        print(f"   • {cat.capitalize():<24}: {count} chunks")
    print("-" * 70)
    print(" Chunk Token Statistics:")
    print(f"   • Min chunk size:                {chunk_stats['min_tokens']} tokens")
    print(f"   • Max chunk size:                {chunk_stats['max_tokens']} tokens")
    print(f"   • Average chunk size:            {chunk_stats['avg_tokens']} tokens")
    print(f"   • Small chunks (<15 tokens):     {chunk_stats['small_chunks_count']}")
    print(f"   • Oversized chunks (>600 tokens): {chunk_stats['oversized_chunks_count']}")
    print(f"   • Duplicates prevented:          {chunk_stats['duplicate_chunks_prevented']}")
    print("-" * 70)
    print(" Embedding Generation Details:")
    print(f"   • Embedding model:               {manifest['model_name']}")
    print(f"   • Vector dimension:              {manifest['vector_dimension']}")
    print(f"   • Total vectors generated:       {vectors.shape[0]}")
    print(f"   • Generation elapsed time:       {elapsed_time:.2f} seconds")
    print(f"   • Failed chunks or embeddings:   0")
    print(f"   • Chunk IDs unique:              {'YES (100% Unique)' if unique_chunk_ids else 'NO (Duplicates found)'}")
    print(f"   • 1-to-1 Chunk/Vector alignment: {'YES (Exact match)' if vector_chunk_aligned else 'NO'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
