"""Phase 4 artifact integrity verification script."""
import json
import numpy as np
from pathlib import Path

processed = Path(__file__).resolve().parent.parent / "data" / "processed"

# Load chunks
with open(processed / "chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load manifest
with open(processed / "embedding_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

# Load embeddings
vectors = np.load(processed / "embeddings.npy")

print("=== ARTIFACT INTEGRITY CHECK ===")
print()

# chunks.json
print(f"chunks.json entries:       {len(chunks)}")
chunk_ids = [c["chunk_id"] for c in chunks]
unique_ids = set(chunk_ids)
print(f"  Unique chunk IDs:        {len(unique_ids)} / {len(chunks)}")
print(f"  All IDs unique:          {len(unique_ids) == len(chunks)}")

# Check metadata completeness
missing_meta = 0
for c in chunks:
    m = c.get("metadata", {})
    if not m.get("source_file") or not m.get("category"):
        missing_meta += 1
print(f"  Missing metadata fields: {missing_meta}")

# Category breakdown
cats: dict[str, int] = {}
for c in chunks:
    cat = c["metadata"]["category"]
    cats[cat] = cats.get(cat, 0) + 1
for cat in sorted(cats):
    print(f"    {cat:30s}: {cats[cat]}")

# Token stats
tokens = [c["metadata"]["token_count"] for c in chunks]
print(f"  Token range:             {min(tokens)} - {max(tokens)} (avg {sum(tokens)/len(tokens):.1f})")
oversized = sum(1 for t in tokens if t > 600)
small = sum(1 for t in tokens if t < 15)
print(f"  Oversized (>600 tokens): {oversized}")
print(f"  Small (<15 tokens):      {small}")

print()

# embeddings.npy
print(f"embeddings.npy shape:      {vectors.shape}")
print(f"  dtype:                   {vectors.dtype}")
norms = np.linalg.norm(vectors, axis=1)
print(f"  L2 norm range:           [{norms.min():.6f}, {norms.max():.6f}]")
print(f"  All norms ~1.0:          {np.allclose(norms, 1.0, atol=1e-4)}")

print()

# embedding_manifest.json
model = manifest["model_name"]
dim = manifest["vector_dimension"]
tc = manifest["total_chunks"]
tv = manifest["total_vectors"]
norm_flag = manifest["normalized"]
bs = manifest["batch_size"]
gen_time = manifest.get("generation_time_seconds", "N/A")
created = manifest["created_at"]
mid_count = len(manifest["chunk_ids"])

print("embedding_manifest.json:")
print(f"  model_name:              {model}")
print(f"  vector_dimension:        {dim}")
print(f"  total_chunks:            {tc}")
print(f"  total_vectors:           {tv}")
print(f"  normalized:              {norm_flag}")
print(f"  batch_size:              {bs}")
print(f"  generation_time_seconds: {gen_time}")
print(f"  created_at:              {created}")
print(f"  chunk_ids count:         {mid_count}")

# Cross-checks
manifest_ids = set(manifest["chunk_ids"])
chunk_json_ids = set(chunk_ids)
print(f"  Manifest IDs == Chunk IDs: {manifest_ids == chunk_json_ids}")
print(f"  Vectors == Chunks:       {vectors.shape[0] == len(chunks)}")
print(f"  Vectors == Manifest:     {vectors.shape[0] == manifest['total_vectors']}")

print()
all_ok = (
    len(unique_ids) == len(chunks)
    and missing_meta == 0
    and vectors.shape[0] == len(chunks)
    and manifest_ids == chunk_json_ids
    and np.allclose(norms, 1.0, atol=1e-4)
)
print("=== ALL INTEGRITY CHECKS PASSED ===" if all_ok else "=== SOME CHECKS FAILED ===")
