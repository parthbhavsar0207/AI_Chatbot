"""Unit tests for FastEmbed embedding generation service."""

import numpy as np
import pytest

from app.rag.embeddings import DEFAULT_MODEL_NAME, DEFAULT_VECTOR_DIM, EmbeddingService
from app.rag.schema import Chunk, ChunkMetadata


def create_sample_chunk(text: str, index: int = 0) -> Chunk:
    meta = ChunkMetadata(
        source_file="test.pdf",
        category="syllabus",
        file_type="pdf",
        title="Sample Course",
        course_code="CSE205",
        page_number=1,
        chunk_index=index,
        total_chunks=1,
    )
    return Chunk.create(document_id=f"doc_{index}", text=text, metadata=meta)


def test_embedding_dimension_and_normalization():
    service = EmbeddingService(model_name=DEFAULT_MODEL_NAME)
    texts = [
        "Lovely Professional University Computer Science and Engineering syllabus.",
        "Examination guidelines for objective and subjective end-term exams.",
    ]
    vectors = service.embed_texts(texts, batch_size=2, normalize=True)

    # 1. Dimension check (BGE-small-en-v1.5 has 384 dimensions)
    assert vectors.shape == (2, DEFAULT_VECTOR_DIM)

    # 2. L2 Normalization check (||v||_2 ≈ 1.0)
    for vec in vectors:
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-3)


def test_embedding_batch_processing():
    service = EmbeddingService(model_name=DEFAULT_MODEL_NAME)
    # 7 items with batch_size=3 verifies partial final batch handling
    texts = [f"Text sample sentence number {i} for batching test." for i in range(7)]
    vectors = service.embed_texts(texts, batch_size=3, normalize=True)

    assert vectors.shape == (7, DEFAULT_VECTOR_DIM)


def test_chunk_vector_alignment_and_manifest():
    service = EmbeddingService(model_name=DEFAULT_MODEL_NAME)
    chunks = [create_sample_chunk(f"Academic subject content chunk {i}", index=i) for i in range(4)]

    vectors, manifest = service.embed_chunks(chunks, batch_size=2, normalize=True)

    # 1. 1-to-1 chunk-to-vector alignment
    assert vectors.shape[0] == len(chunks)
    assert manifest["total_chunks"] == len(chunks)
    assert manifest["total_vectors"] == len(chunks)
    assert manifest["vector_dimension"] == DEFAULT_VECTOR_DIM

    # 2. Chunk ID alignment
    assert manifest["chunk_ids"] == [c.chunk_id for c in chunks]


def test_empty_embedding_input():
    service = EmbeddingService(model_name=DEFAULT_MODEL_NAME)
    vectors = service.embed_texts([], batch_size=32)
    assert vectors.shape == (0, DEFAULT_VECTOR_DIM)


def test_embedding_reproducibility():
    service = EmbeddingService(model_name=DEFAULT_MODEL_NAME)
    text = "Attendance criteria requires 75% for exam eligibility."

    v1 = service.embed_texts([text], normalize=True)
    v2 = service.embed_texts([text], normalize=True)

    assert np.allclose(v1, v2, atol=1e-5)
