"""Unit and integration tests for ChromaDB VectorStoreService and RetrievalService."""

import gc
from pathlib import Path
import tempfile
import numpy as np
import pytest

from app.rag.embeddings import EmbeddingService
from app.rag.retrieval import RetrievalResponse, RetrievalService, RetrievedChunk
from app.rag.schema import Chunk, ChunkMetadata
from app.rag.vector_store import EXPECTED_VECTOR_DIM, VectorStoreService


@pytest.fixture
def temp_chroma_dir():
    """Provides an isolated temporary directory for ChromaDB tests."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir)
        gc.collect()


@pytest.fixture
def sample_chunks_and_embeddings():
    """Generates a small controlled set of sample chunks and normalized vectors."""
    meta1 = ChunkMetadata(
        source_file="academic_regulations.docx",
        category="regulations",
        file_type="docx",
        title="Attendance Requirements",
        applicable_scope="All UG Programmes",
        token_count=28,
        chunk_index=0,
        total_chunks=1,
        headings=["Chapter 2", "Attendance"],
    )
    c1 = Chunk.create(
        document_id="doc_reg_01",
        text="A student must maintain a minimum attendance of 75 percent to appear in end-term examinations.",
        metadata=meta1,
    )

    meta2 = ChunkMetadata(
        source_file="CSE306_syllabus.pdf",
        category="syllabus",
        file_type="pdf",
        title="Operating Systems",
        course_code="CSE306",
        page_number=1,
        token_count=35,
        chunk_index=0,
        total_chunks=2,
        headings=["Course Outcomes", "Unit 1"],
    )
    c2 = Chunk.create(
        document_id="doc_syl_01",
        text="[Course: CSE306 - Operating Systems (Page 1)]\nUnit 1 covers Process Management and CPU Scheduling.",
        metadata=meta2,
    )

    meta3 = ChunkMetadata(
        source_file="faqs.json",
        category="faqs",
        file_type="json",
        title="FAQ: Re-evaluation",
        token_count=22,
        chunk_index=0,
        total_chunks=1,
    )
    c3 = Chunk.create(
        document_id="doc_faq_01",
        text="Q: How do I apply for re-evaluation?\nA: Students can apply through the UMS portal within 7 days.",
        metadata=meta3,
    )

    # 3 distinct normalized float32 vectors
    rng = np.random.RandomState(42)
    raw_vecs = rng.randn(3, EXPECTED_VECTOR_DIM).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norm_vecs = raw_vecs / norms

    return [c1, c2, c3], norm_vecs


# ==============================================================================
# VectorStoreService Unit Tests (Isolated)
# ==============================================================================


def test_collection_creation_and_cosine_space(temp_chroma_dir):
    """Verifies collection creation with cosine distance configuration."""
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    col = vs.collection
    assert col.name == "test_col"
    assert col.metadata.get("hnsw:space") == "cosine"
    assert vs.count() == 0


def test_batch_indexing_and_dimension_validation(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies batch upsert and embedding dimension guardrails."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")

    summary = vs.index_chunks(chunks, embeddings, batch_size=2)
    assert summary["total_upserted"] == 3
    assert vs.count() == 3

    # Reject mismatched count
    with pytest.raises(ValueError, match="Chunk count.*!= embedding row count"):
        vs.index_chunks(chunks[:1], embeddings)

    # Reject mismatched dimension
    wrong_dim = np.ones((len(chunks), 128), dtype=np.float32)
    with pytest.raises(ValueError, match=f"Expected embedding dimension {EXPECTED_VECTOR_DIM}"):
        vs.index_chunks(chunks, wrong_dim)


def test_idempotent_indexing_does_not_duplicate(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies that re-indexing identical chunks does not create duplicate entries."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")

    vs.index_chunks(chunks, embeddings)
    assert vs.count() == 3

    # Run again on the exact same records
    summary2 = vs.index_chunks(chunks, embeddings)
    assert summary2["total_upserted"] == 3
    assert vs.count() == 3  # Count unchanged


def test_reopening_persistent_collection(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies that data persists on disk and is reloaded upon client reconnection."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs1 = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    vs1.index_chunks(chunks, embeddings)
    assert vs1.count() == 3
    del vs1
    gc.collect()

    vs2 = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    assert vs2.count() == 3
    raw = vs2.query(embeddings[0].tolist(), top_k=1)
    assert raw["ids"][0][0] == chunks[0].chunk_id


# ==============================================================================
# RetrievalService Unit Tests
# ==============================================================================


def test_retrieval_empty_query_raises_value_error(temp_chroma_dir):
    """Verifies that empty or whitespace-only queries are rejected with ValueError."""
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    retriever = RetrievalService(vector_store=vs)

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.retrieve("   \t\n  ")


def test_retrieval_invalid_top_k_raises_value_error(temp_chroma_dir):
    """Verifies that non-positive top_k values are rejected."""
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    retriever = RetrievalService(vector_store=vs)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("attendance", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("attendance", top_k=-5)


def test_semantic_retrieval_and_metadata_preservation(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies semantic similarity query, ranking, and metadata decoding."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    vs.index_chunks(chunks, embeddings)

    retriever = RetrievalService(vector_store=vs)
    response = retriever.retrieve_with_response("What is the minimum attendance requirement?", top_k=2)

    assert isinstance(response, RetrievalResponse)
    assert response.total_retrieved == 2
    assert response.retrieval_time_seconds > 0.0

    top_chunk = response.results[0]
    assert isinstance(top_chunk, RetrievedChunk)
    assert top_chunk.chunk_id
    assert top_chunk.text
    assert top_chunk.metadata["category"]
    assert top_chunk.metadata["source_file"]
    # Check that distance and similarity are within valid bounds
    assert 0.0 <= top_chunk.distance <= 2.0
    assert -1.0 <= top_chunk.similarity <= 1.0


def test_retrieval_category_filter(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies category filtering in retrieval."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    vs.index_chunks(chunks, embeddings)

    retriever = RetrievalService(vector_store=vs)
    results = retriever.retrieve("syllabus and rules", category="syllabus", top_k=5)

    assert len(results) > 0
    for r in results:
        assert r.metadata["category"] == "syllabus"


def test_retrieval_course_code_filter(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies course code filtering in retrieval."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    vs.index_chunks(chunks, embeddings)

    retriever = RetrievalService(vector_store=vs)
    results = retriever.retrieve("Operating systems", course_code="CSE306", top_k=5)

    assert len(results) == 1
    assert results[0].metadata["course_code"] == "CSE306"


def test_retrieval_source_file_filter(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies source_file filtering in retrieval."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    vs.index_chunks(chunks, embeddings)

    retriever = RetrievalService(vector_store=vs)
    results = retriever.retrieve("regulations and syllabus", source_file="academic_regulations.docx", top_k=5)

    assert len(results) == 1
    assert results[0].metadata["source_file"] == "academic_regulations.docx"


def test_retrieval_combined_filters(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies combined filters (category + course_code)."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    vs.index_chunks(chunks, embeddings)

    retriever = RetrievalService(vector_store=vs)
    results = retriever.retrieve("Process management", category="syllabus", course_code="CSE306", top_k=5)

    assert len(results) == 1
    assert results[0].metadata["category"] == "syllabus"
    assert results[0].metadata["course_code"] == "CSE306"


def test_retrieval_invalid_filter_values_return_empty_safely(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies that non-existent filter values return empty results without errors."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    vs.index_chunks(chunks, embeddings)

    retriever = RetrievalService(vector_store=vs)

    # Non-existent course code
    res_course = retriever.retrieve("syllabus", course_code="XYZ999", top_k=5)
    assert res_course == []

    # Non-existent category
    res_cat = retriever.retrieve("syllabus", category="astrophysics", top_k=5)
    assert res_cat == []

    # Non-existent source file
    res_src = retriever.retrieve("syllabus", source_file="non_existent.pdf", top_k=5)
    assert res_src == []


def test_retrieval_invalid_filter_types_raise_value_error(temp_chroma_dir):
    """Verifies that non-string filter arguments raise ValueError."""
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    retriever = RetrievalService(vector_store=vs)

    with pytest.raises(ValueError, match="category filter must be a string"):
        retriever.retrieve("query", category=123)

    with pytest.raises(ValueError, match="course_code filter must be a string"):
        retriever.retrieve("query", course_code=["CSE306"])

    with pytest.raises(ValueError, match="source_file filter must be a string"):
        retriever.retrieve("query", source_file={"file": "test.pdf"})


# ==============================================================================
# Metric Calculation Unit Tests
# ==============================================================================


from app.rag.evaluator import (
    compute_hit_rate,
    compute_precision,
    compute_recall,
    compute_reciprocal_rank,
    RetrievalEvaluator,
    EvaluationReport,
)


def test_metric_calculations_pure_functions():
    """Unit tests for Hit Rate@3, Recall@5, Precision@5, and MRR calculations."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    expected = ["c2", "c6"]

    # Hit Rate @ 3: c2 is at rank 2 -> Hit = 1.0
    assert compute_hit_rate(retrieved, expected, k=3) == 1.0

    # Hit Rate @ 1: c1 is at rank 1 (not in expected) -> Hit = 0.0
    assert compute_hit_rate(retrieved, expected, k=1) == 0.0

    # Recall @ 5: 1 hit out of 2 expected -> 1/2 = 0.5
    assert compute_recall(retrieved, expected, k=5) == 0.5

    # Precision @ 5: 1 hit in 5 retrieved -> 1/5 = 0.2
    assert compute_precision(retrieved, expected, k=5) == 0.2

    # MRR: first relevant is c2 at rank 2 -> 1/2 = 0.5
    assert compute_reciprocal_rank(retrieved, expected, max_k=10) == 0.5

    # Edge case: empty expected
    assert compute_hit_rate(retrieved, [], k=3) == 0.0
    assert compute_recall(retrieved, [], k=5) == 0.0
    assert compute_precision(retrieved, [], k=5) == 0.0
    assert compute_reciprocal_rank(retrieved, [], max_k=10) == 0.0

    # Edge case: no hits
    assert compute_hit_rate(["x", "y"], ["z"], k=3) == 0.0
    assert compute_reciprocal_rank(["x", "y"], ["z"], max_k=10) == 0.0


def test_evaluator_class_run_evaluation(temp_chroma_dir, sample_chunks_and_embeddings):
    """Verifies that RetrievalEvaluator generates valid evaluation reports."""
    chunks, embeddings = sample_chunks_and_embeddings
    vs = VectorStoreService(persist_directory=temp_chroma_dir, collection_name="test_col")
    vs.index_chunks(chunks, embeddings)

    retriever = RetrievalService(vector_store=vs)
    evaluator = RetrievalEvaluator(retrieval_service=retriever)

    test_queries = [
        {
            "query_id": "T01",
            "query": "Attendance requirement",
            "domain": "regulations",
            "course_code": None,
            "is_supported": True,
            "expected_chunk_ids": [chunks[0].chunk_id],
        },
        {
            "query_id": "T02",
            "query": "Unsupported question about sports scholarship",
            "domain": "unsupported",
            "course_code": None,
            "is_supported": False,
            "expected_chunk_ids": [],
        },
    ]

    report = evaluator.evaluate_queries(test_queries, mode="unfiltered", top_k=2)
    assert isinstance(report, EvaluationReport)
    assert report.overall_metrics.total_queries == 2
    assert report.overall_metrics.supported_queries == 1
    assert report.overall_metrics.unsupported_queries == 1
    assert report.overall_metrics.mean_hit_rate_at_3 == 1.0


# ==============================================================================
# Production Database Integration Tests
# ==============================================================================

PROD_CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "chroma"


def test_production_chroma_collection_has_65_records():
    """Verifies that the production ChromaDB collection exists with all 65 chunks."""
    if not PROD_CHROMA_DIR.exists():
        pytest.skip(f"Production ChromaDB collection not found at {PROD_CHROMA_DIR}")

    vs = VectorStoreService(persist_directory=PROD_CHROMA_DIR, collection_name="lpu_cse_aiml_docs")
    assert vs.count() == 65


def test_production_retrieval_representative_queries():
    """Tests representative queries across categories on the 65-record production collection."""
    if not PROD_CHROMA_DIR.exists():
        pytest.skip(f"Production ChromaDB collection not found at {PROD_CHROMA_DIR}")

    retriever = RetrievalService(
        vector_store=VectorStoreService(persist_directory=PROD_CHROMA_DIR, collection_name="lpu_cse_aiml_docs")
    )

    # 1. Syllabus query
    res_syl = retriever.retrieve("What are the units in CSE306 Operating Systems?", top_k=3)
    assert len(res_syl) > 0
    syl_categories = {r.metadata.get("category") for r in res_syl}
    assert any("syllabus" in c.lower() for c in syl_categories)
    assert any("CSE306" in r.text or r.metadata.get("course_code") == "CSE306" for r in res_syl)

    # 2. Regulation attendance query
    res_att = retriever.retrieve("What is the attendance requirement at LPU?", top_k=3)
    assert len(res_att) > 0
    assert any("75" in r.text or "attendance" in r.text.lower() for r in res_att)

    # 3. FAQ reappear query
    res_faq = retriever.retrieve("What is the reappear examination policy?", top_k=3)
    assert len(res_faq) > 0
    assert any("reappear" in r.text.lower() or "backlog" in r.text.lower() for r in res_faq)

    # 4. Examination schedule query
    res_exam = retriever.retrieve("When is the INT335 examination scheduled?", top_k=3)
    assert len(res_exam) > 0
    assert any("INT335" in r.text or "notices" in r.metadata.get("category", "").lower() for r in res_exam)
