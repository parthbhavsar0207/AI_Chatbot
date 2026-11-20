"""Modular cross-encoder reranking service for academic RAG corpus.

Uses the lightweight cross-encoder/ms-marco-MiniLM-L-6-v2 model via ONNX Runtime
for fast, CPU-only inference preserving GPU memory for future generation models.
Preserves original chunk identifiers, metadata, and similarity distances.
"""

import logging
from pathlib import Path
import time
from typing import Optional
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app.rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
DEFAULT_MAX_LENGTH = 512


class CrossEncoderReranker:
    """CPU-optimized Cross-Encoder Reranker using ONNX Runtime and Tokenizers."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model_revision: str = DEFAULT_MODEL_REVISION,
        max_length: int = DEFAULT_MAX_LENGTH,
        cache_dir: Optional[str | Path] = None,
        threads: Optional[int] = None,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.max_length = max_length
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.threads = threads

        self._session: Optional[ort.InferenceSession] = None
        self._tokenizer: Optional[Tokenizer] = None

    @property
    def tokenizer(self) -> Tokenizer:
        """Lazy-loaded HuggingFace fast Tokenizer."""
        if self._tokenizer is None:
            logger.info("Loading cross-encoder tokenizer: %s (revision: %s)", self.model_name, self.model_revision)
            tok_path = hf_hub_download(
                repo_id=self.model_name,
                filename="tokenizer.json",
                revision=self.model_revision,
                cache_dir=self.cache_dir,
            )
            tok = Tokenizer.from_file(tok_path)
            tok.enable_truncation(max_length=self.max_length)
            tok.enable_padding(pad_id=0, pad_token="[PAD]")
            self._tokenizer = tok
        return self._tokenizer

    @property
    def session(self) -> ort.InferenceSession:
        """Lazy-loaded ONNX Runtime InferenceSession on CPU."""
        if self._session is None:
            logger.info("Loading cross-encoder ONNX model: %s (CPU)", self.model_name)
            onnx_path = hf_hub_download(
                repo_id=self.model_name,
                filename="onnx/model.onnx",
                revision=self.model_revision,
                cache_dir=self.cache_dir,
            )
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if self.threads:
                opts.intra_op_num_threads = self.threads

            self._session = ort.InferenceSession(
                onnx_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("Cross-encoder ONNX session ready.")
        return self._session

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        """Scores a list of document strings against a single query string.

        Args:
            query: Natural language query string.
            documents: List of document candidate strings.

        Returns:
            list[float]: Raw cross-encoder relevance logits.
        """
        if not documents:
            return []
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        clean_query = query.strip()
        pairs = [(clean_query, doc) for doc in documents]

        encodings = self.tokenizer.encode_batch(pairs)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        logits = self.session.run(None, ort_inputs)[0]
        # Squeeze output of shape (batch_size, 1) to (batch_size,)
        scores = logits.squeeze(-1).tolist()
        if isinstance(scores, float):
            scores = [scores]
        return [round(float(s), 6) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Reranks retrieved candidate chunks based on cross-encoder relevance scoring.

        Args:
            query: Original user query string.
            candidates: Candidate chunks from initial vector retrieval.
            top_k: Optional final number of reranked chunks to return.

        Returns:
            list[RetrievedChunk]: Chunks sorted by reranker score descending, with
                reranker_score, original_rank, and reranked_rank populated.
        """
        if not candidates:
            return []

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        if top_k is not None:
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(f"top_k must be a positive integer, got: {top_k}")

        clean_query = query.strip()
        doc_texts = [c.text for c in candidates]
        scores = self.score_pairs(clean_query, doc_texts)

        # Build list of enriched chunks with original rank preserved
        ranked_candidates: list[RetrievedChunk] = []
        for orig_idx, (chunk, score) in enumerate(zip(candidates, scores)):
            # Create a copy with reranking metadata
            chunk_copy = chunk.model_copy()
            chunk_copy.original_rank = orig_idx + 1 if chunk_copy.original_rank is None else chunk_copy.original_rank
            chunk_copy.reranker_score = score
            ranked_candidates.append(chunk_copy)

        # Sort descending by cross-encoder score
        ranked_candidates.sort(key=lambda x: x.reranker_score if x.reranker_score is not None else -float("inf"), reverse=True)

        # Assign reranked_rank (1-indexed)
        for new_idx, chunk in enumerate(ranked_candidates):
            chunk.reranked_rank = new_idx + 1

        effective_top_k = top_k if top_k is not None else len(ranked_candidates)
        return ranked_candidates[:effective_top_k]
