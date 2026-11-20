"""FastEmbed embedding generation service for academic RAG corpus."""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional
from fastembed import TextEmbedding
import numpy as np

from app.rag.schema import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_VECTOR_DIM = 384


class EmbeddingService:
    """Generates normalized vector embeddings using local FastEmbed on CPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._model: Optional[TextEmbedding] = None

    @property
    def model(self) -> TextEmbedding:
        """Lazy-loaded FastEmbed model instance running on CPU."""
        if self._model is None:
            logger.info(f"Initializing FastEmbed TextEmbedding model: {self.model_name} (CPU)")
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
            )
        return self._model

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embeds a list of texts in batches using CPU ONNX runtime.

        Returns:
            np.ndarray: float32 matrix of shape (len(texts), vector_dimension)
        """
        if not texts:
            return np.empty((0, DEFAULT_VECTOR_DIM), dtype=np.float32)

        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            generator = self.model.embed(batch)
            for vec in generator:
                if normalize:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                vectors.append(vec.astype(np.float32))

        matrix = np.vstack(vectors).astype(np.float32)
        return matrix

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Embeds a list of Chunk objects and produces an accompanying manifest."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size, normalize=normalize)

        manifest = {
            "model_name": self.model_name,
            "vector_dimension": int(embeddings.shape[1]) if len(embeddings) > 0 else DEFAULT_VECTOR_DIM,
            "total_chunks": len(chunks),
            "total_vectors": int(embeddings.shape[0]),
            "normalized": normalize,
            "batch_size": batch_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_ids": [c.chunk_id for c in chunks],
        }

        return embeddings, manifest
