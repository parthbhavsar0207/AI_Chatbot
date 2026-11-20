"""ChromaDB persistent vector store for the LPU CSE-AIML RAG corpus.

Provides idempotent batch indexing of pre-embedded chunks and collection
management with cosine distance.  Reuses precomputed embeddings from
Phase 4 — no re-embedding is done during indexing.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from app.rag.schema import Chunk

logger = logging.getLogger(__name__)

# Hard-coded expected dimension from Phase 4 manifest
EXPECTED_VECTOR_DIM = 384


class VectorStoreService:
    """Manages a ChromaDB persistent collection for academic document chunks."""

    def __init__(
        self,
        persist_directory: str | Path = "data/processed/chroma",
        collection_name: str = "lpu_cse_aiml_docs",
    ):
        self.persist_directory = str(Path(persist_directory).resolve())
        self.collection_name = collection_name
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy-initialised persistent ChromaDB client."""
        if self._client is None:
            logger.info(
                "Initialising persistent ChromaDB client at: %s",
                self.persist_directory,
            )
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get-or-create the cosine-distance collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Collection '%s' ready (%d records).",
                self.collection_name,
                self._collection.count(),
            )
        return self._collection

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 64,
    ) -> dict:
        """Upserts pre-embedded chunks into ChromaDB in batches.

        Uses ``upsert`` so repeated runs are idempotent — identical records
        are overwritten rather than duplicated.

        Args:
            chunks: List of Chunk objects (must be same length as embeddings).
            embeddings: (N, 384) float32 numpy matrix from Phase 4.
            batch_size: Records per upsert batch.

        Returns:
            Indexing summary dict.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding row count "
                f"({embeddings.shape[0]}). They must match 1-to-1."
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_VECTOR_DIM:
            raise ValueError(
                f"Expected embedding dimension {EXPECTED_VECTOR_DIM}, "
                f"got shape {embeddings.shape}."
            )

        total = len(chunks)
        indexed = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]
            batch_vectors = embeddings[start:end]

            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.text for c in batch_chunks]
            metadatas = [self._chunk_metadata_to_chroma(c) for c in batch_chunks]
            embeds = batch_vectors.tolist()

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeds,
            )
            indexed += len(ids)
            logger.info("Indexed batch %d–%d / %d", start + 1, end, total)

        count_after = self.collection.count()
        summary = {
            "collection_name": self.collection_name,
            "total_upserted": indexed,
            "collection_count_after": count_after,
            "persist_directory": self.persist_directory,
        }
        logger.info("Indexing complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """Queries the collection by embedding vector.

        Args:
            query_embedding: 384-d float list for the query.
            top_k: Number of nearest neighbours.
            where: Optional ChromaDB ``where`` filter dict.

        Returns:
            Raw ChromaDB query result dict with keys
            ``ids``, ``documents``, ``metadatas``, ``distances``.
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def count(self) -> int:
        """Returns the number of records in the collection."""
        return self.collection.count()

    def reset_collection(self) -> None:
        """Drops and recreates the collection (useful for tests)."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        _ = self.collection  # recreate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_metadata_to_chroma(chunk: Chunk) -> dict:
        """Converts ChunkMetadata into a flat dict acceptable to ChromaDB.

        ChromaDB metadata values must be str, int, float, or bool — no
        lists, nested dicts, or None.  This method flattens/serialises
        accordingly and drops keys whose values are None.
        """
        m = chunk.metadata
        meta: dict = {
            "document_id": chunk.document_id,
            "source_file": m.source_file,
            "category": m.category,
            "file_type": m.file_type,
            "title": m.title,
            "token_count": m.token_count,
            "chunk_index": m.chunk_index,
            "total_chunks": m.total_chunks,
        }

        # Optional scalar fields — include only if non-None
        if m.course_code is not None:
            meta["course_code"] = m.course_code
        if m.page_number is not None:
            meta["page_number"] = m.page_number
        if m.applicable_scope is not None:
            meta["applicable_scope"] = m.applicable_scope
        if m.verification_status is not None:
            meta["verification_status"] = m.verification_status

        # Lists → JSON strings (ChromaDB does not support list values)
        if m.headings:
            meta["headings"] = json.dumps(m.headings)

        return meta
