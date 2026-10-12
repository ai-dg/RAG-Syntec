"""
Qdrant vector store, a drop-in alternative to Chroma (VECTOR_STORE=qdrant).

Responsibilities:
- Build a Qdrant collection from the chunks: clear it, embed, upsert in batches.
- Answer similarity_search_with_score() with the same signature and return
  shape as langchain_chroma's, so retrieve() and the guardrail are unchanged.
- Reuse a persisted collection when it was built with the current settings,
  with the same index_info.json check as the Chroma index.

Scores:
retrieve() compares the best score with RELEVANCE_THRESHOLD, which was
calibrated on Chroma's squared-L2 distance (lower is closer). Qdrant never
returns that number: measured on unit vectors, Distance.EUCLID returns the
plain L2 distance and Distance.COSINE returns the cosine similarity (higher is
closer). The collection uses COSINE and every score goes through
to_squared_l2(), since for unit vectors ||a - b||^2 = 2 - 2 cos(a, b). The
embeddings returned by Ollama have norm 1 (design/embeddings.md).

Storage:
QdrantClient(path=...) runs Qdrant embedded in the process and persists to a
folder, like Chroma; no server is needed. Only one client may open a folder at
a time, so the app and an evaluation run cannot share one QDRANT_PATH
concurrently.
"""

import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.services.ingestion import (
    INDEX_INFO,
    chunk_text,
    get_embedding,
    index_info,
    load_docs,
    read_index_info,
)

logger = logging.getLogger(__name__)

COLLECTION = "syntec"
BATCH_SIZE = 256
TEXT_KEY = "page_content"


def to_squared_l2(cosine: float) -> float:
    """Cosine similarity to squared-L2 distance, exact for unit vectors."""
    return 2.0 - 2.0 * cosine


def to_qdrant_filter(search_filter: dict | None) -> models.Filter | None:
    """Translate the Chroma-style equality filter ({"in_force": True}) used by
    retrieve(). Qdrant applies it during the search, not after it, so k
    results come back whenever k matching chunks exist."""
    if not search_filter:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in search_filter.items()
        ]
    )


class QdrantStore:
    def __init__(self, client: QdrantClient, embedding):
        self.client = client
        self.embedding = embedding

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        """(chunk, squared-L2 distance) pairs, closest first."""
        points = self.client.query_points(
            COLLECTION,
            query=self.embedding.embed_query(query),
            limit=k,
            query_filter=to_qdrant_filter(filter),
            with_payload=True,
        ).points
        return [
            (
                Document(
                    page_content=point.payload[TEXT_KEY],
                    metadata={
                        key: value
                        for key, value in point.payload.items()
                        if key != TEXT_KEY
                    },
                ),
                to_squared_l2(point.score),
            )
            for point in points
        ]

    def count(self) -> int:
        return self.client.count(COLLECTION).count


def build_qdrant_store(chunks, embedding, client: QdrantClient) -> QdrantStore:
    """Clear the collection, then write every chunk: the same "clear before
    write" guarantee as create_vector_store(), so a rebuild never mixes chunks
    from two indexing runs. Point ids are the chunk positions."""
    if not chunks:
        raise ValueError("No chunks available for indexing")

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = embedding.embed_documents([chunk.page_content for chunk in batch])
        if start == 0:
            client.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]), distance=models.Distance.COSINE
                ),
            )
        client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(
                    id=start + offset,
                    vector=vector,
                    payload={**chunk.metadata, TEXT_KEY: chunk.page_content},
                )
                for offset, (chunk, vector) in enumerate(zip(batch, vectors))
            ],
        )

    return QdrantStore(client, embedding)


def create_qdrant_store(chunks, client: QdrantClient | None = None) -> QdrantStore:
    """Build the collection and record the settings it was built with."""
    settings = get_settings()
    client = client or QdrantClient(path=settings.qdrant_path)
    store = build_qdrant_store(chunks, get_embedding(settings), client)

    (Path(settings.qdrant_path) / INDEX_INFO).write_text(
        json.dumps(index_info(settings), indent=2)
    )
    logger.info("%d chunks stored in Qdrant at %s", len(chunks), settings.qdrant_path)
    return store


def open_qdrant_store(settings) -> QdrantStore:
    """Open the persisted collection without checking or rebuilding it."""
    return QdrantStore(QdrantClient(path=settings.qdrant_path), get_embedding(settings))


def open_or_build_qdrant_store(settings) -> QdrantStore:
    """Reuse the persisted collection if it was built with the current
    settings; otherwise (or if REINDEX_ON_STARTUP is set) rebuild it."""
    client = QdrantClient(path=settings.qdrant_path)
    if not settings.reindex_on_startup and read_index_info(
        settings.qdrant_path
    ) == index_info(settings):
        logger.info("reusing the persisted Qdrant index at %s", settings.qdrant_path)
        return QdrantStore(client, get_embedding(settings))
    logger.info("building the Qdrant index from %s", settings.docs_dir)
    return create_qdrant_store(chunk_text(load_docs()), client)
