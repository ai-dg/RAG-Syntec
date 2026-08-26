"""
Document retrieval pipeline.

This module searches the vector store for the chunks most relevant to the
user's question, then applies a relevance guardrail before any chunk is sent
to generation.

Important — what the score actually is:
Chroma's `similarity_search_with_score` returns a DISTANCE by default (squared
L2), not a similarity: LOWER is closer. With normalized embeddings this is
equal to `2 - 2*cosine_similarity`, so it tracks cosine similarity but on an
inverted, non-linear scale. Comparing it to `relevance_threshold` must use
"reject if score > threshold", never the other way around. See
`scripts/inspect_metric.py` to verify this against a live collection.

The vector store is built once at process startup (see app/main.py's
lifespan handler) and cached here. It is intentionally NOT built at import
time: doing so would make every import of this module trigger a full
re-index, which broke test isolation and startup observability in an earlier
version of this project.
"""

import threading

from app.config import Settings
from app.services.ingestion import build_vector_store

_vector_store = None
_lock = threading.Lock()


def get_vector_store(settings: Settings | None = None):
    """Return the process-wide vector store, building it on first use."""
    global _vector_store
    if _vector_store is None:
        with _lock:
            if _vector_store is None:
                _vector_store = build_vector_store(settings or Settings())
    return _vector_store


def set_vector_store(store) -> None:
    """Inject a vector store directly. Used by tests and by app startup."""
    global _vector_store
    _vector_store = store


def retrieve(question: str) -> dict:
    settings = Settings()
    store = get_vector_store(settings)

    results = store.similarity_search_with_score(question, k=settings.top_k)

    if not results:
        return {"chunks": [], "context_found": False, "best_score": None}

    best_score = results[0][1]
    accepted = [
        (document, score)
        for document, score in results
        if score <= settings.relevance_threshold
    ]

    if not accepted:
        return {"chunks": [], "context_found": False, "best_score": best_score}

    return {"chunks": accepted, "context_found": True, "best_score": best_score}
