"""
Document retrieval pipeline.

This module searches the vector store for the chunks that are most relevant to
the user's question. It also applies a relevance-based guardrail before the
chunks are sent to the generation pipeline.

Responsibilities:
- Receive the user's question.
- Generate an embedding for the question.
- Search Chroma for the nearest document vectors.
- Retrieve the most relevant chunks, together with their scores and metadata.
- Compare the best retrieval score with a configurable relevance threshold.
- Reject the retrieved context when no result is found or when its relevance
  score is below the threshold.

Input:
- The user's question.

Output:
- The relevant chunks, scores, and metadata when sufficient context is found.
- An empty result when the available documents are not relevant enough.

Guardrail:
The generation pipeline should only be called when the retrieval result passes
the relevance threshold. Otherwise, the API returns a controlled "I don't know"
response without calling the LLM.

Important:
Depending on the Chroma method used, the returned value may be a relevance
score, where a higher value is better, or a distance, where a lower value is
better. The threshold comparison must follow the returned score type.
"""

import logging

from app.services.abstention import load_model
from app.services.features import FEATURE_K, extract_features
from langchain_chroma import Chroma

from app.services.ingestion import (
    chunk_text,
    create_vector_store,
    get_embedding,
    index_info,
    load_docs,
    read_index_info,
)
from app.config import get_settings

from app.services.reranking import rerank

logger = logging.getLogger(__name__)

_vector_store = None


def _refused(best_score, reason, confidence=None, mode="threshold"):
    return {
        "chunks": [],
        "context_found": False,
        "best_score": best_score,
        "refusal_reason": reason,
        "confidence": confidence,
        "guardrail_mode": mode,
    }


def retrieve(question):
    """Search the index, decide whether to answer, and return the chunks for the prompt.

    Scores are squared-L2 distances (lower is closer). The answer-or-refuse decision
    is either the distance threshold on the best chunk, or the abstention
    classifier when GUARDRAIL_MODE=classifier and its model file can be loaded
    (otherwise the threshold is used and the fallback is logged).
    """
    settings = get_settings()
    store = get_vector_store(settings)

    model = (
        load_model(settings.abstention_model_path)
        if settings.guardrail_mode == "classifier"
        else None
    )
    mode = "classifier" if model else "threshold"
    if settings.guardrail_mode == "classifier" and model is None:
        logger.warning(
            "guardrail_mode=classifier but no model: using the distance threshold"
        )

    k = settings.rerank_candidates if settings.rerank_enabled else settings.top_k
    search_k = max(k, FEATURE_K) if model else k
    search_filter = None if settings.include_superseded else {"in_force": True}
    found = store.similarity_search_with_score(
        question, k=search_k, filter=search_filter
    )
    if not found:
        return _refused(None, "no_results", mode=mode)

    results = found[:k]
    best_score = results[0][1]
    confidence = None

    if model:
        answer, confidence = model.should_answer(
            extract_features([score for _, score in found], question)
        )
        if not answer:
            return _refused(best_score, "classifier_refused", confidence, mode)
        best_results = list(results)
    else:
        if best_score > settings.relevance_threshold:
            return _refused(best_score, "below_relevance_threshold", mode=mode)
        best_results = [r for r in results if r[1] <= settings.relevance_threshold]

    if settings.rerank_enabled:
        best_results = rerank(question, best_results, settings.top_k)

    return {
        "chunks": best_results,
        "context_found": True,
        "best_score": best_score,
        "refusal_reason": None,
        "confidence": confidence,
        "guardrail_mode": mode,
    }


def open_or_build_vector_store(settings):
    """Reuse the persisted index if it was built with the current settings;
    otherwise (or if REINDEX_ON_STARTUP is set) rebuild it from the documents."""
    if not settings.reindex_on_startup and read_index_info(
        settings.chroma_dir
    ) == index_info(settings):
        logger.info("reusing the persisted index at %s", settings.chroma_dir)
        return Chroma(
            persist_directory=settings.chroma_dir,
            embedding_function=get_embedding(settings),
        )
    logger.info("building the index from %s", settings.docs_dir)
    return create_vector_store(chunk_text(load_docs()))


def get_vector_store(settings=None):
    global _vector_store

    if _vector_store is None:
        _vector_store = open_or_build_vector_store(settings or get_settings())

    return _vector_store


def set_vector_store(store):
    global _vector_store
    _vector_store = store
