"""
Tests for the relevance guardrail in app.services.retrieval.

The vector store is injected via `set_vector_store`, never patched by path —
that keeps the tests decoupled from retrieval's internals and immune to any
future refactor of how the store is obtained.
"""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.services.retrieval import retrieve, set_vector_store

IN_TOPIC = Document(page_content="in-topic content", metadata={"source": "a.md"})
OFF_TOPIC = Document(page_content="off-topic content", metadata={"source": "b.md"})


def fake_store(results):
    store = MagicMock()
    store.similarity_search_with_score.return_value = results
    return store


def test_guardrail_rejects_when_no_result_is_within_threshold():
    set_vector_store(fake_store([(OFF_TOPIC, 1.5)]))

    result = retrieve("an off-topic question")

    assert result["context_found"] is False
    assert result["chunks"] == []
    assert result["best_score"] == 1.5  # kept for observability, even on reject


def test_guardrail_accepts_a_result_within_threshold():
    set_vector_store(fake_store([(IN_TOPIC, 0.3)]))

    result = retrieve("an in-topic question")

    assert result["context_found"] is True
    assert len(result["chunks"]) == 1


def test_guardrail_filters_each_result_independently():
    # Regression test: the guardrail must check every candidate, not only
    # the best-scoring one. A previous version returned all `top_k` results
    # once the first passed, silently feeding poor context to the LLM.
    set_vector_store(fake_store([(IN_TOPIC, 0.3), (OFF_TOPIC, 1.4), (OFF_TOPIC, 1.6)]))

    result = retrieve("a mixed-relevance question")

    assert result["context_found"] is True
    assert len(result["chunks"]) == 1
    assert result["chunks"][0][0] is IN_TOPIC


def test_guardrail_handles_no_search_results():
    set_vector_store(fake_store([]))

    result = retrieve("any question")

    assert result["context_found"] is False
    assert result["chunks"] == []
    assert result["best_score"] is None
