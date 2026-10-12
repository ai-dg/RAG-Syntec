"""
Tests for the Qdrant vector store (app.services.vectorstore).

They run a real embedded Qdrant in a temporary folder, with a fake embedding
that maps each text to a fixed unit vector, so no model or network is needed.
The squared-L2 distances Chroma would return are computed with numpy and
compared with what the Qdrant store returns: that is the contract retrieve()
and the 0.74 threshold depend on.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from app.config import get_settings
from app.services import vectorstore
from app.services.ingestion import INDEX_INFO, index_info
from app.services.retrieval import retrieve, set_vector_store
from app.services.vectorstore import (
    QdrantStore,
    build_qdrant_store,
    open_or_build_qdrant_store,
    to_squared_l2,
)


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


class FakeEmbedding:
    """Each known text maps to a fixed unit vector; counts document calls."""

    def __init__(self, vectors):
        self.vectors = {text: unit(v) for text, v in vectors.items()}
        self.documents_embedded = 0

    def embed_documents(self, texts):
        self.documents_embedded += len(texts)
        return [self.vectors[text].tolist() for text in texts]

    def embed_query(self, text):
        return self.vectors[text].tolist()


VECTORS = {
    "near": [1.0, 0.1, 0.0],
    "middle": [1.0, 1.0, 0.0],
    "far": [0.0, 0.0, 1.0],
    "superseded": [1.0, 0.05, 0.0],
    "question": [1.0, 0.0, 0.0],
}


def chunk(text, in_force=True):
    return Document(
        page_content=text,
        metadata={
            "source": f"{text}.md",
            "chunk_id": f"{text}.md#chunk_0",
            "in_force": in_force,
        },
    )


CHUNKS = [
    chunk("near"),
    chunk("middle"),
    chunk("far"),
    chunk("superseded", in_force=False),
]


@pytest.fixture
def client(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    yield client
    client.close()


@pytest.fixture
def store(client):
    return build_qdrant_store(CHUNKS, FakeEmbedding(VECTORS), client)


def test_to_squared_l2_matches_the_distance_chroma_returns():
    rng = np.random.default_rng(0)
    for _ in range(100):
        a, b = unit(rng.standard_normal(16)), unit(rng.standard_normal(16))
        assert to_squared_l2(a @ b) == pytest.approx(np.sum((a - b) ** 2))


def test_scores_are_squared_l2_distances_closest_first(store):
    found = store.similarity_search_with_score("question", k=4)

    texts = [document.page_content for document, _ in found]
    assert texts == ["superseded", "near", "middle", "far"]
    question = unit(VECTORS["question"])
    for document, score in found:
        expected = np.sum((question - unit(VECTORS[document.page_content])) ** 2)
        assert score == pytest.approx(expected, abs=1e-5)


def test_metadata_comes_back_as_it_was_stored(store):
    document, _ = store.similarity_search_with_score("question", k=1)[0]

    assert document.metadata == {
        "source": "superseded.md",
        "chunk_id": "superseded.md#chunk_0",
        "in_force": False,
    }


def test_the_filter_applies_before_the_limit(store):
    found = store.similarity_search_with_score(
        "question", k=2, filter={"in_force": True}
    )

    assert [document.page_content for document, _ in found] == ["near", "middle"]


def test_rebuilding_clears_the_previous_chunks(client):
    build_qdrant_store(CHUNKS, FakeEmbedding(VECTORS), client)
    store = build_qdrant_store(CHUNKS[:2], FakeEmbedding(VECTORS), client)

    assert store.count() == 2


def test_building_nothing_is_refused(client):
    with pytest.raises(ValueError):
        build_qdrant_store([], FakeEmbedding(VECTORS), client)


def test_retrieve_applies_the_threshold_to_the_converted_distance(store):
    # RELEVANCE_THRESHOLD is 0.9 in the test settings. "near" is at squared-L2
    # 0.00995 and "middle" at 0.586: both pass. "far" is at 2.0: it does not.
    set_vector_store(store)
    try:
        result = retrieve("question")
    finally:
        set_vector_store(None)

    assert result["context_found"] is True
    assert [document.page_content for document, _ in result["chunks"]] == [
        "near",
        "middle",
    ]
    assert result["best_score"] == pytest.approx(0.00995, abs=1e-4)


@pytest.fixture
def qdrant_settings(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    get_settings.cache_clear()
    embedding = FakeEmbedding(VECTORS)
    monkeypatch.setattr(vectorstore, "get_embedding", lambda settings: embedding)
    monkeypatch.setattr(vectorstore, "load_docs", lambda: [])
    monkeypatch.setattr(vectorstore, "chunk_text", lambda docs: CHUNKS)
    return embedding


def test_a_missing_index_is_built_and_its_settings_recorded(qdrant_settings):
    settings = get_settings()
    store = open_or_build_qdrant_store(settings)
    store.client.close()

    assert qdrant_settings.documents_embedded == len(CHUNKS)
    info = json.loads((Path(settings.qdrant_path) / INDEX_INFO).read_text())
    assert info == index_info(settings)


def test_an_index_built_with_the_same_settings_is_reused(qdrant_settings):
    settings = get_settings()
    open_or_build_qdrant_store(settings).client.close()
    embedded_by_first_build = qdrant_settings.documents_embedded

    store = open_or_build_qdrant_store(settings)

    assert qdrant_settings.documents_embedded == embedded_by_first_build
    assert isinstance(store, QdrantStore)
    assert store.count() == len(CHUNKS)
    store.client.close()


def test_rebuilding_can_be_forced(qdrant_settings, monkeypatch):
    settings = get_settings()
    open_or_build_qdrant_store(settings).client.close()
    monkeypatch.setenv("REINDEX_ON_STARTUP", "true")
    get_settings.cache_clear()

    open_or_build_qdrant_store(get_settings()).client.close()

    assert qdrant_settings.documents_embedded == 2 * len(CHUNKS)
