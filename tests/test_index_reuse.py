import json

import pytest

from app.config import get_settings
from app.services import retrieval
from app.services.ingestion import INDEX_INFO, index_info, read_index_info


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(retrieval, "Chroma", lambda **kwargs: seen.append("open") or "opened")
    monkeypatch.setattr(retrieval, "load_docs", lambda: [])
    monkeypatch.setattr(retrieval, "chunk_text", lambda docs: [])
    monkeypatch.setattr(retrieval, "create_vector_store", lambda chunks: seen.append("build") or "built")
    monkeypatch.setattr(retrieval, "get_embedding", lambda settings: None)
    return seen


def write_info(info):
    path = get_settings().chroma_dir
    import pathlib

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(path) / INDEX_INFO).write_text(json.dumps(info))


def test_an_index_built_with_the_same_settings_is_reused(calls):
    write_info(index_info(get_settings()))

    assert retrieval.open_or_build_vector_store(get_settings()) == "opened"
    assert calls == ["open"]


def test_an_index_built_with_other_settings_is_rebuilt(calls):
    info = index_info(get_settings())
    write_info({**info, "chunking": "article"})

    assert retrieval.open_or_build_vector_store(get_settings()) == "built"
    assert calls == ["build"]


def test_a_missing_index_is_built(calls):
    assert read_index_info(get_settings().chroma_dir) is None
    assert retrieval.open_or_build_vector_store(get_settings()) == "built"


def test_rebuilding_can_be_forced(calls, monkeypatch):
    write_info(index_info(get_settings()))
    monkeypatch.setenv("REINDEX_ON_STARTUP", "true")
    get_settings.cache_clear()

    assert retrieval.open_or_build_vector_store(get_settings()) == "built"


def test_index_info_names_what_changes_the_vectors():
    info = index_info(get_settings())

    assert {"chunking", "chunk_size", "chunk_overlap", "embedding_model", "docs_dir", "format"} <= set(info)
