import asyncio
import subprocess
import sys

import pytest
from langchain_core.documents import Document

from app import main as main_module
from app.config import get_settings
from app.services import reranking, retrieval
from app.services.reranking import rerank


def doc(text, chunk_id):
    return Document(page_content=text, metadata={"chunk_id": chunk_id})


def by_length(question, texts):
    return [float(len(t)) for t in texts]


CANDIDATES = [
    (doc("a", "c1"), 0.30),
    (doc("aaaa", "c2"), 0.40),
    (doc("aa", "c3"), 0.50),
    (doc("aaa", "c4"), 0.60),
]


def ids(results):
    return [d.metadata["chunk_id"] for d, _ in results]


def test_rerank_puts_the_highest_score_first_not_the_lowest_distance():
    result = rerank("q", CANDIDATES, top_n=4, scorer=by_length)

    assert ids(result) == ["c2", "c4", "c3", "c1"]


def test_rerank_keeps_at_most_top_n_results():
    assert len(rerank("q", CANDIDATES, top_n=2, scorer=by_length)) == 2


def test_rerank_keeps_the_dense_distance_and_stores_the_score_in_metadata():
    result = rerank("q", CANDIDATES, top_n=3, scorer=by_length)

    assert [distance for _, distance in result] == [0.40, 0.60, 0.50]
    assert [d.metadata["rerank_score"] for d, _ in result] == [4.0, 3.0, 2.0]


def test_rerank_does_not_modify_the_original_documents():
    rerank("q", CANDIDATES, top_n=4, scorer=by_length)

    assert all("rerank_score" not in d.metadata for d, _ in CANDIDATES)


def test_rerank_keeps_the_dense_order_between_equal_scores():
    result = rerank("q", CANDIDATES, top_n=4, scorer=lambda q, texts: [1.0] * len(texts))

    assert ids(result) == ["c1", "c2", "c3", "c4"]


def test_rerank_of_no_candidates_is_empty():
    assert rerank("q", [], top_n=3, scorer=by_length) == []


def test_rerank_rejects_a_scorer_that_returns_the_wrong_number_of_scores():
    with pytest.raises(ValueError):
        rerank("q", CANDIDATES, top_n=2, scorer=lambda q, texts: [1.0])


def test_scoring_without_a_loaded_model_gives_a_clear_error(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(reranking, "_model", None)

    with pytest.raises(RuntimeError, match="load_reranker"):
        reranking._score_with_borrowed_gpu("q", ["text"])


def test_loading_without_the_extra_names_the_command_to_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(RuntimeError, match="rerank"):
        reranking.load_reranker("any-model")


def test_importing_the_reranking_module_does_not_import_torch():
    code = (
        "import sys; import app.services.reranking; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )

    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_rerank_settings_default_to_disabled_with_twenty_candidates(monkeypatch):
    monkeypatch.delenv("RERANK_ENABLED")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.rerank_enabled is False
    assert settings.rerank_candidates == 20
    assert settings.rerank_model == "Qwen/Qwen3-Reranker-0.6B"


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.requested_k = None

    def similarity_search_with_score(self, question, k):
        self.requested_k = k
        return self.results[:k]


def ranked_results():
    return [(doc(f"text {i}", f"c{i}"), 0.1 * (i + 1)) for i in range(10)]


@pytest.fixture
def enable_rerank(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_CANDIDATES", "8")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.55")
    get_settings.cache_clear()


def use_store(monkeypatch, store):
    monkeypatch.setattr(retrieval, "_vector_store", store)


def test_retrieve_without_reranking_is_unchanged(monkeypatch):
    store = FakeStore(ranked_results())
    use_store(monkeypatch, store)
    monkeypatch.setattr(
        retrieval, "rerank", lambda *a, **k: pytest.fail("rerank must not be called")
    )

    result = retrieval.retrieve("q")

    assert store.requested_k == 3
    assert ids(result["chunks"]) == ["c0", "c1", "c2"]
    assert result["context_found"] is True
    assert result["best_score"] == pytest.approx(0.1)


def test_retrieve_with_reranking_asks_for_the_candidates_then_keeps_top_k(
    monkeypatch, enable_rerank
):
    store = FakeStore(ranked_results())
    use_store(monkeypatch, store)
    seen = {}

    def fake_rerank(question, candidates, top_n):
        seen["candidates"] = ids(candidates)
        seen["top_n"] = top_n
        return list(reversed(candidates))[:top_n]

    monkeypatch.setattr(retrieval, "rerank", fake_rerank)

    result = retrieval.retrieve("q")

    assert store.requested_k == 8
    assert seen["top_n"] == 3
    assert ids(result["chunks"]) == ["c4", "c3", "c2"]


def test_the_threshold_filters_candidates_before_they_are_reranked(
    monkeypatch, enable_rerank
):
    use_store(monkeypatch, FakeStore(ranked_results()))
    seen = {}

    def fake_rerank(question, candidates, top_n):
        seen["candidates"] = ids(candidates)
        return candidates[:top_n]

    monkeypatch.setattr(retrieval, "rerank", fake_rerank)

    retrieval.retrieve("q")

    assert seen["candidates"] == ["c0", "c1", "c2", "c3", "c4"]


def test_best_score_stays_the_dense_distance_after_reranking(monkeypatch, enable_rerank):
    use_store(monkeypatch, FakeStore(ranked_results()))
    monkeypatch.setattr(
        retrieval, "rerank", lambda question, candidates, top_n: candidates[::-1][:top_n]
    )

    result = retrieval.retrieve("q")

    assert result["best_score"] == pytest.approx(0.1)


def test_a_question_beyond_the_threshold_is_refused_without_reranking(
    monkeypatch, enable_rerank
):
    far = [(doc("far", "c0"), 0.80), (doc("farther", "c1"), 0.90)]
    use_store(monkeypatch, FakeStore(far))
    monkeypatch.setattr(
        retrieval, "rerank", lambda *a, **k: pytest.fail("rerank must not be called")
    )

    result = retrieval.retrieve("q")

    assert result["context_found"] is False
    assert result["chunks"] == []
    assert result["best_score"] == pytest.approx(0.80)


def run_lifespan():
    async def go():
        async with main_module.lifespan(main_module.app):
            pass

    asyncio.run(go())


def test_lifespan_loads_the_reranker_only_when_it_is_enabled(monkeypatch):
    loaded = []
    monkeypatch.setattr(main_module, "get_vector_store", lambda settings: None)
    monkeypatch.setattr(main_module, "load_reranker", lambda name: loaded.append(name))

    run_lifespan()
    assert loaded == []

    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_MODEL", "some/model")
    get_settings.cache_clear()
    run_lifespan()
    assert loaded == ["some/model"]


class FakeCrossEncoder:
    def __init__(self, fail=False):
        self.fail = fail
        self.moves = []
        self.batch_sizes = []
        self.model = self

    def to(self, device):
        self.moves.append(device)

    def predict(self, pairs, batch_size):
        self.batch_sizes.append(batch_size)
        if self.fail:
            raise RuntimeError("scoring failed")
        return [1.0] * len(pairs)


@pytest.fixture
def gpu_present(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_scoring_uses_a_small_batch_so_it_fits_next_to_the_ollama_model(
    monkeypatch, gpu_present
):
    fake = FakeCrossEncoder()
    monkeypatch.setattr(reranking, "_model", fake)

    scores = reranking._score_with_borrowed_gpu("q", ["text"] * 20)

    assert fake.batch_sizes == [reranking.RERANK_BATCH_SIZE]
    assert reranking.RERANK_BATCH_SIZE <= 5
    assert scores == [1.0] * 20


def test_the_model_is_lent_to_the_gpu_then_returned(monkeypatch, gpu_present):
    fake = FakeCrossEncoder()
    monkeypatch.setattr(reranking, "_model", fake)

    reranking._score_with_borrowed_gpu("q", ["text"])

    assert fake.moves == ["cuda", "cpu"]


def test_the_model_goes_back_to_the_cpu_even_if_scoring_fails(monkeypatch, gpu_present):
    fake = FakeCrossEncoder(fail=True)
    monkeypatch.setattr(reranking, "_model", fake)

    with pytest.raises(RuntimeError, match="scoring failed"):
        reranking._score_with_borrowed_gpu("q", ["text"])

    assert fake.moves == ["cuda", "cpu"]
