import pytest
from langchain_ollama import OllamaEmbeddings
from ollama import ResponseError

from app.services.ingestion import ResilientOllamaEmbeddings

BAD = "texte que le modèle ne sait pas encoder"


@pytest.fixture
def fake_backend(monkeypatch):
    calls = []

    def embed_documents(self, texts):
        calls.append(list(texts))
        if any(t == BAD for t in texts):
            raise ResponseError(
                "failed to encode response: json: unsupported value: NaN", 500
            )
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(OllamaEmbeddings, "embed_documents", embed_documents)
    return calls


def model():
    return ResilientOllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")


def test_a_healthy_batch_is_sent_once(fake_backend):
    vectors = model().embed_documents(["a", "bb"])

    assert vectors == [[1.0], [2.0]]
    assert fake_backend == [["a", "bb"]]


def test_one_bad_text_does_not_fail_the_batch(fake_backend):
    vectors = model().embed_documents(["a", BAD, "bb"])

    assert len(vectors) == 3
    assert vectors[0] == [1.0] and vectors[2] == [2.0]
    assert vectors[1] == [float(2 * len(BAD) + 1)]


def test_a_failing_query_is_embedded_doubled(fake_backend):
    assert model().embed_query(BAD) == [float(2 * len(BAD) + 1)]


def test_a_text_that_fails_even_doubled_raises(monkeypatch):
    def always_fail(self, texts):
        raise ResponseError("failed to encode response", 500)

    monkeypatch.setattr(OllamaEmbeddings, "embed_documents", always_fail)

    with pytest.raises(ResponseError):
        model().embed_query("x")
