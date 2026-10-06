import logging
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api import routes
from app.config import get_settings
from app.logger import RequestIdFilter
from app.main import app
from app.services import retrieval
from app.services.generation import resolve_citations, strip_citations

CHUNK = Document(
    page_content="Le préavis est de deux mois.",
    metadata={
        "source": "a.md",
        "chunk_id": "a.md#chunk_0",
        "in_force": True,
        "article": "Article 4.1",
    },
)


def store(results):
    fake = MagicMock()
    fake.similarity_search_with_score.return_value = results
    return fake


def llm(text):
    model = MagicMock()
    model.invoke.return_value = MagicMock(content=text)
    return model


@pytest.fixture
def client():
    return TestClient(app)


def test_a_refusal_is_a_200_that_states_its_reason(client):
    retrieval.set_vector_store(store([(CHUNK, 1.5)]))

    body = client.post("/query", json={"question": "Lasagne ?"}).json()

    assert body["context_found"] is False
    assert body["refusal_reason"] == "below_relevance_threshold"
    assert body["best_distance"] == pytest.approx(1.5)
    assert body["guardrail_mode"] == "threshold"
    assert set(body["latency_ms"]) == {"retrieval"}


def test_an_answer_reports_its_latency_per_stage(client, monkeypatch):
    retrieval.set_vector_store(store([(CHUNK, 0.3)]))
    monkeypatch.setattr(
        "app.services.generation.get_chat_model", lambda s: llm("Deux mois.")
    )

    body = client.post("/query", json={"question": "Préavis ?"}).json()

    assert body["context_found"] is True
    assert body["refusal_reason"] is None
    assert set(body["latency_ms"]) == {"retrieval", "generation"}


def test_citations_are_returned_when_enabled(client, monkeypatch):
    monkeypatch.setenv("CITE_SOURCES", "true")
    get_settings.cache_clear()
    retrieval.set_vector_store(store([(CHUNK, 0.3)]))
    model = llm("Le préavis est de deux mois [1]. Voir aussi [7].")
    monkeypatch.setattr("app.services.generation.get_chat_model", lambda s: model)

    body = client.post("/query", json={"question": "Préavis ?"}).json()

    assert [c["chunk_id"] for c in body["citations"]] == ["a.md#chunk_0"]
    assert body["citations"][0]["article"] == "Article 4.1"
    prompt = model.invoke.call_args[0][0]
    assert "[1] Le préavis est de deux mois." in prompt


def test_the_prompt_is_unchanged_when_citations_are_off(monkeypatch):
    retrieval.set_vector_store(store([(CHUNK, 0.3)]))
    model = llm("Deux mois.")
    monkeypatch.setattr("app.services.generation.get_chat_model", lambda s: model)

    TestClient(app).post("/query", json={"question": "Préavis ?"})

    prompt = model.invoke.call_args[0][0]
    assert "[1]" not in prompt
    assert "numéro" not in prompt


def test_invented_citations_are_rejected():
    chunks = [(CHUNK, 0.3), (CHUNK, 0.4)]

    valid, invalid = resolve_citations("A [2]. B [1] [1]. C [3] [0].", chunks)

    assert [c["number"] for c in valid] == [1, 2]
    assert invalid == [0, 3]


def test_citation_markers_can_be_stripped_for_scoring():
    assert strip_citations("Deux mois [1]. Trois [2][3].") == "Deux mois. Trois."


def test_the_request_id_is_returned_and_can_be_supplied(client):
    retrieval.set_vector_store(store([(CHUNK, 1.5)]))

    generated = client.post("/query", json={"question": "q"})
    supplied = client.post(
        "/query", json={"question": "q"}, headers={"X-Request-ID": "abc123"}
    )

    assert generated.headers["X-Request-ID"] == generated.json()["request_id"]
    assert supplied.json()["request_id"] == "abc123"


def test_every_log_line_of_a_request_carries_its_id(client, monkeypatch):
    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record)

    handler = Capture()
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        retrieval.set_vector_store(store([(CHUNK, 0.3)]))
        monkeypatch.setattr(
            "app.services.generation.get_chat_model", lambda s: llm("Deux mois.")
        )
        client.post(
            "/query", json={"question": "q"}, headers={"X-Request-ID": "trace-1"}
        )
    finally:
        root.removeHandler(handler)

    request_lines = [r for r in seen if r.name == "app.api.routes"]
    assert len(request_lines) == 3
    assert {r.request_id for r in request_lines} == {"trace-1"}


def test_ready_reports_a_down_model_server(client, monkeypatch):
    retrieval.set_vector_store(store([]))

    def down(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(routes.httpx, "get", down)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "index_loaded": True,
        "ollama_reachable": False,
    }


def test_ready_when_the_index_and_the_model_server_are_up(client, monkeypatch):
    retrieval.set_vector_store(store([]))
    monkeypatch.setattr(routes.httpx, "get", lambda *a, **k: MagicMock(status_code=200))

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_without_an_index(client, monkeypatch):
    monkeypatch.setattr(retrieval, "_vector_store", None)
    monkeypatch.setattr(routes.httpx, "get", lambda *a, **k: MagicMock(status_code=200))

    assert client.get("/ready").status_code == 503


def test_stats_count_outcomes(client, monkeypatch):
    monkeypatch.setattr(routes, "_outcomes", routes.Counter())
    monkeypatch.setattr(routes, "_latencies_ms", routes.deque(maxlen=1000))
    retrieval.set_vector_store(store([(CHUNK, 1.5)]))

    client.post("/query", json={"question": "q"})
    client.post("/query", json={"question": "q"})

    body = client.get("/stats").json()
    assert body["outcomes"] == {"refused:below_relevance_threshold": 2}
    assert body["latency_ms"]["n"] == 2
