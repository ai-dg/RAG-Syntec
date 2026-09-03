from fastapi.testclient import TestClient
from app.services.retrieval import set_vector_store
from langchain_core.documents import Document

from unittest.mock import MagicMock

from app.main import app

IN_TOPIC = Document(page_content="in-topic content", metadata={"source": "a.md"})
OFF_TOPIC = Document(page_content="off-topic content", metadata={"source": "b.md"})

def fake_store(results):
    store = MagicMock()
    store.similarity_search_with_score.return_value = results
    return store

def fake_llm(answer_text="the answer"):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=answer_text)
    return llm

def test_health_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_query_returns_refusal_when_context_not_found():
    set_vector_store(fake_store([(IN_TOPIC, 1.5)]))
    client = TestClient(app)

    response = client.post("/query", json={"question" : "What is Lasagne?"})


    assert response.status_code == 200
    assert response.json()["context_found"] is False
    assert response.json()["sources"] == []
    assert response.json()["answer"] == "I can't answer the question with the available documents."

def test_query_returns_generated_answer_when_context_found(monkeypatch):
    set_vector_store(fake_store([(IN_TOPIC, 0.3)]))    
    monkeypatch.setattr(
        "app.services.generation.get_chat_model", lambda settings: fake_llm()
    )

    client = TestClient(app)

    response = client.post("/query", json={"question" : "Quel est le salaire minimum?"})

    assert response.status_code == 200
    assert response.json()["context_found"] is True
    assert response.json()["sources"] == ["a.md"]
    assert response.json()["answer"] == "the answer"


def test_query_rejects_empty_question():
    client = TestClient(app)

    response = client.post("/query", json={"question" : ""})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "string_too_short"


def test_query_rejects_question_over_max_length():
    client = TestClient(app)

    too_long = "a" * 2001

    response = client.post("/query", json={"question" : too_long})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "string_too_long"