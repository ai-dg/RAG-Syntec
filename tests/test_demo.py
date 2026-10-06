from unittest.mock import MagicMock

import httpx
import pytest

gr = pytest.importorskip("gradio")

from demo import app as demo_app


def response(body):
    fake = MagicMock()
    fake.json.return_value = body
    fake.raise_for_status.return_value = None
    return fake


def test_the_interface_builds():
    assert demo_app.build_interface() is not None


def test_a_refusal_is_shown_with_its_reason(monkeypatch):
    body = {"context_found": False, "answer": "x", "refusal_reason": "below_relevance_threshold",
            "citations": [], "guardrail_mode": "threshold", "best_distance": 1.2}
    monkeypatch.setattr(demo_app.httpx, "post", lambda *a, **k: response(body))

    answer, citations, inspector = demo_app.ask("q")

    assert "Refused" in answer
    assert "close enough" in answer
    assert "refuse" in inspector


def test_an_answer_shows_its_citations(monkeypatch):
    body = {"context_found": True, "answer": "Deux mois [1].", "refusal_reason": None,
            "citations": [{"number": 1, "article": "Article 4.1", "source": "data/converted/a.md", "in_force": True}]}
    monkeypatch.setattr(demo_app.httpx, "post", lambda *a, **k: response(body))

    answer, citations, _ = demo_app.ask("q")

    assert "Deux mois" in answer
    assert "Article 4.1" in citations and "a.md" in citations


def test_an_unreachable_api_is_reported_not_raised(monkeypatch):
    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(demo_app.httpx, "post", down)

    assert "API unavailable" in demo_app.ask("q")[0]
