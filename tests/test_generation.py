"""
Tests for app.services.generation.

`sources` is built by deduplicating `Document.metadata["source"]` values.
Deduplicating via a set and converting straight to a list makes the output
order depend on Python's per-process string hash randomization
(`PYTHONHASHSEED`), not on the source values themselves — see T0.3 in
TODO.md. This test pins the output to a sorted order so it is stable across
process runs, not just within one.
"""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.services.generation import generate


def fake_llm(answer_text="the answer"):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=answer_text)
    return llm


def test_sources_are_deduplicated_and_sorted(monkeypatch):
    monkeypatch.setattr(
        "app.services.generation.get_chat_model", lambda settings: fake_llm()
    )

    documents = [
        (Document(page_content="...", metadata={"source": "syntec_annexe_iii.md"}), 0.2),
        (Document(page_content="...", metadata={"source": "syntec_base.md"}), 0.3),
        (Document(page_content="...", metadata={"source": "syntec_annexe_iii.md"}), 0.4),
        (Document(page_content="...", metadata={"source": "syntec_avenant_4.md"}), 0.5),
    ]

    result = generate("a question", {"chunks": documents})

    assert result["sources"] == [
        "syntec_annexe_iii.md",
        "syntec_avenant_4.md",
        "syntec_base.md",
    ]
