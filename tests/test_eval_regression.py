"""Guardrail regression guard on the real index (no generation).

Marked `eval`: excluded from the default run because it needs the persisted index
and a running Ollama. Run it with `uv run pytest -m eval`. It fails if a change
lets more off-topic questions through, or refuses more in-topic ones, than the
budget below, measured on the visible golden split.
"""

import json
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.eval

ROOT = Path(__file__).resolve().parent.parent
FALSE_ACCEPTANCE_BUDGET = 0.0
FALSE_REFUSAL_BUDGET = 0.20


@pytest.fixture
def real_settings(monkeypatch):
    from app.config import get_settings

    for name in (
        "DOCS_DIR",
        "CHROMA_DIR",
        "RELEVANCE_THRESHOLD",
        "SYSTEM_PROMPT",
        "EMBEDDING_MODEL_LOCAL",
        "CHAT_MODEL_LOCAL",
        "OLLAMA_BASE_URL",
        "TOP_K",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "CHUNKING",
        "GUARDRAIL_MODE",
        "INCLUDE_SUPERSEDED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(ROOT)
    get_settings.cache_clear()
    settings = get_settings()
    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Ollama is not reachable")
    if not (ROOT / settings.chroma_dir).exists():
        pytest.skip("no persisted index")
    return settings


def test_guardrail_error_rates_stay_within_budget(real_settings):
    from langchain_chroma import Chroma

    from app.services import retrieval
    from app.services.ingestion import get_embedding

    store = Chroma(
        persist_directory=real_settings.chroma_dir,
        embedding_function=get_embedding(real_settings),
    )
    retrieval.set_vector_store(store)
    records = [
        json.loads(line)
        for line in (ROOT / "eval" / "golden.jsonl").read_text().splitlines()
        if line.strip()
    ]

    passed = {
        r["id"]: retrieval.retrieve(r["question"])["context_found"] for r in records
    }
    in_topic = [r for r in records if r["class"].startswith("in_topic")]
    off_topic = [r for r in records if r["class"] == "off_topic"]
    false_refusal = sum(not passed[r["id"]] for r in in_topic) / len(in_topic)
    false_acceptance = sum(passed[r["id"]] for r in off_topic) / len(off_topic)

    assert (
        false_acceptance <= FALSE_ACCEPTANCE_BUDGET
    ), f"off-topic accepted: {false_acceptance:.2f}"
    assert (
        false_refusal <= FALSE_REFUSAL_BUDGET
    ), f"in-topic refused: {false_refusal:.2f}"
