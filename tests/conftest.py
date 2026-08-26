"""
Shared test fixtures.

Settings() reads from the environment (and a possible .env file). Tests must
not depend on a developer's local .env, an API key, or a reachable Ollama
instance — so every required field is set here to a valid, inert value.
"""

import pytest


@pytest.fixture(autouse=True)
def settings_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("EMBEDDING_MODEL_LOCAL", "test-embedding-model")
    monkeypatch.setenv("CHAT_MODEL_LOCAL", "test-chat-model")
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("SYSTEM_PROMPT", "You are a test assistant.")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.9")
