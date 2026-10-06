import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.services.ingestion import chunk_text, split_into_articles

LONG_RULE = " ".join(
    f"Phrase numéro {i} de l'article sur le préavis." for i in range(12)
)
TEXT = (
    "# Convention\n\n> Identifiant Légifrance : `KALITEXT0001`\n\n"
    "## Article 1 (non en vigueur)\n\nAncienne règle : préavis de deux ans.\n\n"
    f"## Article 2\n\n{LONG_RULE}\n\n"
    "## Article\n\n"
    "## Article 3\n\nLa période d'essai est de trois mois.\n"
)


@pytest.fixture
def article_mode(monkeypatch):
    monkeypatch.setenv("CHUNKING", "article")
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    get_settings.cache_clear()


def chunks_of(text, source="a.md"):
    return chunk_text([Document(page_content=text, metadata={"source": source})])


def test_sections_start_at_headings_and_drop_heading_only_sections():
    sections = split_into_articles(TEXT)
    firsts = [s.split("\n", 1)[0] for _, s in sections]

    assert firsts == [
        "# Convention",
        "## Article 1 (non en vigueur)",
        "## Article 2",
        "## Article 3",
    ]
    assert all(TEXT[start : start + len(s)] == s for start, s in sections)


def test_no_chunk_crosses_an_article_boundary(article_mode):
    for chunk in chunks_of(TEXT):
        assert chunk.page_content.count("## ") <= 1


def test_every_piece_of_a_long_article_repeats_its_heading(article_mode):
    pieces = [c for c in chunks_of(TEXT) if c.metadata["article"] == "Article 2"]

    assert len(pieces) > 1
    assert all(c.page_content.startswith("## Article 2\n") for c in pieces)


def test_status_comes_from_the_article_heading(article_mode):
    status = {c.metadata["article"]: c.metadata["in_force"] for c in chunks_of(TEXT)}

    assert status["Article 1 (non en vigueur)"] is False
    assert status["Article 2"] is True
    assert status["Article 3"] is True


def test_offsets_point_at_the_original_text(article_mode):
    for chunk in chunks_of(TEXT):
        start, end = chunk.metadata["start_index"], chunk.metadata["end_index"]
        assert TEXT[start:end] in chunk.page_content


def test_no_empty_chunks_and_ids_are_sequential(article_mode):
    chunks = chunks_of(TEXT)

    assert all(c.page_content.strip() for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == [
        f"a.md#chunk_{i}" for i in range(len(chunks))
    ]


def test_fixed_mode_is_still_the_default():
    chunks = chunks_of(TEXT)

    assert all("article" not in c.metadata for c in chunks)
