import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.services import retrieval
from app.services.ingestion import article_sections, chunk_text, in_force_share

TEXT = (
    "# Convention\n\n"
    "## Article 1 (non en vigueur)\n\nAncienne règle : préavis de deux ans.\n\n"
    "## Article 1\n\nRègle actuelle : préavis de huit mois.\n"
)


def test_article_sections_mark_superseded_headings():
    sections = article_sections(TEXT)

    assert [in_force for _, in_force in sections] == [True, False, True]
    assert sections[0][0] == 0


def test_text_before_the_first_heading_counts_as_in_force():
    assert article_sections("Préambule sans titre.") == [(0, True)]


def test_in_force_share_inside_one_section():
    sections = [(0, True), (10, False), (20, True)]

    assert in_force_share(sections, 12, 18) == 0.0
    assert in_force_share(sections, 22, 30) == 1.0


def test_in_force_share_across_a_boundary_is_weighted_by_characters():
    sections = [(0, True), (10, False), (20, True)]

    assert in_force_share(sections, 5, 15) == 0.5
    assert in_force_share(sections, 15, 25) == 0.5
    assert in_force_share(sections, 8, 18) == 0.2


def test_in_force_share_of_an_empty_span_is_one():
    assert in_force_share([(0, False)], 5, 5) == 1.0


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "60")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    get_settings.cache_clear()


def test_chunk_text_tags_each_chunk_with_its_legal_status(small_chunks):
    chunks = chunk_text([Document(page_content=TEXT, metadata={"source": "a.md"})])
    by_text = {c.page_content: c.metadata["in_force"] for c in chunks}

    old = next(v for t, v in by_text.items() if "deux ans" in t)
    new = next(v for t, v in by_text.items() if "huit mois" in t)
    assert old is False
    assert new is True


def test_chunk_ids_are_unchanged_by_the_status_tagging(small_chunks):
    chunks = chunk_text([Document(page_content=TEXT, metadata={"source": "a.md"})])

    assert [c.metadata["chunk_id"] for c in chunks] == [
        f"a.md#chunk_{i}" for i in range(len(chunks))
    ]
    assert all("start_index" in c.metadata for c in chunks)


def test_pages_of_one_pdf_are_tagged_separately(small_chunks):
    pages = [
        Document(
            page_content="## Article 3 (non en vigueur)\n\nAncien texte.",
            metadata={"source": "b.pdf", "page": 0},
        ),
        Document(
            page_content="## Article 3\n\nNouveau texte.",
            metadata={"source": "b.pdf", "page": 1},
        ),
    ]

    chunks = chunk_text(pages)

    assert [c.metadata["in_force"] for c in chunks] == [False, True]


class FilteringStore:
    def __init__(self, results):
        self.results = results
        self.requested_filter = "not called"

    def similarity_search_with_score(self, question, k, filter=None):
        self.requested_filter = filter
        kept = [
            (d, s)
            for d, s in self.results
            if not filter
            or all(d.metadata.get(key) == value for key, value in filter.items())
        ]
        return kept[:k]


def ranked():
    return [
        (
            Document(
                page_content="ancien", metadata={"chunk_id": "old", "in_force": False}
            ),
            0.2,
        ),
        (
            Document(
                page_content="actuel", metadata={"chunk_id": "new", "in_force": True}
            ),
            0.3,
        ),
    ]


def test_retrieve_excludes_superseded_text_by_default(monkeypatch):
    store = FilteringStore(ranked())
    monkeypatch.setattr(retrieval, "_vector_store", store)

    result = retrieval.retrieve("q")

    assert store.requested_filter == {"in_force": True}
    assert [d.metadata["chunk_id"] for d, _ in result["chunks"]] == ["new"]
    assert result["best_score"] == pytest.approx(0.3)


def test_superseded_text_can_be_included_explicitly(monkeypatch):
    monkeypatch.setenv("INCLUDE_SUPERSEDED", "true")
    get_settings.cache_clear()
    store = FilteringStore(ranked())
    monkeypatch.setattr(retrieval, "_vector_store", store)

    result = retrieval.retrieve("q")

    assert store.requested_filter is None
    assert [d.metadata["chunk_id"] for d, _ in result["chunks"]] == ["old", "new"]
