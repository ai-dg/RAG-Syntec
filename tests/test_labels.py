from langchain_core.documents import Document

from eval.labels import overlaps, translate_ids


def chunk(chunk_id, source, start, end):
    return Document(
        page_content="x" * (end - start),
        metadata={
            "chunk_id": chunk_id,
            "source": source,
            "start_index": start,
            "end_index": end,
        },
    )


def test_overlap_needs_the_same_document():
    assert not overlaps(("a.md", 0, 100), ("b.md", 0, 100))


def test_overlap_is_measured_against_the_shorter_span():
    assert overlaps(("a.md", 0, 500), ("a.md", 100, 200))
    assert overlaps(("a.md", 0, 100), ("a.md", 50, 300))
    assert not overlaps(("a.md", 0, 100), ("a.md", 60, 300))


def test_translate_maps_a_fixed_chunk_to_the_article_chunks_that_hold_its_text():
    reference = [
        chunk("a.md#chunk_0", "a.md", 0, 500),
        chunk("a.md#chunk_1", "a.md", 400, 900),
    ]
    new = [
        chunk("a.md#chunk_0", "a.md", 0, 300),
        chunk("a.md#chunk_1", "a.md", 300, 650),
        chunk("a.md#chunk_2", "a.md", 650, 900),
    ]

    assert translate_ids(["a.md#chunk_1"], reference, new) == [
        "a.md#chunk_1",
        "a.md#chunk_2",
    ]


def test_translate_ignores_unknown_ids():
    reference = [chunk("a.md#chunk_0", "a.md", 0, 500)]

    assert (
        translate_ids(["missing"], reference, [chunk("a.md#chunk_0", "a.md", 0, 500)])
        == []
    )
