"""Translate golden labels between chunkings.

The golden set lists relevant chunks by id under the fixed 500-character
chunking. Under another chunking those ids do not exist, so a relevant chunk is
redefined by text position: a new chunk is relevant when it shares, with a
listed chunk of the same document, at least half of the shorter of the two
spans.
"""

from langchain_core.documents import Document

MIN_OVERLAP = 0.5


def span(chunk: Document) -> tuple[str, int, int]:
    metadata = chunk.metadata
    return metadata["source"], metadata["start_index"], metadata["end_index"]


def overlaps(
    a: tuple[str, int, int], b: tuple[str, int, int], min_share: float = MIN_OVERLAP
) -> bool:
    if a[0] != b[0]:
        return False
    shared = min(a[2], b[2]) - max(a[1], b[1])
    shortest = min(a[2] - a[1], b[2] - b[1])
    return shortest > 0 and shared >= min_share * shortest


def translate_ids(
    golden_ids: list[str], reference_chunks: list[Document], new_chunks: list[Document]
) -> list[str]:
    reference = {c.metadata["chunk_id"]: span(c) for c in reference_chunks}
    targets = [reference[g] for g in golden_ids if g in reference]
    return [
        c.metadata["chunk_id"]
        for c in new_chunks
        if any(overlaps(span(c), target) for target in targets)
    ]


def labels_for_chunking(
    golden: dict[str, list[str]], mode: str
) -> dict[str, list[str]]:
    """Golden labels expressed in the chunk ids of `mode` (identity for the fixed chunking)."""
    if mode == "fixed":
        return golden
    from app.services.ingestion import chunk_text, load_docs

    documents = load_docs()
    reference = chunk_text(documents, mode="fixed")
    new = chunk_text(documents, mode=mode)
    return {qid: translate_ids(ids, reference, new) for qid, ids in golden.items()}
