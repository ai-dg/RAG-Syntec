"""Measure how much repeated text the index holds and what it does to the top-k."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.ingestion import chunk_text, load_docs
from eval.metrics import recall_at_k

RESULTS_DIR = ROOT / "eval" / "results"
MIN_CONTAINMENT_CHARS = 100


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def duplicate_groups(texts_by_id: dict[str, str]) -> list[list[str]]:
    groups = defaultdict(list)
    for chunk_id, text in texts_by_id.items():
        groups[normalize(text)].append(chunk_id)
    return [ids for ids in groups.values() if len(ids) > 1]


def duplicate_stats(texts_by_id: dict[str, str]) -> dict:
    groups = duplicate_groups(texts_by_id)
    in_groups = sum(len(g) for g in groups)
    n_chunks = len(texts_by_id)
    spanning = sum(1 for g in groups if len({cid.split("#")[0] for cid in g}) > 1)
    return {
        "n_chunks": n_chunks,
        "n_groups": len(groups),
        "chunks_in_groups": in_groups,
        "share_in_groups": in_groups / n_chunks if n_chunks else 0.0,
        "redundant_copies": in_groups - len(groups),
        "groups_spanning_documents": spanning,
        "largest_group": max((len(g) for g in groups), default=0),
    }


def distinct_texts(chunk_ids: list[str], texts_by_id: dict[str, str]) -> int:
    return len({normalize(texts_by_id[c]) for c in chunk_ids})


def texts_equivalent(a: str, b: str) -> bool:
    a, b = normalize(a), normalize(b)
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= MIN_CONTAINMENT_CHARS and shorter in longer


def text_equivalent_recall(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    texts_by_id: dict[str, str],
    k: int,
) -> float:
    top = [texts_by_id[c] for c in retrieved_ids[:k]]
    found = sum(
        1
        for rel in relevant_ids
        if any(texts_equivalent(text, texts_by_id[rel]) for text in top)
    )
    return found / len(relevant_ids)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure chunk duplication and its effect on the top-k."
    )
    parser.add_argument(
        "result",
        type=Path,
        nargs="?",
        help="result JSON (default: newest in eval/results/)",
    )
    args = parser.parse_args(argv)

    path = args.result or max(
        RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime
    )
    saved = json.loads(path.read_text())
    k = saved["metadata"]["top_k"]

    chunks = chunk_text(load_docs())
    texts_by_id = {c.metadata["chunk_id"]: c.page_content for c in chunks}

    stats = duplicate_stats(texts_by_id)
    print(f"index: {stats['n_chunks']} chunks")
    print(
        f"  chunks whose exact text appears more than once: {stats['chunks_in_groups']} "
        f"({stats['share_in_groups']:.1%}) in {stats['n_groups']} groups"
    )
    print(f"  redundant copies (beyond one per group): {stats['redundant_copies']}")
    print(
        f"  groups spanning more than one document: {stats['groups_spanning_documents']}"
    )
    print(f"  largest group: {stats['largest_group']} copies")

    answerable = [
        p for p in saved["predictions"] if p["true_class"] == "in_topic_answerable"
    ]
    with_results = [
        p for p in saved["predictions"] if len(p["retrieved_chunk_ids"]) >= 2
    ]
    with_duplicates = [
        p
        for p in with_results
        if distinct_texts(p["retrieved_chunk_ids"], texts_by_id)
        < len(p["retrieved_chunk_ids"])
    ]
    print(f"\nresult file: {path.name} (k={k})")
    print(
        f"questions with at least 2 retrieved chunks: {len(with_results)}; "
        f"of those, top-{k} contained a repeated text: {len(with_duplicates)}"
    )

    id_recall = [
        recall_at_k(p["retrieved_chunk_ids"], p["relevant_chunk_ids"], k)
        for p in answerable
    ]
    text_recall = [
        text_equivalent_recall(
            p["retrieved_chunk_ids"], p["relevant_chunk_ids"], texts_by_id, k
        )
        for p in answerable
    ]
    print(f"\nanswerable questions: {len(answerable)}")
    print(f"  id-based recall@{k}:        {np.mean(id_recall):.3f}")
    print(f"  text-equivalent recall@{k}: {np.mean(text_recall):.3f}")
    flipped = [p["id"] for p, a, b in zip(answerable, id_recall, text_recall) if b > a]
    print(
        f"  questions whose recall rises with text equivalence ({len(flipped)}): {flipped}"
    )


if __name__ == "__main__":
    main()
