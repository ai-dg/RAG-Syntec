"""Locate the relevant chunk of each answerable question in the full ranking."""

import argparse
import json
import sys
from pathlib import Path

from langchain_chroma import Chroma

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.ingestion import chunk_text, get_embedding, load_docs
from eval.failures import classify_failure, gist_coverage
from scripts.measure_duplicates import texts_equivalent

RESULTS_DIR = ROOT / "eval" / "results"
K_MAX = 50
HIT_KS = (1, 3, 5, 10, 20, 50)
MIN_BOUNDARY_GAIN = 0.15


def rank_of_relevant(ranked_ids: list[str], relevant_ids: list[str]) -> int | None:
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return position
    return None


def rank_of_equivalent(
    ranked_ids: list[str], relevant_ids: list[str], texts_by_id: dict[str, str]
) -> int | None:
    relevant_texts = [texts_by_id[r] for r in relevant_ids]
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if any(
            texts_equivalent(texts_by_id[chunk_id], text) for text in relevant_texts
        ):
            return position
    return None


def hit_rate_at_k(
    ranks: list[int | None], ks: tuple[int, ...] = HIT_KS
) -> dict[int, float]:
    return {
        k: (
            sum(1 for r in ranks if r is not None and r <= k) / len(ranks)
            if ranks
            else 0.0
        )
        for k in ks
    }


def split_across_chunks(
    relevant_id: str,
    gist: str,
    texts_by_id: dict[str, str],
    min_gain: float = MIN_BOUNDARY_GAIN,
) -> bool:
    document, number = relevant_id.rsplit("#chunk_", 1)
    alone = gist_coverage(texts_by_id[relevant_id], gist)
    best = alone
    for neighbour in (int(number) - 1, int(number) + 1):
        neighbour_id = f"{document}#chunk_{neighbour}"
        if neighbour_id in texts_by_id:
            joined = texts_by_id[relevant_id] + " " + texts_by_id[neighbour_id]
            best = max(best, gist_coverage(joined, gist))
    return best - alone >= min_gain


def load_golden(split: str) -> dict[str, dict]:
    name = "golden_held_out.jsonl" if split == "held_out" else "golden.jsonl"
    with open(ROOT / "eval" / name) as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Rank of the relevant chunk without the distance threshold."
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
    golden = load_golden(saved["metadata"]["split"])

    settings = get_settings()
    store = Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=get_embedding(settings),
    )
    texts_by_id = {
        c.metadata["chunk_id"]: c.page_content for c in chunk_text(load_docs())
    }

    rows = []
    for p in saved["predictions"]:
        if p["true_class"] != "in_topic_answerable":
            continue
        record = golden[p["id"]]
        results = store.similarity_search_with_score(record["question"], k=K_MAX)
        ranked_ids = [doc.metadata["chunk_id"] for doc, _ in results]
        distances = [score for _, score in results]
        relevant = record["relevant_chunk_ids"]
        id_rank = rank_of_relevant(ranked_ids, relevant)
        text_rank = rank_of_equivalent(ranked_ids, relevant, texts_by_id)
        right_rank = min(
            (r for r in (id_rank, text_rank) if r is not None), default=None
        )
        rows.append(
            {
                "id": p["id"],
                "id_rank": id_rank,
                "text_rank": text_rank,
                "right_rank": right_rank,
                "best_distance": distances[0],
                "right_distance": distances[right_rank - 1] if right_rank else None,
                "failure": classify_failure(p, record["expected_answer_gist"]),
                "boundary": any(
                    split_across_chunks(r, record["expected_answer_gist"], texts_by_id)
                    for r in relevant
                ),
            }
        )

    print(
        f"result file: {path.name}; {len(rows)} answerable questions; ranking up to {K_MAX}, no threshold\n"
    )
    print("hit rate = share of questions with at least one relevant chunk in the top k")
    for label, key in (("by id", "id_rank"), ("by text equivalence", "right_rank")):
        curve = hit_rate_at_k([r[key] for r in rows])
        print(f"  {label:22s}", "  ".join(f"@{k}: {v:.2f}" for k, v in curve.items()))

    print("\nretrieval failures (taxonomy), rank of the right chunk:")
    for r in rows:
        if r["failure"] == "retrieval":
            rd = (
                f"{r['right_distance']:.3f}"
                if r["right_distance"] is not None
                else "n/a"
            )
            print(
                f"  {r['id']}: id rank {r['id_rank']}, text rank {r['text_rank']}, "
                f"best distance {r['best_distance']:.3f}, right-chunk distance {rd}, "
                f"split across chunks: {r['boundary']}"
            )

    print("\nall answerable questions (id rank / text rank / best distance):")
    for r in rows:
        print(
            f"  {r['id']}: {r['id_rank']} / {r['text_rank']} / {r['best_distance']:.3f}"
            + ("  [boundary?]" if r["boundary"] else "")
        )


if __name__ == "__main__":
    main()
