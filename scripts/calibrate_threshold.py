"""Recommend a relevance threshold from the visible golden questions.

The guardrail compares the best (lowest) squared-L2 distance of a question to a
threshold. This script measures that distance for in-topic questions (answerable
and unanswerable) and off-topic questions of the visible split, on the persisted
index and with the same in-force filter as the retriever, and recommends the
threshold that separates them best. Adversarial questions are reported but not
used: they are written about the corpus' own subject, so a distance cannot
separate them (see design/guardrail.md).
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_chroma import Chroma

from app.config import get_settings
from app.services.ingestion import get_embedding

IN_TOPIC = ("in_topic_answerable", "in_topic_unanswerable")
OFF_TOPIC = ("off_topic",)


def load_questions(split: str = "visible") -> list[dict]:
    name = "golden_held_out.jsonl" if split == "held_out" else "golden.jsonl"
    with open(ROOT / "eval" / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def best_distance(store, question: str, search_filter) -> float:
    return store.similarity_search_with_score(question, k=1, filter=search_filter)[0][1]


def errors_at(threshold: float, in_scores: list[float], off_scores: list[float]) -> tuple[int, int]:
    false_refusals = sum(1 for s in in_scores if s > threshold)
    false_accepts = sum(1 for s in off_scores if s <= threshold)
    return false_refusals, false_accepts


def recommend_threshold(in_scores: list[float], off_scores: list[float]) -> dict:
    """Midpoint of the gap if the two groups separate; otherwise the midpoint of the
    interval between consecutive scores that minimises false refusals plus false
    acceptances (ties broken by fewer false acceptances)."""
    if max(in_scores) < min(off_scores):
        threshold = (max(in_scores) + min(off_scores)) / 2
        return {"threshold": threshold, "separable": True, "errors": (0, 0)}

    cuts = sorted(set(in_scores + off_scores))
    candidates = [cuts[0] - 1e-6] + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])] + [cuts[-1] + 1e-6]
    best = min(candidates, key=lambda t: (sum(errors_at(t, in_scores, off_scores)), errors_at(t, in_scores, off_scores)[1]))
    return {"threshold": best, "separable": False, "errors": errors_at(best, in_scores, off_scores)}


def describe(label: str, scores: list[float]) -> str:
    return (f"{label:22s} n={len(scores):2d}  min {min(scores):.3f}  median {statistics.median(scores):.3f}"
            f"  max {max(scores):.3f}")


def main() -> None:
    settings = get_settings()
    store = Chroma(persist_directory=settings.chroma_dir, embedding_function=get_embedding(settings))
    search_filter = None if settings.include_superseded else {"in_force": True}

    by_class: dict[str, list[float]] = {}
    for record in load_questions():
        by_class.setdefault(record["class"], []).append(best_distance(store, record["question"], search_filter))

    in_scores = [s for c in IN_TOPIC for s in by_class.get(c, [])]
    off_scores = [s for c in OFF_TOPIC for s in by_class.get(c, [])]
    for cls, scores in sorted(by_class.items()):
        print(describe(cls, scores))

    result = recommend_threshold(in_scores, off_scores)
    refusals, accepts = result["errors"]
    print(f"\nseparable: {result['separable']}")
    print(f"recommended threshold: {result['threshold']:.4f} (cosine similarity {1 - result['threshold'] / 2:.4f})")
    print(f"on these questions: {refusals}/{len(in_scores)} in-topic refused, {accepts}/{len(off_scores)} off-topic accepted")
    current = settings.relevance_threshold
    refusals, accepts = errors_at(current, in_scores, off_scores)
    print(f"current threshold {current}: {refusals}/{len(in_scores)} in-topic refused, {accepts}/{len(off_scores)} off-topic accepted")


if __name__ == "__main__":
    main()
