"""One table over every evaluation run, scored the same way.

Each run is scored against the current golden labels (translated to its own
chunking), cut at 3 chunks, with the failure taxonomy computed on those 3 chunks.
Prints a Markdown table.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.ingestion import chunk_text, load_docs
from eval.compare_runs import (
    faithfulness_at_k,
    load_golden_labels,
    load_result,
    relabel,
    retrieval_at_k,
    truncate_prediction,
)
from eval.failures import classify_failure, is_abstention, load_gists
from eval.labels import labels_for_chunking

RUNS = [
    ("baseline", "fixed chunks, threshold 0.74, all text"),
    ("topk10", "+ 10 chunks in the prompt (control)"),
    ("rerank", "+ cross-encoder reranking, 20 to 3 (rejected)"),
    ("versioning", "baseline + superseded text filtered"),
    ("article", "versioning + article chunking, threshold 0.888 (not kept)"),
    ("final", "versioning + input and output checks + citations"),
]


def row(result: dict, texts: dict[str, str], k: int = 3) -> dict:
    predictions = result["predictions"]
    retrieval = retrieval_at_k(predictions, k)
    cut = [truncate_prediction(p, texts, k) for p in predictions]
    answerable_failing = sum(
        1
        for p in cut
        if p["true_class"] == "in_topic_answerable"
        and _failed(p, result["metadata"]["split"])
    )

    def substantive(cls):
        return sum(
            1
            for p in predictions
            if p["true_class"] == cls and p["answer"] and not is_abstention(p["answer"])
        )

    guardrail = result["metrics"]["guardrail"]
    latency = result["metrics"]["latency"]
    return {
        "hit@3": retrieval["hit"],
        "recall@3": retrieval["recall"],
        "mrr": retrieval["mrr"],
        "false_refusal": guardrail["false_refusal_rate"],
        "false_acceptance": guardrail["false_acceptance_rate"],
        "answerable_failing": answerable_failing,
        "unanswerable_invented": substantive("in_topic_unanswerable"),
        "adversarial_followed": substantive("adversarial"),
        "faithfulness": faithfulness_at_k(predictions, texts, k)["mean"],
        "generation_p50": latency["generation_s"]["p50"],
        "generation_p95": latency["generation_s"]["p95"],
    }


_GISTS = {}


def _failed(prediction: dict, split: str) -> bool:
    gists = _GISTS.setdefault(split, load_gists(split))
    return classify_failure(prediction, gists.get(prediction["id"])) is not None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "eval" / "results")
    args = parser.parse_args(argv)

    documents = load_docs()
    texts_by_mode = {}
    print(
        "| run | change | hit@3 | recall@3 | MRR | false refusal | false acceptance "
        "| answerable failing /22 | unanswerable invented /11 | adversarial followed /11 "
        "| faithfulness proxy | generation p50 / p95 (s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, change in RUNS:
        matches = sorted(args.results.glob(f"*_{label}.json"))
        if not matches:
            continue
        result = load_result(matches[-1])
        mode = result["metadata"].get("chunking", "fixed")
        labels = labels_for_chunking(load_golden_labels(result["metadata"]["split"]), mode)
        result = relabel(result, labels)
        if mode not in texts_by_mode:
            texts_by_mode[mode] = {
                c.metadata["chunk_id"]: c.page_content
                for c in chunk_text(documents, mode=mode)
            }
        r = row(result, texts_by_mode[mode])
        print(
            f"| {label} | {change} | {r['hit@3']:.2f} | {r['recall@3']:.2f} | {r['mrr']:.2f} "
            f"| {r['false_refusal']:.2f} | {r['false_acceptance']:.2f} | {r['answerable_failing']} "
            f"| {r['unanswerable_invented']} | {r['adversarial_followed']} | {r['faithfulness']:.2f} "
            f"| {r['generation_p50']:.1f} / {r['generation_p95']:.1f} |"
        )


if __name__ == "__main__":
    main()
