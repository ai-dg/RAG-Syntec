"""Compare two evaluation result files on the same footing."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.ingestion import chunk_text, load_docs
from eval.failures import load_gists, summarize
from eval.metrics import faithfulness_proxy, mrr, recall_at_k


def load_result(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_golden_labels(split: str) -> dict[str, list[str]]:
    name = "golden_held_out.jsonl" if split == "held_out" else "golden.jsonl"
    with open(ROOT / "eval" / name) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r["relevant_chunk_ids"] for r in records}


def relabel(result: dict, labels: dict[str, list[str]]) -> dict:
    """Score a saved run against the current golden labels, not the ones saved with it."""
    predictions = [
        {**p, "relevant_chunk_ids": labels.get(p["id"], p["relevant_chunk_ids"])}
        for p in result["predictions"]
    ]
    return {**result, "predictions": predictions}


def retrieval_at_k(predictions: list[dict], k: int) -> dict:
    answerable = [p for p in predictions if p["true_class"] == "in_topic_answerable"]
    retrieved = [p["retrieved_chunk_ids"][:k] for p in answerable]
    relevant = [p["relevant_chunk_ids"] for p in answerable]
    if not answerable:
        return {"n": 0, "recall": None, "mrr": None}
    return {
        "n": len(answerable),
        "recall": float(np.mean([recall_at_k(r, rel, k) for r, rel in zip(retrieved, relevant)])),
        "mrr": float(mrr(retrieved, relevant)),
    }


def faithfulness_at_k(predictions: list[dict], texts_by_id: dict[str, str], k: int) -> dict:
    scores = []
    for p in predictions:
        if p["true_class"] != "in_topic_answerable" or p["answer"] is None:
            continue
        context = "\n\n".join(texts_by_id[c] for c in p["retrieved_chunk_ids"][:k])
        scores.append(faithfulness_proxy(p["answer"], context))
    return {"n": len(scores), "mean": float(np.mean(scores)) if scores else None}


def faithfulness_full_context(predictions: list[dict]) -> dict:
    scores = [
        faithfulness_proxy(p["answer"], p["context"])
        for p in predictions
        if p["true_class"] == "in_topic_answerable" and p["answer"] is not None
    ]
    return {"n": len(scores), "mean": float(np.mean(scores)) if scores else None}


def truncate_prediction(prediction: dict, texts_by_id: dict[str, str], k: int) -> dict:
    ids = prediction["retrieved_chunk_ids"][:k]
    truncated = dict(prediction)
    truncated["retrieved_chunk_ids"] = ids
    truncated["context"] = "\n\n".join(texts_by_id[c] for c in ids)
    return truncated


def compare(a: dict, b: dict, texts_by_id: dict[str, str], k: int) -> list[tuple[str, object, object]]:
    pa, pb = a["predictions"], b["predictions"]
    ra, rb = retrieval_at_k(pa, k), retrieval_at_k(pb, k)
    fa, fb = faithfulness_at_k(pa, texts_by_id, k), faithfulness_at_k(pb, texts_by_id, k)
    ffa, ffb = faithfulness_full_context(pa), faithfulness_full_context(pb)
    ga, gb = a["metrics"]["guardrail"], b["metrics"]["guardrail"]
    la, lb = a["metrics"]["latency"], b["metrics"]["latency"]
    cut_a = [truncate_prediction(p, texts_by_id, k) for p in pa]
    cut_b = [truncate_prediction(p, texts_by_id, k) for p in pb]
    sa = summarize(cut_a, load_gists(a["metadata"]["split"]))
    sb = summarize(cut_b, load_gists(b["metadata"]["split"]))

    rows = [
        ("top_k setting", a["metadata"]["top_k"], b["metadata"]["top_k"]),
        (f"recall@{k} (answerable, n={ra['n']})", ra["recall"], rb["recall"]),
        (f"MRR cut at {k}", ra["mrr"], rb["mrr"]),
        (f"faithfulness proxy vs first {k} chunks (biased down)", fa["mean"], fb["mean"]),
        ("faithfulness proxy vs full context (biased up)", ffa["mean"], ffb["mean"]),
        ("false refusal rate", ga["false_refusal_rate"], gb["false_refusal_rate"]),
        ("false acceptance rate", ga["false_acceptance_rate"], gb["false_acceptance_rate"]),
        ("retrieval p50 (s)", la["retrieval_s"]["p50"], lb["retrieval_s"]["p50"]),
        ("retrieval p95 (s)", la["retrieval_s"]["p95"], lb["retrieval_s"]["p95"]),
        ("generation p50 (s)", la["generation_s"]["p50"], lb["generation_s"]["p50"]),
        ("generation p95 (s)", la["generation_s"]["p95"], lb["generation_s"]["p95"]),
        (f"failures, classified on the first {k} chunks", sa["n_failures"], sb["n_failures"]),
    ]
    for category in sa["counts"]:
        rows.append((f"  {category}", sa["counts"][category], sb["counts"][category]))
    return rows


def format_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation result files.")
    parser.add_argument("result_a", type=Path)
    parser.add_argument("result_b", type=Path)
    parser.add_argument("--k", type=int, default=3, help="cut every ranking and context at k (default 3)")
    args = parser.parse_args(argv)

    a, b = load_result(args.result_a), load_result(args.result_b)
    a = relabel(a, load_golden_labels(a["metadata"]["split"]))
    b = relabel(b, load_golden_labels(b["metadata"]["split"]))
    texts_by_id = {c.metadata["chunk_id"]: c.page_content for c in chunk_text(load_docs())}

    print(f"A = {args.result_a.name} ({a['metadata']['label']})   B = {args.result_b.name} ({b['metadata']['label']})")
    print(f"both cut at k = {args.k}; both scored against the current golden labels\n")
    print(f"{'metric':56s} {'A':>10s} {'B':>10s} {'B - A':>10s}")
    for label, va, vb in compare(a, b, texts_by_id, args.k):
        delta = (
            format_value(vb - va)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float))
            else ""
        )
        print(f"{label:56s} {format_value(va):>10s} {format_value(vb):>10s} {delta:>10s}")


if __name__ == "__main__":
    main()
