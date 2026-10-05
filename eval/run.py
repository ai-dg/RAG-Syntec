
import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.generation import generate
from app.services.ingestion import get_embedding
from app.services.retrieval import get_vector_store, retrieve, set_vector_store
from eval.metrics import (
    confusion_matrix,
    faithfulness_proxy,
    false_acceptance_rate,
    false_refusal_rate,
    mrr,
    precision_at_k,
    recall_at_k,
)

EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
SUMMARY_FILE = EVAL_DIR / "RESULTS.md"
CLASSES = ["in_topic_answerable", "in_topic_unanswerable", "off_topic", "adversarial"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the evaluation and write a result file.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--held-out", action="store_true", help="use the held-out split (final measurement only)")
    parser.add_argument("--from-predictions", type=Path, help="replay metrics from a saved result file")
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke tests)")
    parser.add_argument("--reindex", action="store_true", help="rebuild the vector store from the corpus first (slow)")
    return parser.parse_args(argv)


def load_golden(held_out: bool) -> list[dict]:
    name = "golden_held_out.jsonl" if held_out else "golden.jsonl"
    with open(EVAL_DIR / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def prepare_index(reindex: bool) -> int:
    """Open the persisted index (or rebuild it) and return its chunk count."""
    settings = get_settings()
    if reindex:
        store = get_vector_store(settings)
    else:
        store = Chroma(
            persist_directory=settings.chroma_dir,
            embedding_function=get_embedding(settings),
        )
        set_vector_store(store)
    return len(store.get(include=[])["ids"])


def collect_run_metadata(split: str, label: str) -> dict:
    settings = get_settings()
    local = settings.llm_provider == "ollama"
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    return {
        "label": label,
        "date": date.today().isoformat(),
        "split": split,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "corpus_hash": manifest["corpus_hash"],
        "llm_provider": settings.llm_provider,
        "chat_model": settings.chat_model_local if local else settings.chat_model,
        "embedding_model": settings.embedding_model_local if local else settings.embedding_model,
        "top_k": settings.top_k,
        "relevance_threshold": settings.relevance_threshold,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "system_prompt": settings.system_prompt,
    }


def run_predictions(questions: list[dict]) -> list[dict]:
    retrieve("warm-up")

    predictions = []
    for q in questions:
        start = time.perf_counter()
        result = retrieve(q["question"])
        retrieval_s = time.perf_counter() - start

        documents = [doc for doc, _score in result["chunks"]]
        record = {
            "id": q["id"],
            "true_class": q["class"],
            "guardrail_passed": result["context_found"],
            "best_distance": result["best_score"],
            "retrieved_chunk_ids": [d.metadata["chunk_id"] for d in documents],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
            "context": "\n\n".join(d.page_content for d in documents),
            "answer": None,
            "retrieval_s": retrieval_s,
            "generation_s": None,
        }

        if result["context_found"]:
            start = time.perf_counter()
            record["answer"] = generate(q["question"], result)["answer"]
            record["generation_s"] = time.perf_counter() - start

        predictions.append(record)
    return predictions


def latency_percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    return {
        "n": len(values),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _mean(values: list[float]):
    return float(np.mean(values)) if values else None


def compute_metrics(predictions: list[dict]) -> dict:
    k = get_settings().top_k

    answerable = [p for p in predictions if p["true_class"] == "in_topic_answerable"]
    retrieved = [p["retrieved_chunk_ids"] for p in answerable]
    relevant = [p["relevant_chunk_ids"] for p in answerable]
    retrieval = {
        "n": len(answerable),
        "k": k,
        "recall_at_k": _mean([recall_at_k(r, rel, k) for r, rel in zip(retrieved, relevant)]),
        "precision_at_k": _mean([precision_at_k(r, rel, k) for r, rel in zip(retrieved, relevant)]),
        "mrr": float(mrr(retrieved, relevant)) if answerable else None,
    }

    guardrail = {
        "false_refusal_rate": false_refusal_rate(predictions),
        "false_acceptance_rate": false_acceptance_rate(predictions),
        "confusion_matrix": confusion_matrix(predictions),
    }

    faithfulness = {}
    for cls in CLASSES:
        scores = [
            faithfulness_proxy(p["answer"], p["context"])
            for p in predictions
            if p["true_class"] == cls and p["answer"] is not None
        ]
        faithfulness[cls] = {"n": len(scores), "mean": _mean(scores)}

    latency = {
        "retrieval_s": latency_percentiles([p["retrieval_s"] for p in predictions]),
        "generation_s": latency_percentiles(
            [p["generation_s"] for p in predictions if p["generation_s"] is not None]
        ),
    }

    return {
        "n_questions": len(predictions),
        "guardrail": guardrail,
        "retrieval": retrieval,
        "faithfulness_proxy": faithfulness,
        "latency": latency,
    }


def write_result(metadata: dict, predictions: list[dict], metrics: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{metadata['date']}_{metadata['label']}.json"
    payload = {"metadata": metadata, "metrics": metrics, "predictions": predictions}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def _fmt(value, digits=2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def append_summary_row(metadata: dict, metrics: dict, result_path: Path) -> None:
    header = (
        "| date | label | split | git | n | false refusal | false accept | recall@k | MRR "
        "| faithfulness (answerable) | retrieval p50/p95 (s) | generation p50/p95 (s) | result |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    ret = metrics["latency"]["retrieval_s"]
    gen = metrics["latency"]["generation_s"]
    row = (
        f"| {metadata['date']} | {metadata['label']} | {metadata['split']} "
        f"| {metadata['git_sha'][:7]}{'*' if metadata['git_dirty'] else ''} "
        f"| {metrics['n_questions']} "
        f"| {_fmt(metrics['guardrail']['false_refusal_rate'])} "
        f"| {_fmt(metrics['guardrail']['false_acceptance_rate'])} "
        f"| {_fmt(metrics['retrieval']['recall_at_k'])} "
        f"| {_fmt(metrics['retrieval']['mrr'])} "
        f"| {_fmt(metrics['faithfulness_proxy']['in_topic_answerable']['mean'])} "
        f"| {_fmt(ret['p50'], 3)} / {_fmt(ret['p95'], 3)} "
        f"| {_fmt(gen['p50'], 1)} / {_fmt(gen['p95'], 1)} (n={gen['n']}) "
        f"| {result_path.name} |\n"
    )
    new_file = not SUMMARY_FILE.exists()
    with open(SUMMARY_FILE, "a") as f:
        if new_file:
            f.write("# Evaluation results\n\n")
            f.write(header)
        f.write(row)


def main(argv=None) -> None:
    args = parse_args(argv)
    split = "held_out" if args.held_out else "visible"

    if args.from_predictions:
        saved = json.loads(args.from_predictions.read_text())
        predictions = saved["predictions"]
        metadata = {**saved["metadata"], "label": args.label, "date": date.today().isoformat(),
                    "replayed_from": args.from_predictions.name}
    else:
        questions = load_golden(args.held_out)
        if args.limit:
            questions = questions[: args.limit]
        index_chunk_count = prepare_index(args.reindex)
        metadata = collect_run_metadata(split, args.label)
        metadata["index_chunk_count"] = index_chunk_count
        predictions = run_predictions(questions)

    metrics = compute_metrics(predictions)
    path = write_result(metadata, predictions, metrics)
    append_summary_row(metadata, metrics, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
