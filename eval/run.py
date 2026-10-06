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
from app.services.generation import strip_citations
from app.services.pipeline import answer_question
from eval.failures import is_abstention
from app.services.ingestion import (
    chunk_text,
    create_vector_store,
    get_embedding,
    load_docs,
)
from eval.labels import labels_for_chunking
from app.services.reranking import load_reranker
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
WITHDRAWN_ANSWER = "Je ne sais pas."
CLASSES = ["in_topic_answerable", "in_topic_unanswerable", "off_topic", "adversarial"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the evaluation and write a result file."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="use the held-out split (final measurement only)",
    )
    parser.add_argument(
        "--from-predictions", type=Path, help="replay metrics from a saved result file"
    )
    parser.add_argument(
        "--limit", type=int, help="only the first N questions (smoke tests)"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="rebuild the vector store from the corpus first (slow)",
    )
    return parser.parse_args(argv)


def load_golden(held_out: bool) -> list[dict]:
    name = "golden_held_out.jsonl" if held_out else "golden.jsonl"
    with open(EVAL_DIR / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def prepare_index(reindex: bool) -> int:
    """Open the persisted index (or rebuild it) and return its chunk count."""
    settings = get_settings()
    if reindex:
        store = create_vector_store(chunk_text(load_docs()))
        set_vector_store(store)
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
        "embedding_model": (
            settings.embedding_model_local if local else settings.embedding_model
        ),
        "top_k": settings.top_k,
        "chunking": settings.chunking,
        "include_superseded": settings.include_superseded,
        "relevance_threshold": settings.relevance_threshold,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "system_prompt": settings.system_prompt,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_candidates": settings.rerank_candidates,
        "rerank_model": settings.rerank_model,
        "guardrail_mode": settings.guardrail_mode,
        "abstention_model_path": (
            settings.abstention_model_path
            if settings.guardrail_mode == "classifier"
            else None
        ),
        "cite_sources": settings.cite_sources,
        "input_guard": settings.input_guard,
        "output_guard": settings.output_guard,
        "groundedness_threshold": settings.groundedness_threshold,
    }


def run_predictions(questions: list[dict]) -> list[dict]:
    """One record per question through the same pipeline as the API.

    `guardrail_passed` is the decision before generation (input check and
    retrieval guardrail); `answered` is the final outcome after the output check.
    An answer the output check withdrew is recorded as an abstention, which is
    what the user sees."""
    retrieve("Quelle est la durée du préavis en cas de démission ?")

    predictions = []
    for q in questions:
        result = answer_question(q["question"])
        retrieved = result.get("retrieval") or {"chunks": [], "best_score": None}
        generated = result.get("generation")
        documents = [doc for doc, _score in retrieved["chunks"]]
        timings = result["latency_ms"]

        record = {
            "id": q["id"],
            "true_class": q["class"],
            "guardrail_passed": generated is not None,
            "answered": result["answered"],
            "refusal_reason": result["refusal_reason"],
            "best_distance": retrieved.get("best_score"),
            "confidence": retrieved.get("confidence"),
            "retrieved_chunk_ids": [d.metadata["chunk_id"] for d in documents],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
            "context": "\n\n".join(d.page_content for d in documents),
            "answer": None,
            "answer_raw": None,
            "support_score": None,
            "citations": [],
            "invalid_citations": [],
            "retrieval_s": (
                timings["retrieval"] / 1000 if "retrieval" in timings else 0.0
            ),
            "generation_s": None,
        }

        if generated is not None:
            record["generation_s"] = timings["generation"] / 1000
            record["answer_raw"] = generated["answer"]
            record["support_score"] = generated.get("support_score")
            record["answer"] = (
                strip_citations(generated["answer"])
                if result["answered"]
                else WITHDRAWN_ANSWER
            )
            record["citations"] = [
                c["chunk_id"] for c in generated.get("citations", [])
            ]
            record["invalid_citations"] = generated.get("invalid_citations", [])

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


def end_to_end(predictions: list[dict]) -> dict:
    """Per class, how many questions got a substantive answer (not a refusal and
    not an abstention): the outcome the user sees after every layer."""
    by_class = {}
    for cls in CLASSES:
        rows = [p for p in predictions if p["true_class"] == cls]
        substantive = sum(
            1
            for p in rows
            if p.get("answered", p["answer"] is not None)
            and p["answer"] is not None
            and not is_abstention(p["answer"])
        )
        by_class[cls] = {"n": len(rows), "substantive_answers": substantive}
    return by_class


def compute_metrics(predictions: list[dict]) -> dict:
    k = get_settings().top_k

    answerable = [p for p in predictions if p["true_class"] == "in_topic_answerable"]
    retrieved = [p["retrieved_chunk_ids"] for p in answerable]
    relevant = [p["relevant_chunk_ids"] for p in answerable]
    retrieval = {
        "n": len(answerable),
        "k": k,
        "recall_at_k": _mean(
            [recall_at_k(r, rel, k) for r, rel in zip(retrieved, relevant)]
        ),
        "precision_at_k": _mean(
            [precision_at_k(r, rel, k) for r, rel in zip(retrieved, relevant)]
        ),
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

    answered = [p for p in predictions if p["answer"] is not None]
    citations = {
        "answered": len(answered),
        "with_a_citation": sum(1 for p in answered if p.get("citations")),
        "with_an_invalid_citation": sum(
            1 for p in answered if p.get("invalid_citations")
        ),
    }

    return {
        "n_questions": len(predictions),
        "end_to_end": end_to_end(predictions),
        "citations": citations,
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
        metadata = {
            **saved["metadata"],
            "label": args.label,
            "date": date.today().isoformat(),
            "replayed_from": args.from_predictions.name,
        }
    else:
        questions = load_golden(args.held_out)
        mode = get_settings().chunking
        translated = labels_for_chunking(
            {q["id"]: q["relevant_chunk_ids"] for q in questions}, mode
        )
        questions = [
            {**q, "relevant_chunk_ids": translated[q["id"]]} for q in questions
        ]
        if args.limit:
            questions = questions[: args.limit]
        index_chunk_count = prepare_index(args.reindex)
        settings = get_settings()
        if settings.rerank_enabled:
            load_reranker(settings.rerank_model)
        metadata = collect_run_metadata(split, args.label)
        metadata["index_chunk_count"] = index_chunk_count
        predictions = run_predictions(questions)

    metrics = compute_metrics(predictions)
    path = write_result(metadata, predictions, metrics)
    append_summary_row(metadata, metrics, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
