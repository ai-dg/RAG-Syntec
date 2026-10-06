"""
API routes.

This module is the entry point of the RAG system. It receives HTTP requests,
validates their data, coordinates the retrieval and generation pipelines, and
returns structured HTTP responses.

Main endpoints:
- `POST /query`: answer a question from the documents, or refuse and say why.
- `GET /health`: liveness, the process is serving requests.
- `GET /ready`: readiness, the index is loaded and the model server answers.
- `GET /stats`: request outcomes and end-to-end latency.

Request flow:
    HTTP request
        -> request validation
        -> document retrieval and answer-or-refuse decision
        -> answer generation with citations
        -> JSON response (a refusal is a 200 with a reason, not an error)

The API coordinates the components but contains no retrieval, guardrail or prompt
logic itself.
"""

import logging
import statistics
import time
from collections import Counter, deque

import httpx
from fastapi import APIRouter, Response

from app.config import get_settings
from app.logger import request_id_var
from app.schemas import HealthAnswer, QueryAnswer, QueryQuestion, ReadinessAnswer
from app.services import retrieval
from app.services.generation import generate

logger = logging.getLogger(__name__)

router = APIRouter()

REFUSAL_ANSWER = "I can't answer the question with the available documents."
_latencies_ms: deque[float] = deque(maxlen=1000)
_outcomes: Counter = Counter()


@router.get("/health", response_model=HealthAnswer)
def health() -> HealthAnswer:
    """Liveness: the process is up and serving requests."""
    return HealthAnswer(status="ok")


def ollama_reachable(base_url: str) -> bool:
    try:
        return (
            httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0).status_code
            == 200
        )
    except httpx.HTTPError:
        return False


@router.get("/ready", response_model=ReadinessAnswer)
def ready(response: Response) -> ReadinessAnswer:
    """Readiness: the index is loaded and, in local mode, Ollama answers."""
    settings = get_settings()
    checks = {"index_loaded": retrieval._vector_store is not None}
    if settings.llm_provider == "ollama":
        checks["ollama_reachable"] = ollama_reachable(settings.ollama_base_url)
    is_ready = all(checks.values())
    if not is_ready:
        response.status_code = 503
    return ReadinessAnswer(status="ready" if is_ready else "not_ready", checks=checks)


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


@router.get("/stats")
def stats() -> dict:
    """Request counts by outcome and end-to-end latency over the last 1 000 requests."""
    latencies = sorted(_latencies_ms)
    return {
        "requests": sum(_outcomes.values()),
        "outcomes": dict(_outcomes),
        "latency_ms": {
            "n": len(latencies),
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "mean": statistics.fmean(latencies) if latencies else None,
        },
    }


def _record(outcome: str, start: float) -> None:
    _outcomes[outcome] += 1
    _latencies_ms.append((time.perf_counter() - start) * 1000)


@router.post("/query", response_model=QueryAnswer)
def query(payload: QueryQuestion) -> QueryAnswer:
    start = time.perf_counter()
    logger.info("query received", extra={"question_chars": len(payload.question)})

    retrieval_result = retrieval.retrieve(payload.question)
    retrieval_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "retrieval done",
        extra={
            "stage": "retrieval",
            "latency_ms": round(retrieval_ms, 1),
            "chunk_count": len(retrieval_result["chunks"]),
            "best_distance": retrieval_result["best_score"],
            "guardrail_mode": retrieval_result.get("guardrail_mode"),
            "decision": "answer" if retrieval_result["context_found"] else "refuse",
            "refusal_reason": retrieval_result.get("refusal_reason"),
            "confidence": retrieval_result.get("confidence"),
        },
    )

    common = {
        "refusal_reason": retrieval_result.get("refusal_reason"),
        "confidence": retrieval_result.get("confidence"),
        "guardrail_mode": retrieval_result.get("guardrail_mode") or "threshold",
        "best_distance": retrieval_result["best_score"],
        "request_id": request_id_var.get(),
    }

    if not retrieval_result["context_found"]:
        _record(f"refused:{common['refusal_reason']}", start)
        return QueryAnswer(
            answer=REFUSAL_ANSWER,
            sources=[],
            context_found=False,
            latency_ms={"retrieval": round(retrieval_ms, 1)},
            **common,
        )

    generation_start = time.perf_counter()
    generation_result = generate(payload.question, retrieval_result)
    generation_ms = (time.perf_counter() - generation_start) * 1000
    logger.info(
        "generation done",
        extra={
            "stage": "generation",
            "latency_ms": round(generation_ms, 1),
            "source_count": len(generation_result["sources"]),
            "citation_count": len(generation_result.get("citations", [])),
            "invalid_citations": generation_result.get("invalid_citations", []),
        },
    )
    _record("answered", start)

    return QueryAnswer(
        answer=generation_result["answer"],
        sources=generation_result["sources"],
        context_found=True,
        citations=generation_result.get("citations", []),
        latency_ms={
            "retrieval": round(retrieval_ms, 1),
            "generation": round(generation_ms, 1),
        },
        **common,
    )
