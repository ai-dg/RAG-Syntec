"""
API routes.

This module is the entry point of the RAG system. It validates incoming
requests, coordinates retrieval and generation, and returns structured HTTP
responses. It contains no ingestion, vector-search, or prompt-construction
logic of its own.

Request flow:
    HTTP request -> validation -> retrieval -> guardrail -> generation -> response
"""

import logging

from fastapi import APIRouter

from app.schemas import HealthAnswer, QueryAnswer, QueryQuestion
from app.services.generation import generate
from app.services.retrieval import retrieve

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthAnswer)
def health() -> HealthAnswer:
    return HealthAnswer(status="ok")


@router.post("/query", response_model=QueryAnswer)
def query(payload: QueryQuestion) -> QueryAnswer:
    logger.info("Query received: %r", payload.question)

    retrieval_result = retrieve(payload.question)

    logger.info(
        "Retrieval done: context_found=%s best_score=%s",
        retrieval_result["context_found"],
        retrieval_result["best_score"],
    )

    if not retrieval_result["context_found"]:
        return QueryAnswer(
            answer="I cannot answer this from the available documents.",
            sources=[],
            context_found=False,
        )

    generation_result = generate(payload.question, retrieval_result)

    logger.info("Generation done: %d source(s)", len(generation_result["sources"]))

    return QueryAnswer(
        answer=generation_result["answer"],
        sources=generation_result["sources"],
        context_found=True,
    )
