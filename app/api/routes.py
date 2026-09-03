"""
API routes.

This module is the entry point of the RAG system. It receives HTTP requests,
validates their data, coordinates the retrieval and generation pipelines, and
returns structured HTTP responses.

Responsibilities:
- Expose the API endpoints, such as `POST /query` and `GET /health`.
- Validate incoming request data.
- Call the retrieval pipeline to find relevant document chunks.
- Return a controlled "I don't know" response when no relevant context is found.
- Call the generation pipeline when sufficient context is available.
- Build structured JSON responses containing the answer and its sources.
- Handle application errors and return appropriate HTTP status codes.

Main endpoints:
- `POST /query`: receive a user's question and return a grounded answer.
- `GET /health`: report whether the API is running and ready to receive requests.

Request flow:
    HTTP request
        -> request validation
        -> document retrieval
        -> relevance guardrail
        -> answer generation
        -> JSON response

Important:
The API coordinates the different components but should not contain the internal
logic for document ingestion, vector search, or prompt construction.
"""

import logging
from fastapi import APIRouter

from app.schemas import HealthAnswer, QueryAnswer, QueryQuestion
from app.services.retrieval import retrieve
from app.services.generation import generate

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
        "Retrieval done: context_found=%s " "best_score=%s",
        retrieval_result["context_found"],
        retrieval_result["best_score"],
    )

    if not retrieval_result["context_found"]:
        return QueryAnswer(
            answer="I can't answer the question with the available documents.",
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
