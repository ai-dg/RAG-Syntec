"""
FastAPI application entry point.

This module creates the application and registers its API routes.
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.routes import router
from app.config import get_settings
from app.logger import configure_logging, request_id_var
from app.services.reranking import load_reranker
from app.services.retrieval import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    get_vector_store(settings)
    if settings.rerank_enabled:
        load_reranker(settings.rerank_model)
    yield


configure_logging()

app = FastAPI(
    title="RAG-Syntec",
    description=(
        "Questions about the French Syntec collective agreement (IDCC 1486), answered "
        "only from the agreement's text, with citations. When the documents do not "
        "support an answer, the API refuses and states why (`refusal_reason`)."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def assign_request_id(request: Request, call_next):
    """Give each request an id (or keep the caller's X-Request-ID) that every log line carries."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(router)
