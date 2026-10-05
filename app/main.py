"""
FastAPI application entry point.

This module creates the application and registers its API routes.
"""

from fastapi import FastAPI
from app.api.routes import router
from app.config import get_settings
from app.logger import configure_logging
from contextlib import asynccontextmanager
from app.services.retrieval import get_vector_store
from app.services.reranking import load_reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    get_vector_store(settings)
    if settings.rerank_enabled:
        load_reranker(settings.rerank_model)
    yield


configure_logging()

app = FastAPI(title="RAGFoundry", lifespan=lifespan)

app.include_router(router)
