"""
FastAPI application entry point.

The vector store is built inside a `lifespan` handler, not at import time.
This keeps module imports side-effect-free (tests can import this module
without triggering a full re-index) and makes startup failures explicit:
if indexing fails, the app never starts, instead of passing readiness checks
while unable to serve any query.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings
from app.logger import configure_logging
from app.services.retrieval import get_vector_store

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    get_vector_store(settings)
    yield


app = FastAPI(title="RAGFoundry", lifespan=lifespan)

app.include_router(router)
