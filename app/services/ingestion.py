"""
Document ingestion pipeline.

This module prepares source documents for semantic retrieval.

Responsibilities:
- Load documents in supported formats: Markdown, plain text, and PDF.
- Validate files before processing them.
- Extract and clean their textual content.
- Split the content into meaningful, overlapping chunks.
- Attach metadata such as the source filename, file type, and PDF page number.
- Generate an embedding for each chunk.
- Store the chunks and their embeddings in Chroma.

Design notes:

1. Multi-format support
   - `.md` and `.txt` files are read as plain text.
   - `.pdf` files are extracted page by page so that page numbers can be
     preserved in the metadata.

2. Input validation
   - Check that the documents directory exists.
   - Reject empty files and files whose content cannot be extracted.
   - Skip an invalid file when other valid documents can still be processed.
   - Stop ingestion with an explicit error if no valid document remains.

3. Chunking strategy
   - Prefer splitting at paragraph, line, or sentence boundaries.
   - Use an overlap between consecutive chunks to preserve context around
     chunk boundaries.
   - Keep chunk size and overlap configurable.

4. Idempotent indexing
   - Chroma's `from_documents` APPENDS to an existing collection rather than
     replacing it. Re-running ingestion against a persisted collection
     without clearing it first silently duplicates every vector. This module
     always clears the target collection before writing, so re-indexing is
     safe to run any number of times against the same `chroma_dir`.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from pathlib import Path
import logging

from app.config import Settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ragfoundry"


def chunk_text(documents, settings: Settings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunk could be generated from the documents")

    return chunks


def verify_docs_path(docs_dir):
    path = Path(docs_dir)

    if not path.exists():
        raise FileNotFoundError(f"Documents directory doesn't exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Documents path is not a directory: {path}")

    return path


def check_document_content(document: Document):
    content = (
        document.page_content.replace("\x00", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    if not content:
        logger.warning(
            "Skipping document with empty content: source=%s, page=%s",
            document.metadata.get("source", "unknown"),
            document.metadata.get("page", "unknown"),
        )
        return None

    document.page_content = content

    logger.debug(
        "Document content validated: source=%s, page=%s",
        document.metadata.get("source", "unknown"),
        document.metadata.get("page", "unknown"),
    )

    return document


def load_docs(settings: Settings):
    path = verify_docs_path(settings.docs_dir)

    loaders = [
        DirectoryLoader(
            path=path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        ),
        DirectoryLoader(
            path=path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        ),
        DirectoryLoader(
            path=path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True,
        ),
    ]

    documents = []

    for loader in loaders:
        loaded_documents = loader.load()
        for document in loaded_documents:
            valid_document = check_document_content(document)

            if valid_document is not None:
                documents.append(valid_document)

    if not documents:
        logger.error("No valid document found in directory: %s", settings.docs_dir)
        raise ValueError(f"No valid document found in {settings.docs_dir}")

    logger.info(
        "%d valid document part(s) loaded from %s",
        len(documents),
        settings.docs_dir,
    )
    return documents


def get_embedding(settings: Settings):
    if settings.llm_provider == "openai":
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    if settings.llm_provider == "ollama":
        return OllamaEmbeddings(
            model=settings.embedding_model_local,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def create_vector_store(chunks, settings: Settings):
    if not chunks:
        raise ValueError("No chunks available for indexing")

    embedding = get_embedding(settings)

    # Clear any previously persisted collection before writing. Chroma's
    # from_documents() appends rather than replaces, so skipping this step
    # duplicates every vector on each restart against the same chroma_dir.
    existing = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=settings.chroma_dir,
    )
    existing.delete_collection()

    vector_store = Chroma.from_documents(
        collection_name=COLLECTION_NAME,
        documents=chunks,
        embedding=embedding,
        persist_directory=settings.chroma_dir,
    )
    logger.info(
        "%d chunks stored in Chroma at %s",
        len(chunks),
        settings.chroma_dir,
    )

    return vector_store


def build_vector_store(settings: Settings):
    """Full pipeline: load, chunk, and index. Called once at app startup."""
    documents = load_docs(settings)
    chunks = chunk_text(documents, settings)
    return create_vector_store(chunks, settings)
