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

The original implementation only supports Markdown files and splits their
content into fixed chunks of 500 characters:

    Markdown files
        -> text extraction
        -> fixed-size chunking
        -> OpenAI embeddings
        -> Chroma vector store

This implementation improves the pipeline in four areas:

1. Multi-format support
   - `.md` and `.txt` files are read as plain text.
   - `.pdf` files are extracted page by page so that page numbers can be
     preserved in the metadata.

2. Input validation
   - Check that the documents directory exists.
   - Reject unsupported file formats.
   - Reject empty files.
   - Reject files whose content cannot be extracted.
   - Reject documents with empty extracted text.
   - Reject files that exceed the configured size limit.
   - Ensure that at least one valid document is available for indexing.

3. Error handling
   - Log invalid or unreadable files with a clear error message.
   - Skip an invalid file when other valid documents can still be processed.
   - Stop the ingestion with an explicit error if no valid document remains.

4. Chunking strategy
   - Prefer splitting at paragraph, line, or sentence boundaries.
   - Use an overlap between consecutive chunks to preserve context around
     chunk boundaries.
   - Keep the chunk size and overlap configurable.
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

from bisect import bisect_right
from pathlib import Path
import json
import logging
import re

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

ARTICLE_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
SUPERSEDED_MARKER = "non en vigueur"


def article_sections(text: str) -> list[tuple[int, bool]]:
    """(start offset, in force) for each article; text before the first heading is in force."""
    sections = [(0, True)]
    for match in ARTICLE_HEADING.finditer(text):
        sections.append(
            (match.start(), SUPERSEDED_MARKER not in match.group(1).lower())
        )
    return sections


def in_force_share(sections: list[tuple[int, bool]], start: int, end: int) -> float:
    """Share of the characters in [start, end) that belong to articles in force."""
    if end <= start:
        return 1.0
    starts = [s for s, _ in sections]
    index = max(bisect_right(starts, start) - 1, 0)
    covered = 0
    position = start
    while position < end:
        section_end = starts[index + 1] if index + 1 < len(sections) else end
        segment_end = min(section_end, end)
        if sections[index][1]:
            covered += segment_end - position
        position = segment_end
        index += 1
    return covered / (end - start)


MIN_SECTION_CHARS = 20


def split_into_articles(text: str) -> list[tuple[int, str]]:
    """(start offset, text) for each article section; a section starts at its heading."""
    bounds = [0] + [m.start() for m in ARTICLE_HEADING.finditer(text)] + [len(text)]
    sections = []
    for start, end in zip(bounds, bounds[1:]):
        section = text[start:end]
        heading = section.split("\n", 1)[0] if section.startswith("## ") else ""
        if len(section.strip()) - len(heading) >= MIN_SECTION_CHARS:
            sections.append((start, section))
    return sections


def article_chunks(
    document: Document, splitter: RecursiveCharacterTextSplitter
) -> list[Document]:
    """Chunks that never cross an article boundary; each chunk repeats its article heading
    so that a piece cut from the middle of a long article still names it. `start_index`
    points at the original text, not at the repeated heading."""
    chunks = []
    for section_start, section in split_into_articles(document.page_content):
        heading = section.split("\n", 1)[0] if section.startswith("## ") else ""
        body_start = section_start + len(heading)
        for piece in splitter.create_documents([section[len(heading) :]]):
            text = f"{heading}\n{piece.page_content}" if heading else piece.page_content
            start = body_start + piece.metadata["start_index"]
            metadata = {
                **document.metadata,
                "start_index": start,
                "end_index": start + len(piece.page_content),
                "article": heading[3:] if heading else "",
            }
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


def chunk_text(documents, mode: str | None = None):
    settings = get_settings()
    mode = mode or settings.chunking

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )

    if mode == "article":
        chunks = [
            chunk
            for document in documents
            for chunk in article_chunks(document, splitter)
        ]
    else:
        chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunk could be generated from the documents")

    sections_by_part = {
        (document.metadata["source"], document.metadata.get("page")): article_sections(
            document.page_content
        )
        for document in documents
    }

    counters = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        index = counters.get(source, 0)
        chunk.metadata["chunk_id"] = f"{source}#chunk_{index}"
        counters[source] = index + 1

        if "article" in chunk.metadata:
            chunk.metadata["in_force"] = (
                SUPERSEDED_MARKER not in chunk.metadata["article"].lower()
            )
            continue

        start = chunk.metadata.get("start_index", -1)
        chunk.metadata["end_index"] = start + len(chunk.page_content)
        sections = sections_by_part.get((source, chunk.metadata.get("page")))
        if sections is None or start < 0:
            chunk.metadata["in_force"] = True
            continue
        share = in_force_share(sections, start, chunk.metadata["end_index"])
        chunk.metadata["in_force"] = share >= 0.5

    return chunks


def verify_docs_path(docs_dir):
    path = Path(docs_dir)

    if not path.exists():
        raise FileNotFoundError(f"Documents directory doesn't exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Documents path is not a directry: {path}")

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
            "Skipping document with empty content: source= %s, page=%s",
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


def load_docs():
    settings = get_settings()

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
        logger.error(
            "No valid document found in directory: %s",
            settings.docs_dir,
        )
        raise ValueError(f"No valid document found in {settings.docs_dir}")

    logger.info(
        "%d valid document part(s) loaded from %s",
        len(documents),
        settings.docs_dir,
    )
    return documents


def get_embedding(settings: Settings):

    if settings.llm_provider == "openai":
        embedding = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        return embedding
    if settings.llm_provider == "ollama":
        embedding = OllamaEmbeddings(
            model=settings.embedding_model_local, base_url=settings.ollama_base_url
        )
        return embedding

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


INDEX_INFO = "index_info.json"
INDEX_FORMAT = 2


def index_info(settings: Settings) -> dict:
    """What an index was built with; an index is reused only if this matches."""
    local = settings.llm_provider == "ollama"
    return {
        "format": INDEX_FORMAT,
        "docs_dir": str(settings.docs_dir),
        "chunking": settings.chunking,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": (
            settings.embedding_model_local if local else settings.embedding_model
        ),
    }


def read_index_info(chroma_dir) -> dict | None:
    try:
        return json.loads((Path(chroma_dir) / INDEX_INFO).read_text())
    except (OSError, ValueError):
        return None


def create_vector_store(chunks):
    settings = get_settings()

    if not chunks:
        raise ValueError("No chunks available for indexing")

    embedding = get_embedding(settings)

    existing = Chroma(
        persist_directory=settings.chroma_dir, embedding_function=embedding
    )
    existing.delete_collection()

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=settings.chroma_dir
    )

    (Path(settings.chroma_dir) / INDEX_INFO).write_text(
        json.dumps(index_info(settings), indent=2)
    )

    logger.info(
        "%d chunks stored in Chroma at %s",
        len(chunks),
        settings.chroma_dir,
    )

    return vector_store
