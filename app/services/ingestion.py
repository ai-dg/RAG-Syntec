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

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def chunk_text(documents):
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunk could be generated from the documents")

    counters = {}
    for chunk in chunks:
        source = chunk.metadata['source']
        index = counters.get(source, 0)
        chunk.metadata['chunk_id'] = f"{source}#chunk_{index}"
        counters[source] = index + 1    

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

    logger.info(
        "%d chunks stored in Chroma at %s",
        len(chunks),
        settings.chroma_dir,
    )

    return vector_store
