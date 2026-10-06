"""
Answer generation pipeline.

This module uses a Large Language Model (LLM) to generate an answer from the
user's question and the chunks selected by the retrieval pipeline.

Responsibilities:
- Build the prompt sent to the LLM.
- Add the retrieved chunks as context.
- Instruct the model to answer only from the provided context.
- Generate a clear and relevant final answer.
- Associate the answer with its document sources.
- Return a controlled "I don't know" response when the context does not
  contain enough information.

Inputs:
- The user's question.
- The relevant chunks returned by the retrieval pipeline.

Output:
- A generated answer grounded in the retrieved documents.
"""

import re

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from app.config import Settings, get_settings

CITATION_MARKER = re.compile(r"\[(\d+)\]")
CITATION_INSTRUCTION = (
    "Chaque extrait du contexte est numéroté. Après chaque affirmation, cite le "
    "numéro de l'extrait qui la justifie entre crochets, par exemple [1] ou [2]. "
    "N'invente jamais de numéro."
)


def get_chat_model(settings: Settings):
    if settings.llm_provider == "openai":
        model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        return model
    if settings.llm_provider == "ollama":
        model = ChatOllama(
            model=settings.chat_model_local,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        return model
    raise ValueError(f"Unsupporteed LLM provider: {settings.llm_provider}")


def strip_citations(answer: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", answer)


def citation(rank: int, document, distance: float) -> dict:
    metadata = document.metadata
    return {
        "number": rank,
        "chunk_id": metadata.get("chunk_id"),
        "source": metadata.get("source"),
        "article": metadata.get("article") or None,
        "in_force": metadata.get("in_force", True),
        "page": metadata.get("page"),
        "distance": float(distance),
    }


def resolve_citations(answer: str, chunks: list) -> tuple[list[dict], list[int]]:
    """Cited passages, and the cited numbers that match no retrieved passage
    (an invented citation is rejected, never shown as a source)."""
    cited = sorted({int(n) for n in CITATION_MARKER.findall(answer)})
    valid = [citation(n, *chunks[n - 1]) for n in cited if 1 <= n <= len(chunks)]
    invalid = [n for n in cited if not 1 <= n <= len(chunks)]
    return valid, invalid


def generate(question: str, retrieval_result: dict):
    settings = get_settings()
    llm = get_chat_model(settings)

    chunks = retrieval_result["chunks"]
    documents = [doc for doc, _score in chunks]
    sources = sorted({doc.metadata.get("source") for doc in documents})

    if settings.cite_sources:
        context = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(documents, start=1))
        instruction = f"\n    {CITATION_INSTRUCTION}"
    else:
        context = "\n\n".join(doc.page_content for doc in documents)
        instruction = ""

    prompt = f"""{settings.system_prompt}
    Réponds UNIQUEMENT à partir du contexte ci-dessous. Si le contexte ne contient
    pas la réponse, dis que tu ne sais pas.{instruction}

    Contexte :
    {context}

    Question: {question}

    Réponse :"""
    answer = llm.invoke(prompt).content

    citations, invalid = resolve_citations(answer, chunks) if settings.cite_sources else ([], [])
    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "invalid_citations": invalid,
    }
