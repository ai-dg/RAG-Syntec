"""
Answer generation pipeline.

Builds a context-constrained prompt from the retrieved chunks, calls the
configured LLM, and returns the answer together with its sources.
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import Settings

PROMPT_TEMPLATE = """{system_prompt}

Answer ONLY using the context below. If the context does not contain the
answer, say that you don't know.

Context:
{context}

Question: {question}

Answer:"""


def get_chat_model(settings: Settings):
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.chat_model_local,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def generate(question: str, retrieval_result: dict) -> dict:
    settings = Settings()
    llm = get_chat_model(settings)

    documents = [doc for doc, _score in retrieval_result["chunks"]]
    context = "\n\n".join(doc.page_content for doc in documents)
    sources = list({doc.metadata.get("source") for doc in documents})

    prompt = PROMPT_TEMPLATE.format(
        system_prompt=settings.system_prompt,
        context=context,
        question=question,
    )
    answer = llm.invoke(prompt)

    return {"answer": answer.content, "sources": sources}
