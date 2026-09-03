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

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from app.config import Settings, get_settings


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


def generate(question: str, retrieval_result: dict):
    settings = get_settings()
    llm = get_chat_model(settings)

    documents = [doc for doc, _score in retrieval_result["chunks"]]

    context = "\n\n".join(doc.page_content for doc in documents)
    sources = sorted({doc.metadata.get("source") for doc in documents})

    prompt = f"""{settings.system_prompt}
    Réponds UNIQUEMENT à partir du contexte ci-dessous. Si le contexte ne contient
    pas la réponse, dis que tu ne sais pas.

    Contexte :
    {context}

    Question: {question}

    Réponse :"""
    answer = llm.invoke(prompt)

    result = {"answer": answer.content, "sources": sources}
    return result
