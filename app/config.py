"""
Application configuration.

This module centralizes configuration values loaded from environment variables.
"""

from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    llm_provider: Literal["openai", "ollama"] = "openai"

    # API-key mode
    openai_api_key: SecretStr | None = None
    embedding_model: str | None = None
    chat_model: str | None = None

    # Local LLM mode
    ollama_base_url: str | None = None
    embedding_model_local: str | None = None
    chat_model_local: str | None = None

    docs_dir: str | None = None
    chroma_dir: str | None = None
    chunk_size: int
    chunk_overlap: int
    top_k: int
    system_prompt: str

    # Guardrail. Chroma's default metric is squared L2 distance: lower is
    # closer. This is NOT a similarity score. See design/guardrail.md.
    relevance_threshold: float

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> Self:
        if self.llm_provider == "openai":
            if (
                self.openai_api_key is None
                or not self.openai_api_key.get_secret_value().strip()
            ):
                raise ValueError("OPENAI_API_KEY is mandatory with LLM_PROVIDER=openai")

            if not self.embedding_model:
                raise ValueError(
                    "EMBEDDING_MODEL is mandatory with LLM_PROVIDER=openai"
                )

            if not self.chat_model:
                raise ValueError("CHAT_MODEL is mandatory with LLM_PROVIDER=openai")

        elif self.llm_provider == "ollama":
            if not self.ollama_base_url:
                raise ValueError(
                    "OLLAMA_BASE_URL is mandatory with LLM_PROVIDER=ollama"
                )

            if not self.embedding_model_local:
                raise ValueError(
                    "EMBEDDING_MODEL_LOCAL is mandatory with LLM_PROVIDER=ollama"
                )

            if not self.chat_model_local:
                raise ValueError(
                    "CHAT_MODEL_LOCAL is mandatory with LLM_PROVIDER=ollama"
                )

        if self.docs_dir is None:
            raise ValueError("DOCS_DIR is mandatory")
        if self.chroma_dir is None:
            raise ValueError("CHROMA_DIR is mandatory")

        return self
