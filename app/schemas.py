"""
API data schemas.

This module defines and validates the data exchanged through the API. A refusal is
a normal answer (HTTP 200) that says why it refused, not an error.
"""

from pydantic import BaseModel, Field, ConfigDict


class QueryQuestion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    number: int = Field(description="Passage number the answer cites, as [number].")
    chunk_id: str | None = Field(description="Indexed chunk the passage comes from.")
    source: str | None = Field(description="Document file of the passage.")
    article: str | None = Field(
        default=None, description="Article heading, when known."
    )
    in_force: bool = Field(
        description="False if the passage comes from a superseded version."
    )
    page: int | None = Field(default=None, description="Page number, for PDF sources.")
    distance: float = Field(
        description="Squared L2 distance to the question; lower is closer."
    )


class QueryAnswer(BaseModel):
    answer: str
    sources: list[str] = Field(
        default_factory=list, description="Documents used as context."
    )
    context_found: bool = Field(description="False when the system refused to answer.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Passages the answer cites; invented citations are removed.",
    )
    refusal_reason: str | None = Field(
        default=None,
        description="Why the system refused: no_results, below_relevance_threshold or classifier_refused.",
    )
    confidence: float | None = Field(
        default=None,
        description="Probability of answering, when the abstention classifier decided.",
    )
    guardrail_mode: str = Field(
        default="threshold", description="threshold or classifier."
    )
    best_distance: float | None = Field(
        default=None, description="Distance of the closest chunk."
    )
    latency_ms: dict[str, float] = Field(
        default_factory=dict, description="Time spent per stage."
    )
    request_id: str | None = Field(
        default=None, description="Identifier found in every log line of the request."
    )


class HealthAnswer(BaseModel):
    status: str


class ReadinessAnswer(BaseModel):
    status: str
    checks: dict[str, bool]
