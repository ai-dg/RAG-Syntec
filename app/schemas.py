"""
API data schemas.

This module defines and validates the data exchanged through the API.
"""

from pydantic import BaseModel, Field, ConfigDict


class QueryQuestion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


class QueryAnswer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    context_found: bool


class HealthAnswer(BaseModel):
    status: str
