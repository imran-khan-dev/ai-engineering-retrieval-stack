"""Retrieval pipeline models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from retrieval_stack.reranking.models import RerankedResult


class SearchResponse(BaseModel):
    query: str
    results: list[RerankedResult] = Field(default_factory=list)
