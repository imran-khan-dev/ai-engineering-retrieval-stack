"""Reranking models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from retrieval_stack.chunking.models import Chunk
from retrieval_stack.hybrid.models import HybridSearchResult


class RerankedResult(BaseModel):
    chunk_id: str
    chunk: Chunk
    relevance_score: float
    rrf_score: float
    sources: list[str] = Field(default_factory=list)
    bm25_rank: int | None = None
    dense_rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    hybrid_result: HybridSearchResult | None = None
