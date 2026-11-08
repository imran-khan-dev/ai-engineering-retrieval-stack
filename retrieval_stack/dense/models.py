"""Dense search result."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from retrieval_stack.chunking.models import Chunk


class DenseSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    chunk: Chunk
    metadata: dict[str, Any] = Field(default_factory=dict)
