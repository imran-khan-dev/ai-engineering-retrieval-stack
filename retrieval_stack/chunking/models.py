"""Chunk model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    document_id: str
    index: int
    content: str
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    start_char: int | None = None
    end_char: int | None = None
