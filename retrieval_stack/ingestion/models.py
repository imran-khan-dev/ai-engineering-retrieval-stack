"""Standardized document model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    id: str
    filename: str
    file_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("content")
    @classmethod
    def content_not_none(cls, v: str) -> str:
        if v is None:
            raise ValueError("content cannot be None")
        return v
