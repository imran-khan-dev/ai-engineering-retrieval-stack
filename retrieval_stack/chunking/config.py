"""Chunking errors and config."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ChunkingError(Exception):
    pass

class ChunkConfig(BaseModel):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    encoding: str = "cl100k_base"
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self
