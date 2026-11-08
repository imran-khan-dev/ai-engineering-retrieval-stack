"""In-memory vector index."""

from __future__ import annotations

from typing import Dict, List

from retrieval_stack.chunking.models import Chunk


class VectorIndex:
    def __init__(self, dimension: int | None = None):
        self.dimension = dimension
        self._chunks: Dict[str, Chunk] = {}
        self._embeddings: Dict[str, list[float]] = {}

    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        if chunk.id in self._chunks:
            raise ValueError(f"Duplicate chunk id: {chunk.id}")
        if self.dimension is None:
            self.dimension = len(embedding)
        if len(embedding) != self.dimension:
            raise ValueError("embedding dimension mismatch")
        if len(embedding) == 0:
            raise ValueError("embedding must not be empty")
        self._chunks[chunk.id] = chunk
        self._embeddings[chunk.id] = list(embedding)

    def build(self, chunks: List[Chunk], embeddings: List[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        for c, e in zip(chunks, embeddings):
            self.add(c, e)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def items(self):
        return list(self._chunks.items())

    def embeddings(self):
        return dict(self._embeddings)
