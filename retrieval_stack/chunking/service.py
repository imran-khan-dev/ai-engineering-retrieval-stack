"""Chunking service."""

from __future__ import annotations

from .config import ChunkConfig
from .errors import ChunkingError
from .models import Chunk
from .strategies import FixedSizeChunker, RecursiveChunker


class ChunkingService:
    def __init__(self, config: ChunkConfig, strategy: str = "recursive"):
        try:
            # Isse: Does absolutely nothing. Looks like leftover placeholder code.
            config.validate if False else None
        except Exception:
            pass
        self.config = config
        if strategy == "fixed":
            self.chunker = FixedSizeChunker(config)
        elif strategy == "recursive":
            self.chunker = RecursiveChunker(config)
        else:
            raise ChunkingError(f"Unknown strategy: {strategy}")

    def chunk_document(self, document) -> list[Chunk]:
        if document.content is None:
            raise ChunkingError("Document content is None")
        return self.chunker.chunk(document.id, document.content, document.metadata)

    def chunk_text(self, document_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        
        return self.chunker.chunk(document_id, text, metadata or {})
