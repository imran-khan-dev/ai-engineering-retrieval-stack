"""Chunking strategies."""

from __future__ import annotations

import uuid
from typing import List

from .config import ChunkConfig
from .models import Chunk
from .tokenizer import Tokenizer


class FixedSizeChunker:
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.tokenizer = Tokenizer(config.encoding)

    def chunk(
        self,
        document_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> List[Chunk]:

        metadata = metadata or {}

        if not text.strip():
            return []

        tokens, offsets = self.tokenizer.encode_with_offsets(text)

        step = self.config.chunk_size - self.config.chunk_overlap

        chunks = []

        for start in range(0, len(tokens), step):
            end = min(start + self.config.chunk_size, len(tokens))

            piece_tokens = tokens[start:end]

            if not piece_tokens:
                break

            # Map token boundaries to original character positions.
            start_char = offsets[start]

            if end == len(tokens):
                end_char = len(text)
            else:
                end_char = offsets[end]

                # Checks whether we still have another token to inspect.
                # And does the next token start at the same character 
                # position as the previous token, or even before it (for safety).

                while end < len(tokens) and offsets[end] <= offsets[end - 1]:
                    end += 1

                end_char = offsets[end] if end < len(tokens) else len(text)

            # Extract directly from the original text.
            piece = text[start_char:end_char]

            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    index=len(chunks),
                    content=piece,
                    token_count=len(piece_tokens),
                    metadata=dict(metadata),
                    start_char=start_char,
                    end_char=end_char,
                )
            )

            if end >= len(tokens):
                break

        return chunks

class RecursiveChunker:
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.tokenizer = Tokenizer(config.encoding)

        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if config.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def _hard_split(
        self,
        text: str,
        base_offset: int,
    ) -> list[tuple[int, int]]:
        """Split text by token count while preserving source character positions."""

        tokens, offsets = self.tokenizer.encode_with_offsets(text)

        if not tokens:
            return []

        chunk_size = self.config.chunk_size
        step = chunk_size - self.config.chunk_overlap

        spans = []

        for token_start in range(0, len(tokens), step):
            token_end = min(token_start + chunk_size, len(tokens))

            # Find the character boundary for the starting token.
            start_char = offsets[token_start]

            # Avoid starting in the middle of a multi-token character.
            while (
                token_start > 0
                and offsets[token_start - 1] == start_char
            ):
                token_start -= 1

            # Find the character boundary for the ending token.
            if token_end == len(tokens):
                end_char = len(text)
            else:
                # Avoid ending in the middle of a multi-token character.
                while (
                    token_end < len(tokens)
                    and offsets[token_end] == offsets[token_end - 1]
                ):
                    token_end += 1

                end_char = offsets[token_end]

            if end_char > start_char:
                spans.append((
                    base_offset + start_char,
                    base_offset + end_char,
                ))

            if token_end >= len(tokens):
                break

        return spans

    def _split(
        self,
        text: str,
        separators: list[str],
        base_offset: int = 0,
    ) -> list[tuple[int, int]]:
        """Recursively split text and return absolute character spans."""

        if not text:
            return []

        # This text already fits within the configured token limit.
        if len(self.tokenizer.encode(text)) <= self.config.chunk_size:
            return [(base_offset, base_offset + len(text))]

        # No separators remain. Use token-based splitting.
        if not separators:
            return self._hard_split(text, base_offset)

        sep = separators[0]
        rest = separators[1:]

        # Skip empty separators to avoid an infinite splitting loop.
        if not sep:
            return self._split(text, rest, base_offset)

        # Create segments using their original positions.
        # Each segment includes its trailing separator.
        segments = []
        cursor = 0

        while True:
            sep_pos = text.find(sep, cursor)

            if sep_pos == -1:
                break

            segment_end = sep_pos + len(sep)
            segments.append((cursor, segment_end))
            cursor = segment_end

        if cursor < len(text):
            segments.append((cursor, len(text)))

        if not segments:
            return []

        result = []
        current_start = None
        current_end = None

        for part_start, part_end in segments:
            candidate_start = (
                current_start if current_start is not None else part_start
            )

            candidate = text[candidate_start:part_end]

            if len(self.tokenizer.encode(candidate)) <= self.config.chunk_size:
                if current_start is None:
                    current_start = part_start

                current_end = part_end

            else:
                # Save the previous accumulated chunk.
                if current_start is not None:
                    result.append((
                        base_offset + current_start,
                        base_offset + current_end,
                    ))

                    current_start = None
                    current_end = None

                part = text[part_start:part_end]

                # The individual segment is still too large.
                if len(self.tokenizer.encode(part)) > self.config.chunk_size:
                    result.extend(
                        self._split(
                            part,
                            rest,
                            base_offset + part_start,
                        )
                    )
                else:
                    current_start = part_start
                    current_end = part_end

        # Save the final accumulated chunk.
        if current_start is not None:
            result.append((
                base_offset + current_start,
                base_offset + current_end,
            ))

        return result

    def _apply_overlap(
        self,
        text: str,
        spans: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Extend each chunk's start to include the previous chunk's token tail."""

        overlap = self.config.chunk_overlap

        if overlap == 0 or len(spans) <= 1:
            return spans

        overlapped = [spans[0]]

        for i in range(1, len(spans)):
            previous_start, previous_end = spans[i - 1]
            current_start, current_end = spans[i]

            previous_text = text[previous_start:previous_end]
            tokens, offsets = self.tokenizer.encode_with_offsets(previous_text)

            if not tokens:
                overlapped.append((current_start, current_end))
                continue

            tail_start = max(0, len(tokens) - overlap)
            tail_char_start = offsets[tail_start]

            # Keep the overlap boundary at the start of a character.
            while (
                tail_start > 0
                and offsets[tail_start - 1] == tail_char_start
            ):
                tail_start -= 1
                tail_char_start = offsets[tail_start]

            overlap_start = previous_start + tail_char_start

            overlapped.append((
                min(current_start, overlap_start),
                current_end,
            ))

        return overlapped

    def chunk(
        self,
        document_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> List[Chunk]:

        metadata = metadata or {}

        if not text.strip():
            return []

        # Step 1: Recursive splitting with source positions.
        spans = self._split(text, self.config.separators)

        # Step 2: Apply overlap while preserving source positions.
        spans = self._apply_overlap(text, spans)

        # Step 3: Construct chunks directly from original text.
        chunks = []

        for start_char, end_char in spans:
            content = text[start_char:end_char]

            if not content.strip():
                continue

            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    index=len(chunks),
                    content=content,
                    token_count=len(self.tokenizer.encode(content)),
                    metadata=dict(metadata),
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        return chunks