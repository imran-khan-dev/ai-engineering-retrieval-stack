"""Inverted index for BM25."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from retrieval_stack.chunking.models import Chunk

from .tokenizer import Tokenizer


@dataclass 
class InvertedIndex:
    # store in memory
    tokenizer: Tokenizer
    chunks: Dict[str, Chunk] = field(default_factory=dict)
    term_to_chunks: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    term_freq: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    doc_freq: Dict[str, int] = field(default_factory=dict)
    doc_lengths: Dict[str, int] = field(default_factory=dict)
    total_docs: int = 0
    avg_doc_length: float = 0.0

    def add_chunk(self, chunk: Chunk) -> None:
        if chunk.id in self.chunks:
            raise ValueError(f"Duplicate chunk id: {chunk.id}")
        tokens = self.tokenizer.tokenize(chunk.content)
        tf = Counter(tokens)
        self.chunks[chunk.id] = chunk
        self.doc_lengths[chunk.id] = len(tokens)
        for term, freq in tf.items():
            self.term_to_chunks[term].add(chunk.id)
            self.term_freq[term][chunk.id] = freq
            self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        self.total_docs = len(self.chunks) 
        self._recompute_avg()

    def _recompute_avg(self):
        if self.total_docs == 0:
            self.avg_doc_length = 0.0
        else:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

    def build(self, chunks: List[Chunk]) -> None:
        for c in chunks:
            self.add_chunk(c)

    def get_candidates(self, query_tokens: list[str]) -> set:
        result = set()
        for t in set(query_tokens):
            # |= assigning to result
            result |= self.term_to_chunks.get(t, set())
        return result
