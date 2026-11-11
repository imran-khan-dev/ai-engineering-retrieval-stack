"""Retrieval pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from retrieval_stack.bm25 import BM25Searcher, InvertedIndex
from retrieval_stack.bm25 import Tokenizer as BM25Tokenizer
from retrieval_stack.chunking import Chunk, ChunkConfig, ChunkingService
from retrieval_stack.dense import DenseSearcher, EmbeddingProvider, VectorIndex
from retrieval_stack.hybrid import HybridSearcher
from retrieval_stack.ingestion import DocumentIngestor
from retrieval_stack.reranking import Reranker, RerankerProvider

from .models import SearchResponse


class RetrievalPipeline:
    def __init__(
        self,
        ingestor: DocumentIngestor | None = None,
        chunker: ChunkingService | None = None,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | RerankerProvider | None = None,
        bm25_tokenizer: BM25Tokenizer | None = None,
        chunk_config: ChunkConfig | None = None,
    ):
        self.ingestor = ingestor or DocumentIngestor()
        self.chunker = chunker or ChunkingService(chunk_config or ChunkConfig(chunk_size=200, chunk_overlap=20))
        self.embedder = embedder
        if reranker is None:
            self.reranker = None
        elif isinstance(reranker, Reranker):
            self.reranker = reranker
        else:
            self.reranker = Reranker(reranker)

        self.bm25_tokenizer = bm25_tokenizer or BM25Tokenizer()

        self.chunks: list[Chunk] = []
        self.bm25_index = InvertedIndex(tokenizer=self.bm25_tokenizer)
        self.vector_index = VectorIndex()
        self.bm25_searcher = BM25Searcher(self.bm25_index)
        self.dense_searcher = DenseSearcher(self.vector_index, self.embedder) if self.embedder else None
        self.hybrid_searcher = HybridSearcher(self.bm25_searcher, self.dense_searcher) if self.dense_searcher else None

    def index_documents(self, sources: Iterable[str | Path]) -> list[Chunk]:
        source_list = list(sources)
        if not source_list:
            raise ValueError("no documents provided")

        # Stage indexing to avoid mutating live indexes on failure.
        new_chunks: list[Chunk] = []
        seen_chunk_ids: set[str] = set()
        seen_doc_ids: set[str] = set()

        for src in source_list:
            doc = self.ingestor.ingest(src)
            if doc.id in seen_doc_ids:
                raise ValueError(f"Duplicate document id: {doc.id}")
            seen_doc_ids.add(doc.id)
            doc_chunks = self.chunker.chunk_document(doc)
            for ch in doc_chunks:
                if ch.id in seen_chunk_ids:
                    raise ValueError(f"Duplicate chunk id: {ch.id}")
                seen_chunk_ids.add(ch.id)
                new_chunks.append(ch)

        texts = [c.content for c in new_chunks]

        if self.embedder is None:
            raise ValueError("embedder is required for indexing")
        
        embeddings = self.embedder.embed_batch(texts) if texts else []

        bm25_index = InvertedIndex(tokenizer=self.bm25_tokenizer)
        vector_index = VectorIndex()
        bm25_index.build(new_chunks)
        vector_index.build(new_chunks, embeddings)

        # Only replace state after all steps succeed.
        self.chunks = list(new_chunks)
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.bm25_searcher = BM25Searcher(self.bm25_index)
        self.dense_searcher = DenseSearcher(self.vector_index, self.embedder)
        self.hybrid_searcher = HybridSearcher(self.bm25_searcher, self.dense_searcher)
        
        return list(self.chunks)

    def search(
        self,
        query: str,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        hybrid_top_k: int = 20,
        top_k: int = 5,
    ) -> SearchResponse:
        if self.hybrid_searcher is None or self.reranker is None:
            raise ValueError("pipeline is not indexed")
        if self.bm25_index.total_docs == 0 or self.vector_index.size == 0:
            return SearchResponse(query=query, results=[])
        hybrid_results = self.hybrid_searcher.search(
            query,
            top_k=hybrid_top_k,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
        )
        reranked = self.reranker.rerank(query, hybrid_results, top_k=top_k, candidate_count=hybrid_top_k)
        return SearchResponse(query=query, results=reranked)
