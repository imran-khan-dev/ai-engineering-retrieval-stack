import pytest

from retrieval_stack.chunking import Chunk, ChunkConfig, ChunkingService
from retrieval_stack.ingestion import DocumentIngestor
from retrieval_stack.pipeline import RetrievalPipeline


class FakeEmbedder:
    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0
        self.embed_batch_inputs = []

    def embed(self, text):
        self.single_calls += 1
        return [float(len(text)), 1.0]

    def embed_batch(self, texts):
        self.batch_calls += 1
        self.embed_batch_inputs.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


class FakeRerankerProvider:
    def __init__(self):
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        # prefer shorter text as proxy
        return [float(100 - len(t)) for t in texts]


class FixedChunkingService:
    def __init__(self, chunks):
        self.chunks = chunks

    def chunk_document(self, document):
        return list(self.chunks)


def make_txt(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_index_one_document(tmp_path):
    p = make_txt(tmp_path, "a.txt", "hello world")
    pipe = RetrievalPipeline(embedder=FakeEmbedder(), reranker=FakeRerankerProvider(), chunker=ChunkingService(ChunkConfig(chunk_size=50, chunk_overlap=0)))
    chunks = pipe.index_documents([p])
    assert len(chunks) >= 1
    assert pipe.bm25_index.total_docs == len(chunks)
    assert pipe.vector_index.size == len(chunks)
    assert set(pipe.bm25_index.chunks.keys()) == set(pipe.vector_index.embeddings().keys())


def test_index_multiple_documents(tmp_path):
    p1 = make_txt(tmp_path, "a.txt", "alpha beta")
    p2 = make_txt(tmp_path, "b.txt", "gamma delta")
    pipe = RetrievalPipeline(embedder=FakeEmbedder(), reranker=FakeRerankerProvider(), chunker=ChunkingService(ChunkConfig(chunk_size=50, chunk_overlap=0)))
    chunks = pipe.index_documents([p1, p2])
    assert pipe.bm25_index.total_docs == len(chunks)
    assert pipe.vector_index.size == len(chunks)


def test_batch_embedding_used(tmp_path):
    p1 = make_txt(tmp_path, "a.txt", "alpha beta gamma")
    p2 = make_txt(tmp_path, "b.txt", "delta epsilon zeta")
    emb = FakeEmbedder()
    pipe = RetrievalPipeline(embedder=emb, reranker=FakeRerankerProvider(), chunker=ChunkingService(ChunkConfig(chunk_size=2, chunk_overlap=0)))
    chunks = pipe.index_documents([p1, p2])
    assert emb.batch_calls == 1
    assert emb.single_calls == 0
    assert len(emb.embed_batch_inputs[0]) == len(chunks)


def test_empty_input():
    pipe = RetrievalPipeline(embedder=FakeEmbedder(), reranker=FakeRerankerProvider())
    with pytest.raises(ValueError):
        pipe.index_documents([])


def test_ingestion_failure(tmp_path):
    p = tmp_path / "missing.txt"
    pipe = RetrievalPipeline(embedder=FakeEmbedder(), reranker=FakeRerankerProvider())
    with pytest.raises(Exception):
        pipe.index_documents([p])


def test_duplicate_chunk_ids_leave_consistent(tmp_path):
    p = make_txt(tmp_path, "a.txt", "content")
    chunker = FixedChunkingService([
        Chunk(id="c1", document_id="d1", index=0, content="a", token_count=1),
        Chunk(id="c1", document_id="d2", index=1, content="b", token_count=1),
    ])
    pipe = RetrievalPipeline(embedder=FakeEmbedder(), reranker=FakeRerankerProvider(), chunker=chunker)
    with pytest.raises(ValueError):
        pipe.index_documents([p])
    assert pipe.bm25_index.total_docs == 0
    assert pipe.vector_index.size == 0


def test_failed_indexing_does_not_mutate_existing(tmp_path):
    p1 = make_txt(tmp_path, "a.txt", "alpha")
    p2 = make_txt(tmp_path, "b.txt", "beta")
    pipe = RetrievalPipeline(embedder=FakeEmbedder(), reranker=FakeRerankerProvider(), chunker=ChunkingService(ChunkConfig(chunk_size=50, chunk_overlap=0)))
    pipe.index_documents([p1])
    old_chunks = list(pipe.chunks)
    old_bm = set(pipe.bm25_index.chunks.keys())
    old_vec = set(pipe.vector_index.embeddings().keys())

    class BadChunker:
        def chunk_document(self, document):
            return [Chunk(id="dup", document_id="d", index=0, content="x", token_count=1), Chunk(id="dup", document_id="d", index=1, content="y", token_count=1)]

    pipe.chunker = BadChunker()
    with pytest.raises(ValueError):
        pipe.index_documents([p2])
    assert [c.id for c in pipe.chunks] == [c.id for c in old_chunks]
    assert set(pipe.bm25_index.chunks.keys()) == old_bm
    assert set(pipe.vector_index.embeddings().keys()) == old_vec


def test_search_after_index(tmp_path):
    p = make_txt(tmp_path, "a.txt", "recover account access")
    pipe = RetrievalPipeline(embedder=FakeEmbedder(), reranker=FakeRerankerProvider(), chunker=ChunkingService(ChunkConfig(chunk_size=50, chunk_overlap=0)))
    pipe.index_documents([p])
    res = pipe.search("recover account", top_k=2)
    assert res.query == "recover account"
    assert len(res.results) >= 1
    assert hasattr(res.results[0], "relevance_score")
    assert hasattr(res.results[0], "rrf_score")


def test_search_empty_index():
    pipe = RetrievalPipeline(embedder=FakeEmbedder(), reranker=FakeRerankerProvider())
    res = pipe.search("q")
    assert res.results == []
    assert res.query == "q"


def test_repeated_queries_no_reembedding(tmp_path):
    p = make_txt(tmp_path, "a.txt", "hello world hello world")
    emb = FakeEmbedder()
    pipe = RetrievalPipeline(embedder=emb, reranker=FakeRerankerProvider(), chunker=ChunkingService(ChunkConfig(chunk_size=50, chunk_overlap=0)))
    pipe.index_documents([p])
    b = emb.batch_calls
    s = emb.single_calls
    pipe.search("hello", top_k=1)
    pipe.search("hello", top_k=1)
    assert emb.batch_calls == b
    assert emb.single_calls == s + 2  # one query embed per search


def test_hybrid_and_reranker_called(tmp_path):
    p = make_txt(tmp_path, "a.txt", "alpha beta gamma delta")
    class SpyHybrid:
        def __init__(self, inner):
            self.inner = inner
            self.calls = []
        def search(self, query, top_k=5, bm25_top_k=None, dense_top_k=None):
            self.calls.append((query, top_k, bm25_top_k, dense_top_k))
            return self.inner.search(query, top_k=top_k, bm25_top_k=bm25_top_k, dense_top_k=dense_top_k)

    pipe = RetrievalPipeline(embedder=FakeEmbedder(), reranker=FakeRerankerProvider(), chunker=ChunkingService(ChunkConfig(chunk_size=50, chunk_overlap=0)))
    pipe.index_documents([p])
    spy = SpyHybrid(pipe.hybrid_searcher)
    pipe.hybrid_searcher = spy
    pipe.search("alpha", bm25_top_k=3, dense_top_k=4, hybrid_top_k=5, top_k=2)
    assert spy.calls == [("alpha", 5, 3, 4)]
