import pytest

from retrieval_stack.chunking import Chunk
from retrieval_stack.hybrid import HybridSearchResult
from retrieval_stack.reranking import Reranker


class FakeCrossEncoder:
    def __init__(self, scores, record=None):
        self.scores = list(scores)
        self.record = record if record is not None else {}

    def score(self, query, texts):
        self.record["query"] = query
        self.record["texts"] = list(texts)
        return list(self.scores[: len(texts)])


def make_chunk(cid, content):
    return Chunk(id=cid, document_id="d1", index=0, content=content, token_count=1, metadata={"k": cid})


def make_hybrid(cid, content, rrf=1.0, sources=None):
    chunk = make_chunk(cid, content)
    return HybridSearchResult(
        chunk_id=cid,
        chunk=chunk,
        rrf_score=rrf,
        sources=sources or ["bm25", "dense"],
        bm25_rank=1,
        dense_rank=2,
        metadata={"k": cid},
    )


def test_rerank_order_and_scores():
    h1 = make_hybrid("c1", "apple")
    h2 = make_hybrid("c2", "banana")
    rec = {}
    reranker = Reranker(FakeCrossEncoder([0.1, 0.9], rec))
    out = reranker.rerank("q", [h1, h2], top_k=2)
    assert rec["query"] == "q"
    assert rec["texts"] == ["apple", "banana"]
    assert [r.chunk_id for r in out] == ["c2", "c1"]
    assert out[0].relevance_score == 0.9
    assert out[0].rrf_score == 1.0
    assert out[0].sources == ["bm25", "dense"]
    assert out[0].bm25_rank == 1
    assert out[0].dense_rank == 2
    assert out[0].hybrid_result is h2


def test_preserves_chunk_and_metadata():
    h1 = make_hybrid("c1", "text")
    reranker = Reranker(FakeCrossEncoder([1.0]))
    out = reranker.rerank("q", [h1], top_k=1)
    assert out[0].chunk is h1.chunk
    assert out[0].metadata["k"] == "c1"


def test_empty_candidates():
    reranker = Reranker(FakeCrossEncoder([]))
    assert reranker.rerank("q", [], top_k=5) == []


def test_invalid_top_k():
    reranker = Reranker(FakeCrossEncoder([]))
    with pytest.raises(ValueError):
        reranker.rerank("q", [], top_k=0)


def test_incorrect_score_count():
    h1 = make_hybrid("c1", "a")
    h2 = make_hybrid("c2", "b")
    reranker = Reranker(FakeCrossEncoder([1.0]))
    with pytest.raises(ValueError):
        reranker.rerank("q", [h1, h2], top_k=2)


def test_tie_deterministic():
    h1 = make_hybrid("b", "b")
    h2 = make_hybrid("a", "a")
    reranker = Reranker(FakeCrossEncoder([1.0, 1.0]))
    out = reranker.rerank("q", [h1, h2], top_k=2)
    assert [r.chunk_id for r in out] == ["a", "b"]


def test_candidate_count_and_top_k():
    items = [make_hybrid(f"c{i}", f"t{i}") for i in range(5)]
    rec = {}
    reranker = Reranker(FakeCrossEncoder([0.1, 0.2, 0.3, 0.4, 0.5], rec))
    out = reranker.rerank("q", items, top_k=2, candidate_count=3)
    assert rec["texts"] == ["t0", "t1", "t2"]
    assert len(out) == 2
    assert out[0].chunk_id == "c2"


def test_batch_scoring_single_call():
    items = [make_hybrid("c1", "a"), make_hybrid("c2", "b")]
    calls = {"n": 0}

    class Counter:
        def score(self, query, texts):
            calls["n"] += 1
            return [0.2, 0.1]

    reranker = Reranker(Counter())
    reranker.rerank("q", items, top_k=2)
    assert calls["n"] == 1
