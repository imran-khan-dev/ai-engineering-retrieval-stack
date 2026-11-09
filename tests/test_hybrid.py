import pytest

from retrieval_stack.chunking import Chunk
from retrieval_stack.hybrid import HybridSearcher, reciprocal_rank_fusion


class R:
    def __init__(self, chunk_id, chunk):
        self.chunk_id = chunk_id
        self.chunk = chunk


def make_chunk(cid, doc="d"):
    return Chunk(id=cid, document_id=doc, index=0, content=cid, token_count=1, metadata={"k": cid})


def test_rrf_both_lists():
    c1 = make_chunk("c1")
    c2 = make_chunk("c2")
    out = reciprocal_rank_fusion([[R("c1", c1)], [R("c1", c1)]], source_names=["bm25", "dense"], k=60)
    assert out[0].chunk_id == "c1"
    assert out[0].sources == ["bm25", "dense"]
    assert out[0].rrf_score == pytest.approx(1 / 61 + 1 / 61)


def test_rrf_single_source_bm25():
    c1 = make_chunk("c1")
    out = reciprocal_rank_fusion([[R("c1", c1)], []], source_names=["bm25", "dense"], k=60)
    assert out[0].chunk_id == "c1"
    assert out[0].sources == ["bm25"]
    assert out[0].bm25_rank == 1
    assert out[0].dense_rank is None


def test_rrf_single_source_dense():
    c1 = make_chunk("c1")
    out = reciprocal_rank_fusion([[], [R("c1", c1)]], source_names=["bm25", "dense"], k=60)
    assert out[0].sources == ["dense"]
    assert out[0].dense_rank == 1


def test_rrf_combined_ranking():
    c1, c2, c3 = make_chunk("c1"), make_chunk("c2"), make_chunk("c3")
    out = reciprocal_rank_fusion([[R("c1", c1), R("c2", c2)], [R("c2", c2), R("c3", c3)]], source_names=["bm25", "dense"], k=60)
    assert out[0].chunk_id == "c2"
    assert out[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_duplicate_in_input():
    c1 = make_chunk("c1")
    out = reciprocal_rank_fusion([[R("c1", c1), R("c1", c1)]], source_names=["bm25"], k=60)
    assert out[0].rrf_score == pytest.approx(1 / 61)


def test_rrf_tie_deterministic():
    c1, c2 = make_chunk("b"), make_chunk("a")
    out = reciprocal_rank_fusion([[R("b", c1)], [R("a", c2)]], source_names=["bm25", "dense"], k=60)
    assert [r.chunk_id for r in out] == ["a", "b"]


def test_rrf_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[]], k=0)


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], source_names=["bm25", "dense"]) == []


def test_rrf_preserves_chunk_metadata():
    c1 = make_chunk("c1")
    out = reciprocal_rank_fusion([[R("c1", c1)]], source_names=["bm25"], k=60)
    assert out[0].chunk is c1
    assert out[0].metadata["k"] == "c1"


class FakeSearcher:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_hybrid_calls_both():
    c1 = make_chunk("c1")
    bm = FakeSearcher([R("c1", c1)])
    de = FakeSearcher([R("c1", c1)])
    h = HybridSearcher(bm, de)
    out = h.search("hello", top_k=3, bm25_top_k=2, dense_top_k=4)
    assert bm.calls == [("hello", 2)]
    assert de.calls == [("hello", 4)]
    assert out[0].chunk_id == "c1"


def test_hybrid_final_top_k():
    chunks = [make_chunk(f"c{i}") for i in range(5)]
    bm = FakeSearcher([R(c.id, c) for c in chunks])
    de = FakeSearcher([])
    h = HybridSearcher(bm, de)
    out = h.search("q", top_k=2)
    assert len(out) == 2


def test_hybrid_empty_bm25():
    c1 = make_chunk("c1")
    h = HybridSearcher(FakeSearcher([]), FakeSearcher([R("c1", c1)]))
    out = h.search("q")
    assert out[0].sources == ["dense"]


def test_hybrid_empty_dense():
    c1 = make_chunk("c1")
    h = HybridSearcher(FakeSearcher([R("c1", c1)]), FakeSearcher([]))
    out = h.search("q")
    assert out[0].sources == ["bm25"]


def test_hybrid_both_empty():
    h = HybridSearcher(FakeSearcher([]), FakeSearcher([]))
    assert h.search("q") == []
