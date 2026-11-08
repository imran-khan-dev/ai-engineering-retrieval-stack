import pytest

from retrieval_stack.chunking import Chunk
from retrieval_stack.dense import DenseSearcher, VectorIndex, cosine_similarity


class FakeEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping

    def embed(self, text):
        return list(self.mapping[text])

    def embed_batch(self, texts):
        return [list(self.mapping[t]) for t in texts]


def make_chunk(cid, doc, content):
    return Chunk(id=cid, document_id=doc, index=0, content=content, token_count=1)


def test_cosine_known():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_zero():
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_cosine_dim_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_index_duplicate():
    idx = VectorIndex()
    idx.add(make_chunk("c1", "d", "a"), [1.0, 0.0])
    with pytest.raises(ValueError):
        idx.add(make_chunk("c1", "d", "b"), [0.0, 1.0])


def test_index_dim_mismatch():
    idx = VectorIndex()
    idx.add(make_chunk("c1", "d", "a"), [1.0, 0.0])
    with pytest.raises(ValueError):
        idx.add(make_chunk("c2", "d", "b"), [1.0, 0.0, 0.0])


def test_search_ranking():
    idx = VectorIndex()
    idx.add(make_chunk("c1", "d", "a"), [1.0, 0.0])
    idx.add(make_chunk("c2", "d", "b"), [0.0, 1.0])
    emb = FakeEmbedder({"q": [1.0, 0.0]})
    searcher = DenseSearcher(idx, emb)
    res = searcher.search("q", top_k=2)
    assert res[0].chunk_id == "c1"
    assert res[0].score > res[1].score


def test_top_k():
    idx = VectorIndex()
    for i in range(5):
        idx.add(make_chunk(f"c{i}", "d", "x"), [1.0, 0.0])
    emb = FakeEmbedder({"q": [1.0, 0.0]})
    searcher = DenseSearcher(idx, emb)
    res = searcher.search("q", top_k=3)
    assert len(res) == 3


def test_empty_index():
    idx = VectorIndex()
    emb = FakeEmbedder({"q": [1.0, 0.0]})
    searcher = DenseSearcher(idx, emb)
    assert searcher.search("q", top_k=5) == []


def test_invalid_top_k():
    idx = VectorIndex()
    emb = FakeEmbedder({"q": [1.0, 0.0]})
    searcher = DenseSearcher(idx, emb)
    with pytest.raises(ValueError):
        searcher.search("q", top_k=0)


def test_tie_deterministic():
    idx = VectorIndex()
    idx.add(make_chunk("b", "d", "x"), [1.0, 0.0])
    idx.add(make_chunk("a", "d", "x"), [1.0, 0.0])
    emb = FakeEmbedder({"q": [1.0, 0.0]})
    searcher = DenseSearcher(idx, emb)
    res = searcher.search("q", top_k=2)
    assert [r.chunk_id for r in res] == ["a", "b"]


def test_metadata_traceability():
    c = make_chunk("c1", "d1", "hello")
    c.metadata["k"] = "v"
    idx = VectorIndex()
    idx.add(c, [1.0, 0.0])
    emb = FakeEmbedder({"q": [1.0, 0.0]})
    searcher = DenseSearcher(idx, emb)
    res = searcher.search("q", top_k=1)
    assert res[0].document_id == "d1"
    assert res[0].metadata["k"] == "v"
    assert res[0].chunk.id == "c1"
