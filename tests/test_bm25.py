import math

import pytest

from retrieval_stack.bm25 import BM25Scorer, BM25Searcher, InvertedIndex, Tokenizer
from retrieval_stack.chunking import Chunk


def make_chunk(cid, doc_id, content):
    return Chunk(id=cid, document_id=doc_id, index=0, content=content, token_count=0, metadata={"src": doc_id})


def test_tokenizer():
    t = Tokenizer()
    assert t.tokenize("Hello, World! 123") == ["hello", "world", "123"]
    assert t.tokenize("") == []
    assert t.tokenize("café") == ["café"]


def test_index_basics():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "the cat sat"))
    assert idx.total_docs == 1
    assert idx.doc_lengths["c1"] == 3
    assert idx.avg_doc_length == 3.0
    assert idx.doc_freq["cat"] == 1
    assert idx.term_freq["cat"]["c1"] == 1


def test_duplicate_chunk_id():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "a"))
    with pytest.raises(ValueError):
        idx.add_chunk(make_chunk("c1", "d2", "b"))


def test_idf_manual():
    scorer = BM25Scorer()
    # N=2, df=1
    assert scorer.idf(2, 1) == pytest.approx(math.log(1 + (2 - 1 + 0.5) / (1 + 0.5)))


def test_scoring_manual():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "cat cat dog"))
    idx.add_chunk(make_chunk("c2", "d2", "cat"))
    scorer = BM25Scorer(k1=1.5, b=0.75)
    # manual for c1 query "cat"
    n = 2
    df = 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    avgdl = (3 + 1) / 2
    dl = 3
    tf = 2
    denom = tf + 1.5 * (1 - 0.75 + 0.75 * (dl / avgdl))
    expected = idf * (tf * 2.5) / denom
    assert scorer.score(idx, ["cat"], "c1") == pytest.approx(expected)


def test_ranking_order():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "python python python"))
    idx.add_chunk(make_chunk("c2", "d2", "python java"))
    searcher = BM25Searcher(idx)
    res = searcher.search("python", top_k=2)
    assert res[0].chunk_id == "c1"
    assert res[0].score > res[1].score


def test_length_normalization():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("short", "d", "python"))
    idx.add_chunk(make_chunk("long", "d", "python " + "x " * 50))
    searcher = BM25Searcher(idx)
    res = searcher.search("python", top_k=2)
    # short doc should score higher due to length norm
    assert res[0].chunk_id == "short"


def test_tf_saturation():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("a", "d", "term " * 1))
    idx.add_chunk(make_chunk("b", "d", "term " * 10))
    scorer1 = BM25Scorer(k1=0.5, b=0.0)
    # with low k1, extra tf less impact; but b=0 no length norm. Just ensure finite
    s1 = scorer1.score(idx, ["term"], "a")
    s2 = scorer1.score(idx, ["term"], "b")
    assert s2 > s1


def test_multiple_query_terms():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "cat dog"))
    idx.add_chunk(make_chunk("c2", "d2", "cat"))
    searcher = BM25Searcher(idx)
    res = searcher.search("cat dog", top_k=2)
    assert res[0].chunk_id == "c1"


def test_repeated_query_terms():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "cat"))
    searcher = BM25Searcher(idx)
    r1 = searcher.search("cat", top_k=1)
    r2 = searcher.search("cat cat", top_k=1)
    # repeated query terms do not increase score
    assert r2[0].score == pytest.approx(r1[0].score)


def test_empty_query():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "cat"))
    searcher = BM25Searcher(idx)
    assert searcher.search("", top_k=5) == []
    assert searcher.search("!!!", top_k=5) == []


def test_no_matching_terms():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "cat"))
    searcher = BM25Searcher(idx)
    assert searcher.search("zzzzz", top_k=5) == []


def test_top_k():
    idx = InvertedIndex(tokenizer=Tokenizer())
    for i in range(5):
        idx.add_chunk(make_chunk(f"c{i}", "d", "apple"))
    searcher = BM25Searcher(idx)
    res = searcher.search("apple", top_k=3)
    assert len(res) == 3


def test_empty_index():
    idx = InvertedIndex(tokenizer=Tokenizer())
    searcher = BM25Searcher(idx)
    assert searcher.search("anything", top_k=5) == []


def test_metadata_traceability():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "hello"))
    searcher = BM25Searcher(idx)
    res = searcher.search("hello", top_k=1)
    assert res[0].chunk_id == "c1"
    assert res[0].document_id == "d1"
    assert res[0].metadata["src"] == "d1"
    assert res[0].chunk.id == "c1"


def test_unicode():
    idx = InvertedIndex(tokenizer=Tokenizer())
    idx.add_chunk(make_chunk("c1", "d1", "cafe latte"))
    searcher = BM25Searcher(idx)
    res = searcher.search("cafe", top_k=1)
    assert len(res) == 1
