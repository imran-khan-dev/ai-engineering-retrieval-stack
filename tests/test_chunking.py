import pytest

from retrieval_stack.chunking import Chunk, ChunkConfig, ChunkingService
from retrieval_stack.ingestion import Document


def make_config(size=10, overlap=2):
    return ChunkConfig(chunk_size=size, chunk_overlap=overlap)


def test_short_document_single_chunk():
    cfg = make_config(100, 0)
    svc = ChunkingService(cfg, "fixed")
    chunks = svc.chunk_text("doc1", "Hello world.")
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world."
    assert chunks[0].document_id == "doc1"


def test_empty_content_returns_no_chunks():
    cfg = make_config(10, 2)
    svc = ChunkingService(cfg, "recursive")
    assert svc.chunk_text("doc1", "") == []
    assert svc.chunk_text("doc1", "   ") == []


def test_fixed_overlap_correctness():
    text = " ".join(["word"] * 30)
    cfg = make_config(10, 2)
    svc = ChunkingService(cfg, "fixed")
    chunks = svc.chunk_text("d", text)
    assert len(chunks) >= 2
    # overlap: last 2 tokens of chunk i appear at start of chunk i+1
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    t0 = enc.encode(chunks[0].content)[-2:]
    t1 = enc.encode(chunks[1].content)[:2]
    assert t0 == t1


def test_exact_boundary():
    text = " ".join(["a"] * 10)
    cfg = make_config(10, 0)
    svc = ChunkingService(cfg, "fixed")
    chunks = svc.chunk_text("d", text)
    assert len(chunks) == 1


def test_recursive_splitting():
    text = "Para one.\n\nPara two.\n\nPara three."
    cfg = make_config(5, 0)
    svc = ChunkingService(cfg, "recursive")
    chunks = svc.chunk_text("d", text)
    assert len(chunks) >= 2
    assert all(c.content.strip() for c in chunks)


def test_oversized_paragraph():
    text = "This is a very long sentence without separators " * 20
    cfg = make_config(8, 0)
    svc = ChunkingService(cfg, "recursive")
    chunks = svc.chunk_text("d", text)
    assert all(len(__import__("tiktoken").get_encoding("cl100k_base").encode(c.content)) <= 8 for c in chunks)


def test_metadata_preservation():
    doc = Document(id="doc99", filename="f.txt", file_type="txt", content="Hello world foo bar", metadata={"page_count": 1})
    cfg = make_config(100, 0)
    svc = ChunkingService(cfg, "recursive")
    chunks = svc.chunk_document(doc)
    assert chunks[0].metadata["page_count"] == 1
    assert chunks[0].document_id == "doc99"


def test_invalid_config():
    with pytest.raises(Exception):
        ChunkConfig(chunk_size=5, chunk_overlap=5)
    with pytest.raises(Exception):
        ChunkConfig(chunk_size=0, chunk_overlap=0)


def test_no_lost_text_fixed():
    text = "one two three four five six seven eight nine ten"
    cfg = make_config(4, 1)
    svc = ChunkingService(cfg, "fixed")
    chunks = svc.chunk_text("d", text)
    # reconstruct unique tokens
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    all_tokens = []
    for c in chunks:
        all_tokens.extend(enc.encode(c.content))
    # ensure all original tokens present
    orig = enc.encode(text)
    for t in orig:
        assert t in all_tokens


def test_chunk_ids_unique():
    text = "a " * 50
    cfg = make_config(5, 1)
    svc = ChunkingService(cfg, "recursive")
    chunks = svc.chunk_text("d", text)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
