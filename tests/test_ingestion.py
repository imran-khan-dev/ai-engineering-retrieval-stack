import fitz
import pytest

from retrieval_stack.ingestion import Document, DocumentIngestor
from retrieval_stack.ingestion.errors import (
    ExtractionError,
    FileValidationError,
    UnsupportedFileTypeError,
)
from retrieval_stack.ingestion.loader import validate_path
from retrieval_stack.ingestion.normalizer import normalize_text


def make_txt(tmp_path, content, name="sample.txt"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def make_pdf(tmp_path, text="Hello world.\n\nSecond paragraph.", name="sample.pdf"):
    p = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    doc.save(str(p))
    doc.close()
    return p


def test_txt_ingestion(tmp_path):
    p = make_txt(tmp_path, "Hello   world\n\n\n\nSecond line")
    doc = DocumentIngestor().ingest(p)
    assert isinstance(doc, Document)
    assert doc.file_type == "txt"
    assert "Hello world" in doc.content
    assert "\n\n\n" not in doc.content


def test_pdf_ingestion(tmp_path):
    p = make_pdf(tmp_path)
    doc = DocumentIngestor().ingest(p)
    assert doc.file_type == "pdf"
    assert "Hello world" in doc.content
    assert doc.metadata.get("page_count") == 1


def test_normalization():
    assert normalize_text("a\r\nb\r\nc") == "a\nb\nc"
    assert normalize_text("x   y") == "x y"
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_invalid_file_type(tmp_path):
    p = tmp_path / "x.docx"
    p.write_text("x")
    with pytest.raises(UnsupportedFileTypeError):
        DocumentIngestor().ingest(p)


def test_missing_file(tmp_path):
    with pytest.raises(FileValidationError):
        DocumentIngestor().ingest(tmp_path / "nope.txt")


def test_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(FileValidationError):
        DocumentIngestor().ingest(p)


def test_unreadable_file(tmp_path, monkeypatch):
    p = make_txt(tmp_path, "hi")
    monkeypatch.setattr("pathlib.Path.open", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(FileValidationError):
        validate_path(p)


def test_extraction_failure_pdf(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not a pdf")
    with pytest.raises(ExtractionError):
        DocumentIngestor().ingest(p)


def test_empty_pdf(tmp_path):
    p = make_pdf(tmp_path, text="", name="empty.pdf")
    with pytest.raises(ExtractionError):
        DocumentIngestor().ingest(p)


def test_scanned_pdf_no_text(tmp_path):
    # PDF page with no text layer (image-only simulation: no inserted text)
    p = make_pdf(tmp_path, text="", name="scanned.pdf")
    with pytest.raises(ExtractionError):
        DocumentIngestor().ingest(p)


def test_empty_extraction_result(tmp_path):
    p = make_txt(tmp_path, "   \n\n   ")
    with pytest.raises(ExtractionError):
        DocumentIngestor().ingest(p)


def test_corrupt_txt_unreadable(tmp_path):
    p = tmp_path / "corrupt.txt"
    p.write_bytes(b"\xff\xfe\x00\x00invalid")
    # Should either extract (latin-1 fallback) or raise ExtractionError, never silently succeed empty
    try:
        doc = DocumentIngestor().ingest(p)
        assert doc.content.strip() != ""
    except ExtractionError:
        pass

