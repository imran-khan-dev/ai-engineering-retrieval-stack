"""Document loading: file validation and reading."""

from __future__ import annotations

from pathlib import Path

from .errors import FileValidationError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def validate_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileValidationError(f"File not found: {path}")
    if not p.is_file():
        raise FileValidationError(f"Not a file: {path}")
    if p.stat().st_size == 0:
        raise FileValidationError(f"File is empty: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file type: {p.suffix}")
    try:
        with p.open("rb") as f:
            f.read(1)
    except OSError as e:
        raise FileValidationError(f"Unreadable file: {path}") from e
    return p


def read_bytes(path: str | Path) -> bytes:
    p = validate_path(path)
    try:
        return p.read_bytes()
    except OSError as e:
        raise FileValidationError(f"Unreadable file: {path}") from e
