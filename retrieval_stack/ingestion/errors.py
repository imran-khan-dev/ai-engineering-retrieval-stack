"""Document ingestion errors."""


class IngestionError(Exception):
    """Base error for document ingestion."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when a file type is not supported."""


class FileValidationError(IngestionError):
    """Raised when a file is missing, empty, or unreadable."""


class ExtractionError(IngestionError):
    """Raised when text extraction fails."""
