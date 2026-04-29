"""Извлечение текста из загруженных файлов."""
from app.ingestion.exceptions import (
    CorruptFileError,
    EmptyTextError,
    ParserError,
    UnsupportedFormatError,
)
from app.ingestion.parser import parse_file

__all__ = [
    "CorruptFileError",
    "EmptyTextError",
    "ParserError",
    "UnsupportedFormatError",
    "parse_file",
]