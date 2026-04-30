"""Извлечение текста из загруженных файлов и подготовка к индексации.

Здесь намеренно не реэкспортируется ``IngestionService`` из pipeline.py:
тот тащит за собой ``app.models`` и ``app.storage.database``, а они на
импорте инициализируют ``Settings()`` и движок SQLAlchemy. Лёгкие
утилиты (парсер, чанкер, исключения) не должны платить за это
ценой ``ImportError`` в окружениях без сконфигурированной БД (тесты,
скрипты, REPL). ``IngestionService`` импортируется по полному пути:
``from app.ingestion.pipeline import IngestionService``.
"""
from app.ingestion.chunker import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    PAGE_SEPARATOR,
    TextChunk,
    chunk_text,
    get_tokenizer,
)
from app.ingestion.exceptions import (
    CorruptFileError,
    EmptyTextError,
    ParserError,
    UnsupportedFormatError,
)
from app.ingestion.parser import parse_file

__all__ = [
    "CorruptFileError",
    "DEFAULT_OVERLAP_TOKENS",
    "DEFAULT_TARGET_TOKENS",
    "EmptyTextError",
    "PAGE_SEPARATOR",
    "ParserError",
    "TextChunk",
    "UnsupportedFormatError",
    "chunk_text",
    "get_tokenizer",
    "parse_file",
]