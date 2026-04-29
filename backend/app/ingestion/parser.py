"""Диспетчер парсеров по MIME-типу."""
import logging
from collections.abc import Callable
from pathlib import Path

from app.ingestion.exceptions import EmptyTextError, UnsupportedFormatError
from app.ingestion.parsers import docx as docx_parser
from app.ingestion.parsers import pdf as pdf_parser
from app.ingestion.parsers import txt as txt_parser

logger = logging.getLogger(__name__)

_PARSERS: dict[str, Callable[[Path], str]] = {
    "application/pdf": pdf_parser.parse,
    "text/plain": txt_parser.parse,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        docx_parser.parse
    ),
}


def parse_file(path: Path, mime_type: str) -> str:
    """Извлекает текст из файла, выбирая парсер по MIME-типу.

    Бросает UnsupportedFormatError, если для типа нет парсера,
    EmptyTextError, если результат пустой после нормализации,
    или CorruptFileError, если парсер не смог прочитать файл.
    """
    parser = _PARSERS.get(mime_type)
    if parser is None:
        raise UnsupportedFormatError(f"no parser for MIME type {mime_type!r}")

    text = parser(path)
    if not text.strip():
        raise EmptyTextError(
            f"no text extracted from {path.name} (possibly a scanned PDF)"
        )

    logger.info(
        "parsed %s (%s): %d characters extracted",
        path.name,
        mime_type,
        len(text),
    )
    return text