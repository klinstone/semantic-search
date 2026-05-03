"""Извлечение текста из .txt файлов."""
import logging
from pathlib import Path

from app.ingestion.exceptions import CorruptFileError
from app.ingestion.normalize import normalize_text

logger = logging.getLogger(__name__)

# Кодировки для проверки по порядку.
_ENCODINGS_TO_TRY: tuple[str, ...] = ("utf-8", "cp1251")


def parse(path: Path) -> str:
    """Читает текстовый файл, пробуя кодировки по очереди.

    Если ни одна из строгих попыток не удалась, делает финальный проход
    UTF-8 с заменой невалидных байтов — лучше частично искажённый текст,
    чем падение всего пайплайна.
    """
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise CorruptFileError(f"failed to read file: {e}") from e

    for encoding in _ENCODINGS_TO_TRY:
        try:
            text = raw.decode(encoding, errors="strict")
            logger.debug("parsed %s as %s", path.name, encoding)
            return normalize_text(text)
        except UnicodeDecodeError:
            continue

    logger.warning(
        "no strict encoding matched for %s, falling back to utf-8 with replacement",
        path.name,
    )
    text = raw.decode("utf-8", errors="replace")
    return normalize_text(text)