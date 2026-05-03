"""Извлечение текста из .pdf файлов."""
import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.exceptions import CorruptFileError
from app.ingestion.normalize import normalize_text

logger = logging.getLogger(__name__)

PAGE_SEPARATOR = "\f"


def parse(path: Path) -> str:
    """Извлекает текст из PDF постранично, разделяя страницы form feed.

    Сканы без OCR-слоя дадут пустой или почти пустой результат —
    обработка этого случая делегируется вызывающему коду, который
    проверяет результат на пустоту после нормализации.
    """
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError) as e:
        raise CorruptFileError(f"failed to open PDF: {e}") from e

    pages: list[str] = []
    for page_num, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:  # noqa: BLE001
            # pypdf может бросать самые разные исключения на битых страницах.
            # Логируем и продолжаем — лучше получить часть текста, чем ничего.
            logger.warning(
                "failed to extract text from page %d of %s: %s",
                page_num,
                path.name,
                e,
            )
            pages.append("")

    text = PAGE_SEPARATOR.join(pages)
    return normalize_text(text)