"""Нормализация извлечённого текста перед чанкингом."""
import re

# Невидимые/нулевой ширины символы, которые часто появляются в PDF/DOCX
# и мешают токенизации. \ufeff в начале строки (BOM) обрабатывается отдельно.
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")

# Три и более переводов строки подряд → два (граница абзаца).
_EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")

# Trailing whitespace на каждой строке.
_TRAILING_WHITESPACE_PATTERN = re.compile(r"[ \t]+$", re.MULTILINE)


def normalize_text(text: str) -> str:
    """Приводит текст к каноническому виду.

    Сохраняет символ \\f (form feed) — он используется PDF-парсером
    как маркер границы страниц для последующих этапов.
    """
    if text.startswith("\ufeff"):
        text = text[1:]

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH_PATTERN.sub("", text)
    text = _TRAILING_WHITESPACE_PATTERN.sub("", text)
    text = _EXCESS_NEWLINES_PATTERN.sub("\n\n", text)

    return text.strip()