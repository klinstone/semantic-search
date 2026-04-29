"""Извлечение текста из .docx файлов."""
import logging
from pathlib import Path

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from lxml.etree import _Element

from app.ingestion.exceptions import CorruptFileError
from app.ingestion.normalize import normalize_text

logger = logging.getLogger(__name__)


def _iter_block_text(element: _Element) -> list[str]:
    """Рекурсивно обходит элементы документа, собирая текст из параграфов и таблиц.

    Стандартный итератор `doc.paragraphs` пропускает содержимое таблиц.
    Этот обход спускается внутрь `<w:tbl>` и забирает текст всех ячеек,
    сохраняя порядок появления в документе.
    """
    parts: list[str] = []
    for child in element.iterchildren():
        tag = child.tag
        if tag == qn("w:p"):
            text = "".join(t.text or "" for t in child.iter(qn("w:t")))
            parts.append(text)
        elif tag == qn("w:tbl"):
            for row in child.iter(qn("w:tr")):
                row_cells: list[str] = []
                for cell in row.iter(qn("w:tc")):
                    cell_text = " ".join(
                        "".join(t.text or "" for t in p.iter(qn("w:t")))
                        for p in cell.iter(qn("w:p"))
                    )
                    row_cells.append(cell_text)
                parts.append("\t".join(row_cells))
    return parts


def parse(path: Path) -> str:
    """Извлекает текст из DOCX, включая содержимое таблиц.

    Параграфы и строки таблиц разделяются переводами строк,
    ячейки одной строки таблицы — табуляцией.
    """
    try:
        doc = DocxDocument(str(path))
    except (PackageNotFoundError, OSError) as e:
        raise CorruptFileError(f"failed to open DOCX: {e}") from e

    parts = _iter_block_text(doc.element.body)
    text = "\n".join(parts)
    return normalize_text(text)