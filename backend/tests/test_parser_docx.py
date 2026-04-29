import pytest

from app.ingestion.exceptions import CorruptFileError, EmptyTextError
from app.ingestion.parser import parse_file
from app.ingestion.parsers.docx import parse as parse_docx

MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_parses_paragraphs(make_docx):
    path = make_docx("simple.docx", ["First paragraph.", "Second paragraph."])
    text = parse_docx(path)
    assert "First paragraph." in text
    assert "Second paragraph." in text


def test_parses_russian_text(make_docx):
    path = make_docx("ru.docx", ["Это первый параграф.", "Это второй."])
    text = parse_docx(path)
    assert "Это первый параграф." in text
    assert "Это второй." in text


def test_extracts_table_content(make_docx):
    path = make_docx(
        "with_table.docx",
        paragraphs=["Before table"],
        tables=[
            [["A1", "B1"], ["A2", "B2"]],
        ],
    )
    text = parse_docx(path)
    assert "Before table" in text
    assert "A1" in text and "B1" in text
    assert "A2" in text and "B2" in text


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a docx")
    with pytest.raises(CorruptFileError):
        parse_docx(path)


def test_empty_docx_raises_via_dispatcher(make_docx):
    path = make_docx("empty.docx", paragraphs=[])
    with pytest.raises(EmptyTextError):
        parse_file(path, MIME_DOCX)