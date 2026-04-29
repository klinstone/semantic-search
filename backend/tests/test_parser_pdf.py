import pytest

from app.ingestion.exceptions import CorruptFileError, EmptyTextError
from app.ingestion.parser import parse_file
from app.ingestion.parsers.pdf import PAGE_SEPARATOR, parse as parse_pdf

MIME_PDF = "application/pdf"


def test_parses_single_page(make_pdf):
    path = make_pdf("one.pdf", ["Hello PDF world"])
    text = parse_pdf(path)
    assert "Hello PDF world" in text


def test_separates_pages_with_form_feed(make_pdf):
    path = make_pdf("multi.pdf", ["First page text", "Second page text"])
    text = parse_pdf(path)
    assert "First page text" in text
    assert "Second page text" in text
    assert PAGE_SEPARATOR in text


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf")
    with pytest.raises(CorruptFileError):
        parse_pdf(path)


def test_empty_pdf_raises_via_dispatcher(make_pdf):
    path = make_pdf("blank.pdf", [""])
    with pytest.raises(EmptyTextError):
        parse_file(path, MIME_PDF)