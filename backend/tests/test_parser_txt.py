import pytest

from app.ingestion.exceptions import EmptyTextError
from app.ingestion.parser import parse_file
from app.ingestion.parsers.txt import parse as parse_txt

MIME_TXT = "text/plain"


def test_parses_utf8(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Hello, мир!\nLine two.", encoding="utf-8")
    assert parse_txt(path) == "Hello, мир!\nLine two."


def test_parses_cp1251_fallback(tmp_path):
    path = tmp_path / "windows.txt"
    path.write_bytes("Привет, мир!".encode("cp1251"))
    assert parse_txt(path) == "Привет, мир!"


def test_strips_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfhello")
    assert parse_txt(path) == "hello"


def test_empty_file_raises_via_dispatcher(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(EmptyTextError):
        parse_file(path, MIME_TXT)


def test_whitespace_only_file_raises_via_dispatcher(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("\n\n   \n\n", encoding="utf-8")
    with pytest.raises(EmptyTextError):
        parse_file(path, MIME_TXT)


def test_garbage_bytes_does_not_crash(tmp_path):
    """Невалидные байты в обоих кодеках не должны валить парсер."""
    path = tmp_path / "garbage.txt"
    path.write_bytes(b"valid\x80\x81\x82end")
    result = parse_txt(path)
    assert "valid" in result
    assert "end" in result