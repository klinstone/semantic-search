import pytest

from app.ingestion.exceptions import UnsupportedFormatError
from app.ingestion.parser import parse_file


def test_unsupported_mime_raises(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"some bytes")
    with pytest.raises(UnsupportedFormatError):
        parse_file(path, "application/octet-stream")


def test_dispatcher_routes_to_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello dispatcher", encoding="utf-8")
    assert parse_file(path, "text/plain") == "hello dispatcher"