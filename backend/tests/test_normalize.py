from app.ingestion.normalize import normalize_text


def test_strips_bom():
    assert normalize_text("\ufeffhello") == "hello"


def test_normalizes_line_endings():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_collapses_excess_newlines():
    assert normalize_text("a\n\n\n\nb") == "a\n\nb"


def test_keeps_paragraph_separator():
    assert normalize_text("a\n\nb") == "a\n\nb"


def test_strips_trailing_whitespace_per_line():
    assert normalize_text("a   \nb\t\nc") == "a\nb\nc"


def test_removes_zero_width_chars():
    assert normalize_text("hel\u200blo\u200dworld") == "helloworld"


def test_preserves_form_feed():
    # PDF-парсер использует \f как разделитель страниц.
    assert "\f" in normalize_text("page1\fpage2")


def test_strips_outer_whitespace():
    assert normalize_text("  \n\nhello\n\n  ") == "hello"