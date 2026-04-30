import pytest

from app.ingestion.chunker import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    PAGE_SEPARATOR,
    TextChunk,
    chunk_text,
)

# Лимит контекста intfloat/multilingual-e5-base.
MODEL_MAX_TOKENS = 512
PASSAGE_PREFIX = "passage: "


def test_empty_text_returns_no_chunks(e5_tokenizer):
    assert chunk_text("", tokenizer=e5_tokenizer) == []


def test_whitespace_only_returns_no_chunks(e5_tokenizer):
    assert chunk_text("   \n\n  ", tokenizer=e5_tokenizer) == []


def test_short_text_is_one_chunk(e5_tokenizer):
    text = "Это короткий русский текст для проверки чанкера."
    chunks = chunk_text(text, tokenizer=e5_tokenizer)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == text
    assert chunk.char_start == 0
    assert chunk.char_end == len(text)
    assert chunk.token_count > 0
    assert chunk.metadata == {}


def test_long_text_splits_into_multiple_chunks(e5_tokenizer):
    paragraph = "Семантический поиск помогает находить документы по смыслу. " * 30
    text = "\n\n".join([paragraph] * 5)
    chunks = chunk_text(text, tokenizer=e5_tokenizer)
    assert len(chunks) > 1


def test_text_offsets_reconstruct_chunk_text(e5_tokenizer):
    text = "Первый абзац для проверки.\n\nВторой абзац немного длиннее. " * 40
    chunks = chunk_text(text, tokenizer=e5_tokenizer, target_tokens=120, overlap_tokens=20)
    assert len(chunks) > 1
    for ch in chunks:
        assert text[ch.char_start:ch.char_end] == ch.text


def test_chunks_within_model_context_after_passage_prefix(e5_tokenizer):
    """Каждый чанк после "passage: " должен помещаться в 512 токенов модели."""
    text = "Семантический поиск работает на основе векторных представлений текста. " * 200
    chunks = chunk_text(text, tokenizer=e5_tokenizer, target_tokens=400, overlap_tokens=50)
    assert len(chunks) > 1
    for ch in chunks:
        with_prefix = PASSAGE_PREFIX + ch.text
        encoded = e5_tokenizer.encode(with_prefix, add_special_tokens=True)
        assert len(encoded) <= MODEL_MAX_TOKENS, (
            f"chunk at {ch.char_start}-{ch.char_end} = {len(encoded)} tokens"
        )


def test_consecutive_chunks_overlap_in_chars(e5_tokenizer):
    text = "Семантический поиск. " * 200
    chunks = chunk_text(text, tokenizer=e5_tokenizer, target_tokens=80, overlap_tokens=20)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        # Перекрытие: следующий чанк стартует до конца предыдущего.
        assert nxt.char_start < prev.char_end
        # И всё-таки продвигается вперёд (нет зацикливания).
        assert nxt.char_start > prev.char_start
        assert nxt.char_end > prev.char_end


def test_chunks_are_in_order(e5_tokenizer):
    text = "Слово раз. " * 200
    chunks = chunk_text(text, tokenizer=e5_tokenizer, target_tokens=80, overlap_tokens=10)
    assert all(a.char_start <= b.char_start for a, b in zip(chunks, chunks[1:]))
    assert all(a.char_end <= b.char_end for a, b in zip(chunks, chunks[1:]))


def test_form_feed_yields_per_page_chunks(e5_tokenizer):
    text = "Текст первой страницы." + PAGE_SEPARATOR + "Текст второй страницы."
    chunks = chunk_text(text, tokenizer=e5_tokenizer)
    for ch in chunks:
        assert PAGE_SEPARATOR not in ch.text
    pages = sorted({ch.metadata["page"] for ch in chunks})
    assert pages == [1, 2]


def test_form_feed_offsets_index_original_text(e5_tokenizer):
    page1 = "Содержимое первой страницы. " * 50
    page2 = "Содержимое второй страницы немного отличается. " * 50
    text = page1 + PAGE_SEPARATOR + page2
    chunks = chunk_text(text, tokenizer=e5_tokenizer, target_tokens=100, overlap_tokens=20)

    for ch in chunks:
        assert text[ch.char_start:ch.char_end] == ch.text
        assert PAGE_SEPARATOR not in ch.text

    pages_seen = {ch.metadata["page"] for ch in chunks}
    assert pages_seen == {1, 2}


def test_long_unbroken_string_falls_back_to_token_split(e5_tokenizer):
    # Длинная строка из одинаковых символов без пробелов и переводов строк.
    text = "x" * 5000
    chunks = chunk_text(text, tokenizer=e5_tokenizer, target_tokens=100, overlap_tokens=10)
    assert len(chunks) >= 2
    for ch in chunks:
        encoded = e5_tokenizer.encode(PASSAGE_PREFIX + ch.text, add_special_tokens=True)
        assert len(encoded) <= MODEL_MAX_TOKENS


def test_metadata_is_independent_per_chunk(e5_tokenizer):
    text = "Текст. " * 100 + PAGE_SEPARATOR + "Ещё текст. " * 100
    chunks = chunk_text(text, tokenizer=e5_tokenizer, target_tokens=80, overlap_tokens=10)
    assert len(chunks) >= 2
    chunks[0].metadata["foo"] = "bar"
    for other in chunks[1:]:
        assert "foo" not in other.metadata


def test_invalid_overlap_raises(e5_tokenizer):
    with pytest.raises(ValueError):
        chunk_text("abc", tokenizer=e5_tokenizer, target_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        chunk_text("abc", tokenizer=e5_tokenizer, target_tokens=100, overlap_tokens=-1)


def test_invalid_target_raises(e5_tokenizer):
    with pytest.raises(ValueError):
        chunk_text("abc", tokenizer=e5_tokenizer, target_tokens=0)


def test_returns_textchunk_instances(e5_tokenizer):
    chunks = chunk_text("Привет мир", tokenizer=e5_tokenizer)
    assert all(isinstance(ch, TextChunk) for ch in chunks)


def test_token_count_reflects_chunk_text(e5_tokenizer):
    text = "Простой короткий текст."
    chunks = chunk_text(text, tokenizer=e5_tokenizer)
    assert len(chunks) == 1
    expected = len(e5_tokenizer.encode(text, add_special_tokens=False))
    assert chunks[0].token_count == expected


def test_paragraph_boundaries_preferred_when_possible(e5_tokenizer):
    """При наличии двойных переводов строк граница чанка предпочтительнее
    приходится на конец абзаца, а не в середину предложения."""
    paragraphs = [
                     "Первый небольшой абзац с законченной мыслью.",
                     "Второй абзац немного отличается по содержанию.",
                     "Третий абзац продолжает развивать тему документа.",
                     "Четвёртый абзац завершает первый раздел текста.",
                 ] * 8
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, tokenizer=e5_tokenizer, target_tokens=120, overlap_tokens=20)

    # Для каждого чанка кроме последнего: либо его конец совпадает с
    # концом одного из абзацев, либо его конец — это конец текста, либо
    # он попадает на границу "\n\n" в исходном тексте.
    matched_any_boundary = 0
    for ch in chunks[:-1]:
        if ch.char_end == len(text):
            continue
        # Конец чанка должен попасть на момент сразу перед "\n\n" или на пробел/точку.
        # Жёсткая проверка: подстрока сразу после чанка начинается с разделителя.
        tail = text[ch.char_end:ch.char_end + 2]
        if tail.startswith("\n\n") or text[ch.char_end - 1] in (".", "!", "?"):
            matched_any_boundary += 1

    # Хотя бы половина внутренних границ — на разумных местах.
    assert matched_any_boundary >= max(1, (len(chunks) - 1) // 2)


def test_short_segment_after_form_feed_still_chunked(e5_tokenizer):
    text = ("A" * 5) + PAGE_SEPARATOR + ("B" * 5)
    chunks = chunk_text(text, tokenizer=e5_tokenizer)
    pages = sorted({ch.metadata["page"] for ch in chunks})
    assert pages == [1, 2]
    page1 = [ch for ch in chunks if ch.metadata["page"] == 1][0]
    page2 = [ch for ch in chunks if ch.metadata["page"] == 2][0]
    assert page1.text == "AAAAA"
    assert page2.text == "BBBBB"
    # Смещения второй страницы учитывают ушедший \f.
    assert page2.char_start == 6
    assert page2.char_end == 11


def test_default_constants_match_specification():
    assert DEFAULT_TARGET_TOKENS == 400
    assert DEFAULT_OVERLAP_TOKENS == 50
    assert PAGE_SEPARATOR == "\f"


def test_all_chunks_are_nonempty(e5_tokenizer):
    text = "Слово. " * 300
    chunks = chunk_text(text, tokenizer=e5_tokenizer, target_tokens=80, overlap_tokens=15)
    for ch in chunks:
        assert ch.text.strip()
        assert ch.token_count > 0
        assert ch.char_end > ch.char_start
