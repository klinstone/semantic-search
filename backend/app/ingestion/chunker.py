"""Токенный чанкер для подготовки текста к индексации.

Алгоритм: рекурсивное разбиение текста по структурным разделителям
(\\n\\n -> \\n -> ". " -> " "), затем жадная склейка получившихся "атомов"
в чанки, ограниченные `target_tokens`. Между соседними чанками сохраняется
перекрытие около `overlap_tokens` для устойчивости поиска на границах.

Длина считается через токенайзер целевой модели эмбеддингов: счёт по
символам у мультиязычных моделей (e5/MiniLM) даёт большую погрешность и
приводит к тихим обрезаниям на длинных русских чанках.

Form-feed (\\f) трактуется как граница страницы (PDF-парсер расставляет
их между страницами). Чанки никогда не пересекают \\f и получают
`metadata={"page": N}` с 1-based номером страницы.
"""
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Подобраны под intfloat/multilingual-e5-base (контекст 512). Запас закрывает
# префикс "passage: " (~2 токена), спецтокены [CLS]/[SEP] (2 токена) и шум
# на разделителях между атомами.
DEFAULT_TARGET_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50

# Маркер границы страниц от PDF-парсера. Чанки не пересекают это значение.
PAGE_SEPARATOR = "\f"

# Разделители от наиболее структурных к наименее. Пробел — последний шанс
# до жёсткого разреза по токенам.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class TextChunk:
    """Один чанк, готовый к векторизации.

    `char_start` / `char_end` — смещения в исходном тексте (полученном из
    парсера, до удаления form-feed). Инвариант: `text == original[char_start:char_end]`
    для непустых чанков, не пересекающих границу страницы.
    """
    text: str
    char_start: int
    char_end: int
    token_count: int
    metadata: dict


@dataclass(frozen=True)
class _Atom:
    """Внутреннее представление неделимого фрагмента, влезающего в бюджет."""
    start: int
    end: int
    tokens: int


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Загружает и кэширует токенайзер модели.

    Скачиваются только файлы токенайзера (несколько МБ); веса модели не
    подгружаются. Кэш на диске задаётся через переменную HF_HOME.
    """
    logger.info("loading tokenizer %s", model_name)
    return AutoTokenizer.from_pretrained(model_name)


def chunk_text(
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[TextChunk]:
    """Делит нормализованный текст на перекрывающиеся чанки.

    Ожидает на вход результат `app.ingestion.normalize.normalize_text`.
    Возвращает чанки в порядке появления в исходном тексте; чанк не
    содержит символа `\\f`.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be in [0, target_tokens)")
    if not text:
        return []

    sep_tuple = tuple(separators)

    if PAGE_SEPARATOR not in text:
        return _chunk_segment(
            segment=text,
            segment_offset=0,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            separators=sep_tuple,
            metadata={},
        )

    chunks: list[TextChunk] = []
    cursor = 0
    for page_idx, page_text in enumerate(text.split(PAGE_SEPARATOR), start=1):
        if page_text.strip():
            chunks.extend(_chunk_segment(
                segment=page_text,
                segment_offset=cursor,
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                separators=sep_tuple,
                metadata={"page": page_idx},
            ))
        # +1 учитывает сам символ form-feed между страницами.
        cursor += len(page_text) + 1
    return chunks


def _chunk_segment(
        segment: str,
        segment_offset: int,
        tokenizer: PreTrainedTokenizerBase,
        target_tokens: int,
        overlap_tokens: int,
        separators: tuple[str, ...],
        metadata: dict,
) -> list[TextChunk]:
    if not segment.strip():
        return []

    atoms = _split_recursive(
        text=segment,
        char_offset=segment_offset,
        separators=separators,
        tokenizer=tokenizer,
        max_tokens=target_tokens,
    )
    if not atoms:
        return []

    return _pack_atoms(
        atoms=atoms,
        segment=segment,
        segment_offset=segment_offset,
        tokenizer=tokenizer,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        metadata=metadata,
    )


def _count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    # add_special_tokens=False: спецтокены добавляются моделью на этапе
    # эмбеддинга вместе с префиксом "passage: " / "query: ".
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_recursive(
        text: str,
        char_offset: int,
        separators: tuple[str, ...],
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
) -> list[_Atom]:
    if not text:
        return []

    n_tokens = _count_tokens(text, tokenizer)
    if n_tokens <= max_tokens:
        return [_Atom(start=char_offset, end=char_offset + len(text), tokens=n_tokens)]

    for idx, sep in enumerate(separators):
        if sep and sep in text:
            return _split_by_separator(
                text=text,
                char_offset=char_offset,
                sep=sep,
                rest_separators=separators[idx + 1:],
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )

    # Ни один разделитель не помог — это длинная неразрывная строка
    # (например, base64-кусок или ссылка без пробелов). Режем по токенам.
    return _split_by_tokens(text, char_offset, tokenizer, max_tokens)


def _split_by_separator(
        text: str,
        char_offset: int,
        sep: str,
        rest_separators: tuple[str, ...],
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
) -> list[_Atom]:
    atoms: list[_Atom] = []
    parts = text.split(sep)
    pos = 0
    for idx, part in enumerate(parts):
        if part:
            atoms.extend(_split_recursive(
                text=part,
                char_offset=char_offset + pos,
                separators=rest_separators,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            ))
        pos += len(part)
        if idx < len(parts) - 1:
            pos += len(sep)
    return atoms


def _split_by_tokens(
        text: str,
        char_offset: int,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
) -> list[_Atom]:
    """Жёсткий разрез по токенам — fallback для строк без разделителей.

    Требует "fast" токенайзер (return_offsets_mapping). Все современные
    HF-токенайзеры таковы по умолчанию.
    """
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if not ids:
        return []

    atoms: list[_Atom] = []
    for start_idx in range(0, len(ids), max_tokens):
        end_idx = min(start_idx + max_tokens, len(ids))
        slice_offsets = offsets[start_idx:end_idx]
        if not slice_offsets:
            continue
        atom_char_start = slice_offsets[0][0]
        atom_char_end = slice_offsets[-1][1]
        if atom_char_end <= atom_char_start:
            continue
        atoms.append(_Atom(
            start=char_offset + atom_char_start,
            end=char_offset + atom_char_end,
            tokens=end_idx - start_idx,
        ))
    return atoms


def _pack_atoms(
        atoms: list[_Atom],
        segment: str,
        segment_offset: int,
        tokenizer: PreTrainedTokenizerBase,
        target_tokens: int,
        overlap_tokens: int,
        metadata: dict,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    i = 0
    n = len(atoms)
    while i < n:
        # Жадно набираем атомы, пока сумма их токен-счётчиков не превысит target.
        j = i
        cumulative = 0
        while j < n and cumulative + atoms[j].tokens <= target_tokens:
            cumulative += atoms[j].tokens
            j += 1

        if j == i:
            # Один атом превышает target. _split_by_tokens должен был не
            # допустить такого, но защищаемся от зацикливания.
            j = i + 1

        chunk_start = atoms[i].start
        chunk_end = atoms[j - 1].end
        chunk_text_value = segment[chunk_start - segment_offset:chunk_end - segment_offset]

        # Точный счёт с учётом разделителей между атомами в исходном тексте.
        actual_tokens = _count_tokens(chunk_text_value, tokenizer)

        chunks.append(TextChunk(
            text=chunk_text_value,
            char_start=chunk_start,
            char_end=chunk_end,
            token_count=actual_tokens,
            metadata=dict(metadata),
        ))

        if j >= n:
            break

        # Откатываемся назад от j на ~overlap_tokens, чтобы следующий чанк
        # начался с конца текущего. Размер перекрытия дискретен — кратен
        # размерам атомов.
        new_i = j
        accumulated = 0
        while new_i > i + 1 and accumulated < overlap_tokens:
            new_i -= 1
            accumulated += atoms[new_i].tokens

        i = max(new_i, i + 1)

    return chunks
