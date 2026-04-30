"""Тесты эмбеддера. Используют реальную модель intfloat/multilingual-e5-base
через сессионный фикстур: первый запуск качает модель в HF-кэш (~1.1 ГБ),
последующие запуски — из кэша.
"""
import math

import pytest

from app.embedding import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    Embedder,
)

E5_BASE_DIM = 768


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    return Embedder("intfloat/multilingual-e5-base", batch_size=4)


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_dim_matches_e5_base(embedder):
    assert embedder.dim == E5_BASE_DIM


def test_model_name_property(embedder):
    assert embedder.model_name == "intfloat/multilingual-e5-base"


def test_embed_passages_returns_list_of_vectors(embedder):
    vectors = embedder.embed_passages(["Первый текст", "Второй текст"])
    assert isinstance(vectors, list)
    assert len(vectors) == 2
    for v in vectors:
        assert isinstance(v, list)
        assert len(v) == E5_BASE_DIM
        assert all(isinstance(x, float) for x in v)


def test_embed_passages_empty_input_returns_empty(embedder):
    assert embedder.embed_passages([]) == []


def test_embed_query_returns_flat_vector(embedder):
    vec = embedder.embed_query("семантический поиск")
    assert isinstance(vec, list)
    assert len(vec) == E5_BASE_DIM
    assert all(isinstance(x, float) for x in vec)


def test_embed_query_empty_raises(embedder):
    with pytest.raises(ValueError):
        embedder.embed_query("")
    with pytest.raises(ValueError):
        embedder.embed_query("   \n  ")


def test_passage_vectors_are_l2_normalized(embedder):
    [v] = embedder.embed_passages(["Проверка нормализации вектора."])
    assert abs(_norm(v) - 1.0) < 1e-4


def test_query_vectors_are_l2_normalized(embedder):
    v = embedder.embed_query("какой-нибудь запрос для проверки нормы")
    assert abs(_norm(v) - 1.0) < 1e-4


def test_paraphrase_has_higher_similarity_than_unrelated(embedder):
    """Парафраз должен быть ближе к оригиналу, чем неродственный текст."""
    [v_a, v_para, v_other] = embedder.embed_passages([
        "Семантический поиск помогает находить документы по смыслу.",
        "Поиск по смыслу нужен для нахождения релевантных документов.",
        "Рецепт классического борща с пампушками и салом.",
    ])
    sim_paraphrase = _cosine(v_a, v_para)
    sim_unrelated = _cosine(v_a, v_other)
    assert sim_paraphrase > sim_unrelated + 0.1


def test_query_finds_relevant_passage_over_noise(embedder):
    """Кросс-роль: query-вектор должен быть ближе к релевантному passage,
    чем к нерелевантному. Проверяет, что разные префиксы для query и
    passage не ломают саму идею поиска."""
    relevant = "Семантический поиск использует векторные представления текста."
    noise = "Сегодня хорошая погода для прогулки в парке."

    q = embedder.embed_query("как работает поиск по смыслу")
    [v_rel, v_noise] = embedder.embed_passages([relevant, noise])

    assert _cosine(q, v_rel) > _cosine(q, v_noise)


def test_passage_and_query_prefixes_produce_different_vectors(embedder):
    """Один и тот же текст с разными ролями должен давать разные векторы.
    Если совпадают — значит префиксы где-то теряются."""
    text = "Один и тот же текст без всякого изменения"
    v_passage = embedder.embed_passages([text])[0]
    v_query = embedder.embed_query(text)
    sim = _cosine(v_passage, v_query)
    assert sim < 0.999, "passage и query векторы идентичны — префиксы не применяются"


def test_batch_encoding_matches_singleton_encoding(embedder):
    """Эмбеддинг текста по одному и в составе батча должен совпадать
    с точностью до численных погрешностей float."""
    texts = ["Первый", "Второй текст", "Третий пример для сравнения"]
    batch = embedder.embed_passages(texts)
    singles = [embedder.embed_passages([t])[0] for t in texts]
    for b, s in zip(batch, singles):
        assert _cosine(b, s) > 0.9999


def test_warmup_does_not_raise(embedder):
    embedder.warmup()


def test_constants_match_e5_card():
    assert PASSAGE_PREFIX == "passage: "
    assert QUERY_PREFIX == "query: "


def test_invalid_batch_size_raises():
    """Проверка batch_size делается ДО загрузки модели, чтобы тест
    не платил за её скачивание."""
    with pytest.raises(ValueError):
        Embedder("intfloat/multilingual-e5-base", batch_size=0)
    with pytest.raises(ValueError):
        Embedder("intfloat/multilingual-e5-base", batch_size=-1)