"""Интеграционные тесты ингест-пайплайна.

Используется реальный Embedder (загруженный один раз на сессию) и
реальный chunker — это самые "хитрые" компоненты, и подмена их
фейками ослабила бы тесты. БД заменена на in-memory FakeSession,
Qdrant — на :memory: режим (полноценная локальная реализация).
"""
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from app.embedding import Embedder
from app.ingestion.exceptions import EmptyTextError
from app.ingestion.pipeline import IndexingError, IngestionService
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from tests._fake_session import FakeSession

COLLECTION = "test_chunks"


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    # Та же модель, что в проекте — при первом запуске уже скачана,
    # тест эмбеддера её прогрел.
    return Embedder("intfloat/multilingual-e5-base", batch_size=4)


@pytest.fixture
def qdrant(embedder) -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )
    return client


@pytest.fixture
def upload_dir(tmp_path):
    d = tmp_path / "uploads"
    d.mkdir()
    return d


@pytest.fixture
def service(embedder, upload_dir) -> IngestionService:
    return IngestionService(
        embedder=embedder,
        target_tokens=120,
        overlap_tokens=20,
        collection_name=COLLECTION,
        upload_dir=upload_dir,
    )


def _seed_document(db: FakeSession, document_id: UUID, filename: str, mime: str) -> Document:
    doc = Document(
        id=document_id,
        filename=filename,
        mime_type=mime,
        size_bytes=1234,
        status=DocumentStatus.PENDING,
        chunks_count=0,
        text_length=None,
    )
    db.seed(doc)
    return doc


def _write_txt(upload_dir, document_id: UUID, content: str) -> None:
    (upload_dir / f"{document_id}.txt").write_text(content, encoding="utf-8")


def _qdrant_count(qdrant: QdrantClient, document_id: UUID) -> int:
    flt = Filter(must=[FieldCondition(
        key="document_id", match=MatchValue(value=str(document_id))
    )])
    return qdrant.count(collection_name=COLLECTION, count_filter=flt, exact=True).count


def test_indexes_simple_text_document(service, qdrant, upload_dir):
    """Базовый счастливый путь: документ → indexed, чанки в БД и Qdrant."""
    db = FakeSession()
    doc_id = uuid4()
    text = (
        "Семантический поиск помогает находить документы по смыслу.\n\n"
        "Векторные представления текста позволяют сравнивать близость "
        "запроса и документа в латентном пространстве.\n\n"
        "Чанкинг разбивает длинный текст на фрагменты, помещающиеся "
        "в контекст модели."
    ) * 5
    _seed_document(db, doc_id, "manual.txt", "text/plain")
    _write_txt(upload_dir, doc_id, text)

    result = service.index_document(doc_id, db, qdrant)

    assert result.document_id == doc_id
    assert result.chunks_count > 0
    assert result.text_length == len(text)

    doc = db.get(Document, doc_id)
    assert doc.status == DocumentStatus.INDEXED
    assert doc.chunks_count == result.chunks_count
    assert doc.text_length == len(text)
    assert doc.indexed_at is not None
    assert doc.error_message is None

    chunks = [c for c in db.all(Chunk) if c.document_id == doc_id]
    assert len(chunks) == result.chunks_count
    assert _qdrant_count(qdrant, doc_id) == result.chunks_count


def test_chunk_ids_match_qdrant_point_ids(service, qdrant, upload_dir):
    """ID чанков в БД должны совпадать с ID точек в Qdrant.

    Это инвариант, на котором будет построен /search: vector → point.id →
    JOIN в БД по chunks.id. Если связка нарушится, поиск вернёт 404.
    """
    db = FakeSession()
    doc_id = uuid4()
    _seed_document(db, doc_id, "doc.txt", "text/plain")
    _write_txt(upload_dir, doc_id, "Текст для проверки. " * 100)

    service.index_document(doc_id, db, qdrant)

    chunk_ids = {str(c.id) for c in db.all(Chunk) if c.document_id == doc_id}

    # Достаём все точки документа из Qdrant.
    flt = Filter(must=[FieldCondition(
        key="document_id", match=MatchValue(value=str(doc_id))
    )])
    points, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=flt,
        limit=1000,
        with_payload=True,
    )
    point_ids = {str(p.id) for p in points}
    assert chunk_ids == point_ids


def test_status_is_processing_during_run(service, qdrant, upload_dir, monkeypatch):
    """Между PENDING и INDEXED статус должен быть PROCESSING.

    Проверяем, перехватив момент: подменяем embed_passages так, чтобы
    он сначала проверил статус документа в БД, а потом вернул векторы.
    """
    db = FakeSession()
    doc_id = uuid4()
    _seed_document(db, doc_id, "doc.txt", "text/plain")
    _write_txt(upload_dir, doc_id, "Короткий текст.")

    captured_status: list[str] = []
    real_embed = service._embedder.embed_passages

    def spy(texts):
        doc = db.get(Document, doc_id)
        captured_status.append(doc.status)
        return real_embed(texts)

    monkeypatch.setattr(service._embedder, "embed_passages", spy)
    service.index_document(doc_id, db, qdrant)

    assert captured_status == [DocumentStatus.PROCESSING]
    assert db.get(Document, doc_id).status == DocumentStatus.INDEXED


def test_missing_file_marks_document_failed(service, qdrant, upload_dir):
    """Файла на диске нет — документ должен попасть в FAILED, не зависнуть."""
    db = FakeSession()
    doc_id = uuid4()
    _seed_document(db, doc_id, "missing.txt", "text/plain")
    # Файл намеренно не пишем.

    with pytest.raises(IndexingError):
        service.index_document(doc_id, db, qdrant)

    doc = db.get(Document, doc_id)
    assert doc.status == DocumentStatus.FAILED
    assert doc.error_message
    assert _qdrant_count(qdrant, doc_id) == 0


def test_empty_text_marks_document_failed(service, qdrant, upload_dir):
    """Пустой файл — EmptyTextError, документ FAILED, сообщение пользователю."""
    db = FakeSession()
    doc_id = uuid4()
    _seed_document(db, doc_id, "empty.txt", "text/plain")
    _write_txt(upload_dir, doc_id, "   \n\n   ")

    with pytest.raises(IndexingError):
        service.index_document(doc_id, db, qdrant)

    doc = db.get(Document, doc_id)
    assert doc.status == DocumentStatus.FAILED
    assert "no text" in doc.error_message.lower() or "empty" in doc.error_message.lower()


def test_qdrant_failure_compensates_chunks(service, qdrant, upload_dir, monkeypatch):
    """Если Qdrant упал после записи в БД — чанки откатываются,
    документ переходит в FAILED. Postgres не должен оставаться с
    "чанками без векторов"."""
    db = FakeSession()
    doc_id = uuid4()
    _seed_document(db, doc_id, "doc.txt", "text/plain")
    _write_txt(upload_dir, doc_id, "Тест " * 200)

    def boom(*args, **kwargs):
        raise RuntimeError("qdrant exploded")

    monkeypatch.setattr(qdrant, "upsert", boom)

    with pytest.raises(IndexingError):
        service.index_document(doc_id, db, qdrant)

    doc = db.get(Document, doc_id)
    assert doc.status == DocumentStatus.FAILED
    # Чанки этого документа должны быть удалены из БД.
    surviving = [c for c in db.all(Chunk) if c.document_id == doc_id]
    assert surviving == []


def test_reindex_replaces_existing_data(service, qdrant, upload_dir):
    """Повторная индексация того же документа удаляет старые чанки и
    точки, заменяя их новыми. Идемпотентность."""
    db = FakeSession()
    doc_id = uuid4()
    _seed_document(db, doc_id, "doc.txt", "text/plain")

    # Первая индексация: длинный текст → много чанков.
    _write_txt(upload_dir, doc_id, "Первый вариант. " * 200)
    first = service.index_document(doc_id, db, qdrant)

    # Вторая индексация того же id, но с другим содержимым.
    _write_txt(upload_dir, doc_id, "Короткий новый текст.")
    second = service.index_document(doc_id, db, qdrant)

    # В БД и Qdrant — только новые чанки.
    assert second.chunks_count != first.chunks_count
    surviving_chunks = [c for c in db.all(Chunk) if c.document_id == doc_id]
    assert len(surviving_chunks) == second.chunks_count
    assert all("Короткий" in c.text or "Короткий" in (c.text or "")
               for c in surviving_chunks) or any(
        "Короткий" in c.text for c in surviving_chunks
    )
    assert _qdrant_count(qdrant, doc_id) == second.chunks_count


def test_other_documents_not_affected_by_indexing(service, qdrant, upload_dir):
    """Индексация одного документа не должна влиять на чанки/точки другого."""
    db = FakeSession()

    doc_a = uuid4()
    doc_b = uuid4()
    _seed_document(db, doc_a, "a.txt", "text/plain")
    _seed_document(db, doc_b, "b.txt", "text/plain")
    _write_txt(upload_dir, doc_a, "Документ A. " * 50)
    _write_txt(upload_dir, doc_b, "Документ B. " * 50)

    service.index_document(doc_a, db, qdrant)
    a_chunks_before = sum(1 for c in db.all(Chunk) if c.document_id == doc_a)
    a_points_before = _qdrant_count(qdrant, doc_a)

    service.index_document(doc_b, db, qdrant)

    # A не пострадал.
    assert sum(1 for c in db.all(Chunk) if c.document_id == doc_a) == a_chunks_before
    assert _qdrant_count(qdrant, doc_a) == a_points_before


def test_chunks_have_offsets_and_metadata(service, qdrant, upload_dir):
    """char_start/char_end и chunk_metadata должны попадать в БД."""
    db = FakeSession()
    doc_id = uuid4()
    _seed_document(db, doc_id, "doc.txt", "text/plain")
    text = "Первый абзац.\n\nВторой абзац немного длиннее. " * 30
    _write_txt(upload_dir, doc_id, text)

    service.index_document(doc_id, db, qdrant)

    chunks = sorted(
        (c for c in db.all(Chunk) if c.document_id == doc_id),
        key=lambda c: c.chunk_index,
    )
    for c in chunks:
        assert c.char_start is not None
        assert c.char_end is not None
        assert c.char_end > c.char_start
        # chunk_metadata — словарь (для txt без \f — пустой).
        assert isinstance(c.chunk_metadata, dict)


def test_unknown_document_raises(service, qdrant):
    """Документа нет в БД — IndexingError, никаких побочных эффектов."""
    db = FakeSession()
    doc_id = uuid4()

    with pytest.raises(IndexingError):
        service.index_document(doc_id, db, qdrant)


def test_indexed_document_has_ascending_chunk_indices(service, qdrant, upload_dir):
    """chunk_index идёт 0,1,2,…N-1 без пропусков. Нужно для UI."""
    db = FakeSession()
    doc_id = uuid4()
    _seed_document(db, doc_id, "doc.txt", "text/plain")
    _write_txt(upload_dir, doc_id, "Слово. " * 200)

    result = service.index_document(doc_id, db, qdrant)

    indices = sorted(
        c.chunk_index for c in db.all(Chunk) if c.document_id == doc_id
    )
    assert indices == list(range(result.chunks_count))


def test_failed_document_stays_failed_on_subsequent_unrelated_call(
    service, qdrant, upload_dir
):
    """Если индексация одного документа упала, повторный успешный
    запуск ДРУГОГО документа не должен сбросить статус первого."""
    db = FakeSession()

    doc_bad = uuid4()
    doc_good = uuid4()
    _seed_document(db, doc_bad, "bad.txt", "text/plain")
    _seed_document(db, doc_good, "good.txt", "text/plain")

    # bad: пустой файл → FAILED.
    _write_txt(upload_dir, doc_bad, "")
    with pytest.raises(IndexingError):
        service.index_document(doc_bad, db, qdrant)

    _write_txt(upload_dir, doc_good, "Нормальный текст. " * 50)
    service.index_document(doc_good, db, qdrant)

    assert db.get(Document, doc_bad).status == DocumentStatus.FAILED
    assert db.get(Document, doc_good).status == DocumentStatus.INDEXED