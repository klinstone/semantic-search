"""Тесты GET/DELETE-эндпоинтов документов через FastAPI TestClient.

Тестовое приложение собирается отдельно от ``app.main.app``, чтобы не
запускать lifespan (он лезет в Qdrant/HF и тянет тяжёлые ресурсы).
Зависимости подменяются через ``app.dependency_overrides``: get_db
указывает на FakeSession, get_qdrant — на in-memory клиент.
"""
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.api.deps import get_qdrant
from app.api.documents import router as documents_router
from app.api.errors import register_exception_handlers
from app.config import settings
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.storage.database import get_db
from tests._fake_session import FakeSession

COLLECTION = settings.qdrant_collection


@pytest.fixture
def db() -> FakeSession:
    return FakeSession()


@pytest.fixture
def qdrant() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        # Размерность не важна для тестов удаления — выбран минимум.
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    return client


@pytest.fixture
def upload_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "uploads"
    d.mkdir()
    monkeypatch.setattr(settings, "upload_dir", d)
    return d


@pytest.fixture
def client(db, qdrant, upload_dir) -> TestClient:
    """Тестовый FastAPI собран отдельно, чтобы избежать lifespan основного
    приложения. Регистрируются те же handlers ошибок и тот же роутер."""
    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.include_router(documents_router, prefix="/api/v1")

    test_app.dependency_overrides[get_db] = lambda: db
    test_app.dependency_overrides[get_qdrant] = lambda: qdrant

    # raise_server_exceptions=False: иначе TestClient пробрасывает
    # необработанные исключения в pytest, и мы не можем проверить, что
    # обработчик ошибок реально возвращает 500.
    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c


def _make_doc(
    db: FakeSession,
    *,
    filename: str = "doc.txt",
    status: str = DocumentStatus.INDEXED,
    chunks_count: int = 0,
    text_length: int | None = None,
    error_message: str | None = None,
    uploaded_at: datetime | None = None,
    indexed_at: datetime | None = None,
    mime_type: str = "text/plain",
) -> Document:
    doc = Document(
        id=uuid4(),
        filename=filename,
        mime_type=mime_type,
        size_bytes=100,
        status=status,
        chunks_count=chunks_count,
        text_length=text_length,
        error_message=error_message,
        uploaded_at=uploaded_at or datetime.now(UTC),
        indexed_at=indexed_at,
    )
    db.seed(doc)
    return doc


# -------------------- GET /documents --------------------


def test_list_documents_empty(client):
    r = client.get("/api/v1/documents")
    assert r.status_code == 200
    body = r.json()
    assert body == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_list_documents_returns_items(client, db):
    _make_doc(db, filename="a.txt")
    _make_doc(db, filename="b.txt")

    r = client.get("/api/v1/documents")
    body = r.json()
    assert r.status_code == 200
    assert body["total"] == 2
    assert {it["filename"] for it in body["items"]} == {"a.txt", "b.txt"}


def test_list_documents_sorted_by_uploaded_at_desc(client, db):
    now = datetime.now(UTC)
    _make_doc(db, filename="oldest", uploaded_at=now - timedelta(hours=2))
    _make_doc(db, filename="newest", uploaded_at=now)
    _make_doc(db, filename="middle", uploaded_at=now - timedelta(hours=1))

    r = client.get("/api/v1/documents")
    names = [it["filename"] for it in r.json()["items"]]
    assert names == ["newest", "middle", "oldest"]


def test_list_documents_pagination(client, db):
    now = datetime.now(UTC)
    for i in range(5):
        _make_doc(db, filename=f"doc-{i}.txt", uploaded_at=now - timedelta(seconds=i))

    r = client.get("/api/v1/documents?limit=2&offset=0")
    body = r.json()
    assert body["total"] == 5
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert len(body["items"]) == 2
    assert [it["filename"] for it in body["items"]] == ["doc-0.txt", "doc-1.txt"]

    r = client.get("/api/v1/documents?limit=2&offset=2")
    assert [it["filename"] for it in r.json()["items"]] == ["doc-2.txt", "doc-3.txt"]

    r = client.get("/api/v1/documents?limit=2&offset=4")
    assert [it["filename"] for it in r.json()["items"]] == ["doc-4.txt"]


def test_list_documents_filter_by_status(client, db):
    _make_doc(db, filename="ok-1.txt", status=DocumentStatus.INDEXED)
    _make_doc(db, filename="ok-2.txt", status=DocumentStatus.INDEXED)
    _make_doc(db, filename="bad.txt", status=DocumentStatus.FAILED)
    _make_doc(db, filename="pending.txt", status=DocumentStatus.PENDING)

    r = client.get("/api/v1/documents?status=indexed")
    body = r.json()
    assert body["total"] == 2
    assert {it["filename"] for it in body["items"]} == {"ok-1.txt", "ok-2.txt"}

    r = client.get("/api/v1/documents?status=failed")
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["filename"] == "bad.txt"


def test_list_documents_invalid_status_returns_422(client):
    r = client.get("/api/v1/documents?status=foo")
    assert r.status_code == 422


def test_list_documents_invalid_pagination_returns_422(client):
    assert client.get("/api/v1/documents?limit=0").status_code == 422
    assert client.get("/api/v1/documents?limit=101").status_code == 422
    assert client.get("/api/v1/documents?offset=-1").status_code == 422


# -------------------- GET /documents/{id} --------------------


def test_get_document_returns_full_detail(client, db):
    indexed_at = datetime.now(UTC)
    doc = _make_doc(
        db,
        filename="report.pdf",
        status=DocumentStatus.INDEXED,
        chunks_count=42,
        text_length=10_000,
        indexed_at=indexed_at,
        mime_type="application/pdf",
    )

    r = client.get(f"/api/v1/documents/{doc.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == str(doc.id)
    assert body["filename"] == "report.pdf"
    assert body["status"] == "indexed"
    assert body["chunks_count"] == 42
    assert body["text_length"] == 10_000
    assert body["error_message"] is None
    assert body["indexed_at"] is not None


def test_get_document_includes_error_message_for_failed(client, db):
    doc = _make_doc(
        db,
        status=DocumentStatus.FAILED,
        error_message="no text extracted",
    )
    r = client.get(f"/api/v1/documents/{doc.id}")
    assert r.status_code == 200
    assert r.json()["error_message"] == "no text extracted"


def test_get_document_returns_404(client):
    r = client.get(f"/api/v1/documents/{uuid4()}")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "NOT_FOUND"


def test_get_document_invalid_uuid_returns_422(client):
    r = client.get("/api/v1/documents/not-a-uuid")
    assert r.status_code == 422


# -------------------- DELETE /documents/{id} --------------------


def _seed_qdrant_points(qdrant: QdrantClient, document_id: UUID, n: int) -> None:
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=[0.0, 0.0, 0.0, 1.0],
            payload={"document_id": str(document_id), "chunk_index": i},
        )
        for i in range(n)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)


def _qdrant_count(qdrant: QdrantClient, document_id: UUID) -> int:
    flt = Filter(must=[FieldCondition(
        key="document_id", match=MatchValue(value=str(document_id))
    )])
    return qdrant.count(collection_name=COLLECTION, count_filter=flt, exact=True).count


def test_delete_document_removes_everything(client, db, qdrant, upload_dir):
    doc = _make_doc(db, filename="to-delete.txt")

    file_path = upload_dir / f"{doc.id}.txt"
    file_path.write_text("payload", encoding="utf-8")

    _seed_qdrant_points(qdrant, doc.id, n=3)
    # Эмулируем chunks в БД (без них нечего проверять CASCADE).
    for i in range(3):
        chunk = Chunk(
            id=uuid4(),
            document_id=doc.id,
            chunk_index=i,
            text=f"chunk {i}",
            chunk_metadata={},
        )
        db.seed(chunk)

    r = client.delete(f"/api/v1/documents/{doc.id}")
    assert r.status_code == 204
    assert r.content == b""

    assert db.get(Document, doc.id) is None
    assert [c for c in db.all(Chunk) if c.document_id == doc.id] == []
    assert _qdrant_count(qdrant, doc.id) == 0
    assert not file_path.exists()


def test_delete_document_returns_404_if_missing(client):
    r = client.delete(f"/api/v1/documents/{uuid4()}")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_delete_document_idempotent_on_second_call(client, db, qdrant, upload_dir):
    """Спецификация требует 404 на удалённый документ. Это даёт клиенту
    однозначный сигнал, в отличие от 204 на «и так нет»."""
    doc = _make_doc(db, filename="x.txt")
    (upload_dir / f"{doc.id}.txt").write_text("x", encoding="utf-8")

    assert client.delete(f"/api/v1/documents/{doc.id}").status_code == 204
    assert client.delete(f"/api/v1/documents/{doc.id}").status_code == 404


def test_delete_does_not_touch_other_documents(client, db, qdrant, upload_dir):
    keep = _make_doc(db, filename="keep.txt")
    drop = _make_doc(db, filename="drop.txt")
    (upload_dir / f"{keep.id}.txt").write_text("k", encoding="utf-8")
    (upload_dir / f"{drop.id}.txt").write_text("d", encoding="utf-8")
    _seed_qdrant_points(qdrant, keep.id, n=2)
    _seed_qdrant_points(qdrant, drop.id, n=2)

    r = client.delete(f"/api/v1/documents/{drop.id}")
    assert r.status_code == 204

    assert db.get(Document, keep.id) is not None
    assert (upload_dir / f"{keep.id}.txt").exists()
    assert _qdrant_count(qdrant, keep.id) == 2


def test_delete_no_file_on_disk_still_succeeds(client, db, qdrant, upload_dir):
    """Если файл уже отсутствует (например, был удалён вручную или
    после неполного предыдущего DELETE) — операция всё равно должна
    завершиться 204, а не 500."""
    doc = _make_doc(db, filename="orphan.txt")
    # Файл намеренно не создаём.

    r = client.delete(f"/api/v1/documents/{doc.id}")
    assert r.status_code == 204
    assert db.get(Document, doc.id) is None


def test_delete_qdrant_failure_returns_500(client, db, qdrant, upload_dir, monkeypatch):
    """При сбое Qdrant DELETE отдаёт 500. Запись в БД и файл должны
    остаться — клиент сможет повторить операцию."""
    doc = _make_doc(db, filename="x.txt")
    file_path = upload_dir / f"{doc.id}.txt"
    file_path.write_text("x", encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(qdrant, "delete", boom)

    r = client.delete(f"/api/v1/documents/{doc.id}")
    assert r.status_code == 500
    assert db.get(Document, doc.id) is not None
    assert file_path.exists()