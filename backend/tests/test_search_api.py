"""Тесты POST /search.

Делятся на две группы:
  1. SearchService напрямую: проверяем оркестрацию (qdrant.query_points →
     SELECT chunks → SELECT documents → сборка). Эмбеддер мокаем,
     Qdrant — :memory: с реальными точками.
  2. HTTP-слой через TestClient: валидация запроса, формат ответа,
     обработка ошибок. SearchService подменяется на простой stub.
"""
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from app.api.deps import get_qdrant, get_search_service
from app.api.errors import register_exception_handlers
from app.api.search import router as search_router
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.schemas.search import SearchHit, SearchResponse
from app.search.service import SearchService
from app.storage.database import get_db
from tests._fake_session import FakeSession

COLLECTION = "test_search"
VECTOR_DIM = 4


# ----------------------- shared fixtures -----------------------


@pytest.fixture
def db() -> FakeSession:
    return FakeSession()


@pytest.fixture
def qdrant() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    return client


@pytest.fixture
def fake_embedder():
    """Эмбеддер-заглушка с управляемой векторизацией.

    embed_query возвращает фиксированный вектор. embed_passages не
    используется в /search вообще, поэтому реального не нужно — модель
    e5 в тестах /search не загружается.
    """
    embedder = MagicMock()
    embedder.embed_query.return_value = [1.0, 0.0, 0.0, 0.0]
    return embedder


@pytest.fixture
def service(fake_embedder) -> SearchService:
    return SearchService(embedder=fake_embedder, collection_name=COLLECTION)


def _make_doc(db: FakeSession, *, filename: str = "doc.txt") -> Document:
    doc = Document(
        id=uuid4(),
        filename=filename,
        mime_type="text/plain",
        size_bytes=100,
        status=DocumentStatus.INDEXED,
        chunks_count=0,
        text_length=None,
        uploaded_at=datetime.now(UTC),
    )
    db.seed(doc)
    return doc


def _make_chunk(
    db: FakeSession,
    document_id: UUID,
    *,
    text: str = "chunk text",
    chunk_index: int = 0,
    chunk_id: UUID | None = None,
    metadata: dict | None = None,
) -> Chunk:
    cid = chunk_id or uuid4()
    chunk = Chunk(
        id=cid,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        char_start=0,
        char_end=len(text),
        chunk_metadata=metadata or {},
    )
    db.seed(chunk)
    return chunk


def _seed_point(
    qdrant: QdrantClient,
    chunk_id: UUID,
    document_id: UUID,
    *,
    vector: list[float] | None = None,
    chunk_index: int = 0,
) -> None:
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=str(chunk_id),
            vector=vector or [1.0, 0.0, 0.0, 0.0],
            payload={"document_id": str(document_id), "chunk_index": chunk_index},
        )],
        wait=True,
    )


# ===================== SearchService unit tests =====================


def test_search_returns_results_in_qdrant_order(service, db, qdrant, fake_embedder):
    doc = _make_doc(db, filename="manual.txt")
    # Три чанка, каждый со своим вектором. Запрос совпадает с третьим точно.
    c1 = _make_chunk(db, doc.id, text="alpha", chunk_index=0)
    c2 = _make_chunk(db, doc.id, text="beta", chunk_index=1)
    c3 = _make_chunk(db, doc.id, text="gamma", chunk_index=2)
    _seed_point(qdrant, c1.id, doc.id, vector=[0.0, 1.0, 0.0, 0.0], chunk_index=0)
    _seed_point(qdrant, c2.id, doc.id, vector=[0.0, 0.0, 1.0, 0.0], chunk_index=1)
    _seed_point(qdrant, c3.id, doc.id, vector=[1.0, 0.0, 0.0, 0.0], chunk_index=2)

    fake_embedder.embed_query.return_value = [1.0, 0.0, 0.0, 0.0]

    resp = service.search(
        query="что-то", limit=10, document_ids=None, db=db, qdrant=qdrant,
    )
    assert isinstance(resp, SearchResponse)
    assert resp.query == "что-то"
    # Самый ближний по cosine (то же направление) должен идти первым.
    assert resp.results[0].chunk_id == c3.id
    assert resp.results[0].text == "gamma"
    assert resp.results[0].document_filename == "manual.txt"
    assert resp.results[0].chunk_index == 2
    assert 0.99 <= resp.results[0].score <= 1.0
    assert resp.total_found == 3
    assert resp.took_ms >= 0


def test_search_respects_limit(service, db, qdrant):
    doc = _make_doc(db)
    for i in range(5):
        c = _make_chunk(db, doc.id, text=f"chunk-{i}", chunk_index=i)
        _seed_point(qdrant, c.id, doc.id, chunk_index=i)

    resp = service.search(query="q", limit=2, document_ids=None, db=db, qdrant=qdrant)
    assert len(resp.results) == 2
    assert resp.total_found == 2


def test_search_returns_empty_for_empty_collection(service, db, qdrant):
    resp = service.search(query="q", limit=10, document_ids=None, db=db, qdrant=qdrant)
    assert resp.results == []
    assert resp.total_found == 0


def test_search_filters_by_document_ids(service, db, qdrant):
    doc_a = _make_doc(db, filename="a.txt")
    doc_b = _make_doc(db, filename="b.txt")
    c_a = _make_chunk(db, doc_a.id, text="from a")
    c_b = _make_chunk(db, doc_b.id, text="from b")
    _seed_point(qdrant, c_a.id, doc_a.id)
    _seed_point(qdrant, c_b.id, doc_b.id)

    resp = service.search(
        query="q", limit=10, document_ids=[doc_b.id], db=db, qdrant=qdrant,
    )
    assert len(resp.results) == 1
    assert resp.results[0].document_id == doc_b.id
    assert resp.results[0].text == "from b"


def test_search_with_empty_document_ids_returns_no_results(
    service, db, qdrant, fake_embedder
):
    """Пустой список — это «искать в нуле документов», не «во всём».
    Важный контракт: эмбеддер и Qdrant даже не вызываются."""
    doc = _make_doc(db)
    c = _make_chunk(db, doc.id)
    _seed_point(qdrant, c.id, doc.id)

    resp = service.search(
        query="q", limit=10, document_ids=[], db=db, qdrant=qdrant,
    )
    assert resp.results == []
    assert resp.total_found == 0
    fake_embedder.embed_query.assert_not_called()


def test_search_skips_chunks_missing_in_db(service, db, qdrant):
    """Точка есть в Qdrant, но её chunk-запись удалили из БД (race
    с DELETE). Handler должен пропустить такую точку, не падать."""
    doc = _make_doc(db)
    real_chunk = _make_chunk(db, doc.id, text="real", chunk_index=0)
    # Сирота в Qdrant — точка с UUID, которого нет в Chunk-таблице.
    orphan_chunk_id = uuid4()
    _seed_point(qdrant, real_chunk.id, doc.id, chunk_index=0)
    _seed_point(qdrant, orphan_chunk_id, doc.id, chunk_index=99)

    resp = service.search(query="q", limit=10, document_ids=None, db=db, qdrant=qdrant)
    returned = {hit.chunk_id for hit in resp.results}
    assert returned == {real_chunk.id}
    assert resp.total_found == 1


def test_search_skips_chunks_with_missing_document(service, db, qdrant):
    """Чанк в БД есть, документа уже нет (race теоретический, FK CASCADE
    обычно его не пускает, но проверяем устойчивость кода)."""
    doc = _make_doc(db)
    chunk = _make_chunk(db, doc.id, text="orphan-doc")
    _seed_point(qdrant, chunk.id, doc.id)

    # Удаляем документ напрямую из storage, оставляя чанк.
    db._storage[Document].pop(doc.id)

    resp = service.search(query="q", limit=10, document_ids=None, db=db, qdrant=qdrant)
    assert resp.results == []


def test_search_passes_metadata_through(service, db, qdrant):
    doc = _make_doc(db)
    chunk = _make_chunk(
        db, doc.id, text="page-aware", metadata={"page": 7, "source": "ocr"},
    )
    _seed_point(qdrant, chunk.id, doc.id)

    resp = service.search(query="q", limit=10, document_ids=None, db=db, qdrant=qdrant)
    assert resp.results[0].metadata == {"page": 7, "source": "ocr"}


def test_search_query_is_embedded_with_query_prefix(service, db, qdrant, fake_embedder):
    """Эмбеддер должен получить ровно текст запроса. Префикс "query: "
    добавляется внутри Embedder.embed_query, его проверяет test_embedder.
    Здесь следим, что SearchService не делает ничего лишнего с текстом."""
    service.search(
        query="как работает поиск", limit=5, document_ids=None, db=db, qdrant=qdrant,
    )
    fake_embedder.embed_query.assert_called_once_with("как работает поиск")


def test_search_qdrant_filter_uses_document_ids(service, db, qdrant, monkeypatch):
    captured = {}

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        # Минимальный валидный ответ: пустой список точек.
        return type("R", (), {"points": []})()

    monkeypatch.setattr(qdrant, "query_points", fake_query_points)

    doc_id = uuid4()
    service.search(
        query="q", limit=7, document_ids=[doc_id], db=db, qdrant=qdrant,
    )

    assert captured["limit"] == 7
    flt = captured["query_filter"]
    assert isinstance(flt, Filter)
    assert isinstance(flt.must[0], FieldCondition)
    assert flt.must[0].key == "document_id"
    assert isinstance(flt.must[0].match, MatchAny)
    assert flt.must[0].match.any == [str(doc_id)]


# ===================== HTTP-layer tests =====================


@pytest.fixture
def http_client(db, qdrant, service):
    """Тестовый FastAPI-инстанс без lifespan, с подменёнными зависимостями."""
    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.include_router(search_router, prefix="/api/v1")

    test_app.dependency_overrides[get_db] = lambda: db
    test_app.dependency_overrides[get_qdrant] = lambda: qdrant
    test_app.dependency_overrides[get_search_service] = lambda: service

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c


def test_http_search_returns_structured_response(http_client, db, qdrant):
    doc = _make_doc(db, filename="article.pdf")
    chunk = _make_chunk(db, doc.id, text="some content", chunk_index=3)
    _seed_point(qdrant, chunk.id, doc.id, chunk_index=3)

    r = http_client.post("/api/v1/search", json={"query": "hello", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "hello"
    assert body["total_found"] == 1
    assert body["took_ms"] >= 0
    [hit] = body["results"]
    assert hit["chunk_id"] == str(chunk.id)
    assert hit["document_id"] == str(doc.id)
    assert hit["document_filename"] == "article.pdf"
    assert hit["text"] == "some content"
    assert hit["chunk_index"] == 3
    assert "score" in hit
    assert "metadata" in hit


def test_http_search_default_limit_is_10(http_client, db, qdrant, monkeypatch, service):
    captured = {}
    real_search = service.search

    def spy(**kwargs):
        captured.update(kwargs)
        return real_search(**kwargs)

    monkeypatch.setattr(service, "search", spy)

    http_client.post("/api/v1/search", json={"query": "hi"})
    assert captured["limit"] == 10


def test_http_search_rejects_empty_query(http_client):
    r = http_client.post("/api/v1/search", json={"query": "", "limit": 5})
    assert r.status_code == 422


def test_http_search_rejects_whitespace_only_query(http_client):
    r = http_client.post("/api/v1/search", json={"query": "   \n  ", "limit": 5})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_QUERY"


def test_http_search_rejects_too_long_query(http_client):
    r = http_client.post("/api/v1/search", json={"query": "x" * 1001})
    assert r.status_code == 422


def test_http_search_rejects_invalid_limit(http_client):
    assert http_client.post(
        "/api/v1/search", json={"query": "q", "limit": 0}
    ).status_code == 422
    assert http_client.post(
        "/api/v1/search", json={"query": "q", "limit": 51}
    ).status_code == 422


def test_http_search_invalid_document_ids_returns_422(http_client):
    r = http_client.post(
        "/api/v1/search",
        json={"query": "q", "document_ids": ["not-a-uuid"]},
    )
    assert r.status_code == 422


def test_http_search_empty_document_ids_returns_no_results(http_client, db, qdrant):
    doc = _make_doc(db)
    chunk = _make_chunk(db, doc.id)
    _seed_point(qdrant, chunk.id, doc.id)

    r = http_client.post(
        "/api/v1/search", json={"query": "q", "document_ids": []},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == []
    assert body["total_found"] == 0


def test_http_search_no_results(http_client):
    """Пустая коллекция → 200 с пустым списком, не 500."""
    r = http_client.post("/api/v1/search", json={"query": "q"})
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == []
    assert body["total_found"] == 0