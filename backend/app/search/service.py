"""Сервис семантического поиска.

Цепочка одного запроса:
    1. embed_query(text)            — векторизация (~50 мс на CPU)
    2. qdrant.query_points(...)     — top-k по косинусной близости
    3. SELECT chunks WHERE id IN    — загрузка текстов
    4. SELECT documents WHERE id IN — имена для отображения
    5. сборка ответа в порядке Qdrant

Сами тексты чанков хранятся только в Postgres — Qdrant выступает как
индекс по векторам, в его payload только document_id и chunk_index.
Поэтому шаги 3–4 нужны для каждой выдачи.

Шаги 3 и 4 — два независимых SELECT по primary key, а не JOIN.
Это проще тестируется (FakeSession не делает JOIN) и так же быстро на
наших объёмах: top-k обычно ≤ 50 строк.
"""
from __future__ import annotations

import logging
import time
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embedding import Embedder
from app.models.chunk import Chunk
from app.models.document import Document
from app.schemas.search import SearchHit, SearchResponse

logger = logging.getLogger(__name__)


class SearchService:
    """Stateful обёртка над Embedder + Qdrant. Один на процесс."""

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str,
    ) -> None:
        self._embedder = embedder
        self._collection = collection_name

    def search(
        self,
        *,
        query: str,
        limit: int,
        document_ids: list[UUID] | None,
        db: Session,
        qdrant: QdrantClient,
    ) -> SearchResponse:
        t0 = time.perf_counter()

        # Особый случай: явный пустой список = «искать в нуле документов».
        # Не идём в эмбеддер и Qdrant — экономим ~50 мс на CPU.
        if document_ids is not None and len(document_ids) == 0:
            return SearchResponse(
                query=query,
                results=[],
                total_found=0,
                took_ms=_ms(t0),
            )

        # 1. Векторизация запроса.
        query_vector = self._embedder.embed_query(query)

        # 2. Поиск в Qdrant.
        scored_points = qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
            query_filter=_build_filter(document_ids),
            with_payload=True,
        ).points

        if not scored_points:
            return SearchResponse(
                query=query,
                results=[],
                total_found=0,
                took_ms=_ms(t0),
            )

        # 3. Загрузка чанков из БД одним SELECT'ом по PK.
        chunk_ids = [UUID(str(p.id)) for p in scored_points]
        chunks_by_id = self._fetch_chunks(db, chunk_ids)

        # 4. Загрузка документов одним SELECT'ом по PK.
        doc_ids = {ch.document_id for ch in chunks_by_id.values()}
        docs_by_id = self._fetch_documents(db, list(doc_ids))

        # 5. Сборка ответа в порядке Qdrant. Пропускаем точки, у которых
        #    нет соответствия в БД — это последствие race с DELETE
        #    (документ или чанк удалили между search и SELECT).
        results: list[SearchHit] = []
        for sp in scored_points:
            chunk_id = UUID(str(sp.id))
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                logger.info(
                    "search: chunk %s not in DB (likely deleted between "
                    "qdrant search and join)", chunk_id,
                )
                continue
            document = docs_by_id.get(chunk.document_id)
            if document is None:
                logger.info(
                    "search: document %s not in DB (likely deleted)",
                    chunk.document_id,
                )
                continue
            results.append(SearchHit(
                chunk_id=chunk.id,
                document_id=document.id,
                document_filename=document.filename,
                text=chunk.text,
                score=float(sp.score),
                chunk_index=chunk.chunk_index,
                metadata=dict(chunk.chunk_metadata or {}),
            ))

        return SearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            took_ms=_ms(t0),
        )

    @staticmethod
    def _fetch_chunks(db: Session, ids: list[UUID]) -> dict[UUID, Chunk]:
        if not ids:
            return {}
        rows = db.execute(select(Chunk).where(Chunk.id.in_(ids))).scalars().all()
        return {ch.id: ch for ch in rows}

    @staticmethod
    def _fetch_documents(db: Session, ids: list[UUID]) -> dict[UUID, Document]:
        if not ids:
            return {}
        rows = db.execute(select(Document).where(Document.id.in_(ids))).scalars().all()
        return {d.id: d for d in rows}


def _build_filter(document_ids: list[UUID] | None) -> Filter | None:
    """None / пустой список / непустой — три разных режима.
    Вызывающий код должен короткозамкнуть пустой список ДО этого вызова;
    здесь обрабатываем только None и непустой случай.
    """
    if not document_ids:
        return None
    return Filter(must=[FieldCondition(
        key="document_id",
        match=MatchAny(any=[str(uid) for uid in document_ids]),
    )])


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)