"""Сервис семантического поиска.

Цепочка одного запроса (без reranking):
    1. embed_query(text)            — векторизация (~50 мс на CPU)
    2. qdrant.query_points(...)     — top-k по косинусной близости
    3. SELECT chunks WHERE id IN    — загрузка текстов
    4. SELECT documents WHERE id IN — имена для отображения
    5. сборка ответа в порядке Qdrant

С включённым reranker'ом меняется шаг 2: из Qdrant запрашивается
расширенный пул limit*pool_factor кандидатов. После шагов 3–4
добавляется промежуточный шаг: cross-encoder переранжирует
выживший пул и возвращает финальные top-K с собственными скорами.

Сами тексты чанков хранятся только в Postgres — Qdrant выступает как
индекс по векторам, в его payload только document_id и chunk_index.
Поэтому шаги 3–4 нужны для каждой выдачи, и они же дают reranker'у
текст пассажа для совместной оценки с запросом.

Шаги 3 и 4 — два независимых SELECT по primary key, а не JOIN.
Это проще тестируется (FakeSession не делает JOIN) и так же быстро на
наших объёмах: top-k обычно ≤ 50 строк, с reranking ≤ 250.
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
from app.reranking import Reranker
from app.schemas.search import SearchHit, SearchResponse

logger = logging.getLogger(__name__)


class SearchService:
    """Stateful обёртка над Embedder + Qdrant (+ опционально Reranker).
    Один на процесс."""

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str,
        reranker: Reranker | None = None,
        rerank_pool_factor: int = 5,
    ) -> None:
        self._embedder = embedder
        self._collection = collection_name
        self._reranker = reranker
        self._rerank_pool_factor = rerank_pool_factor

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

        # 2. Поиск в Qdrant. С reranker'ом берём расширенный пул, чтобы
        #    дать CE шанс переставить релевантные кандидаты, лежащие
        #    за пределами top-K dense-выдачи.
        qdrant_limit = limit * self._rerank_pool_factor if self._reranker else limit

        scored_points = qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=qdrant_limit,
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

        # 5a. Фильтрация выживших точек: пропускаем те, чьи chunk/document
        #     отсутствуют в БД (race с DELETE между Qdrant-поиском и SELECT).
        survived: list[tuple[float, Chunk, Document]] = []
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
            survived.append((float(sp.score), chunk, document))

        # 5b. Reranking (если включён) переупорядочивает выживший пул.
        #     Скор Qdrant'а заменяется на CE-логит, который имеет смысл
        #     только в рамках одного запроса.
        if self._reranker is not None and survived:
            candidates = [(str(chunk.id), chunk.text) for _, chunk, _ in survived]
            reranked = self._reranker.rerank(query, candidates, top_k=limit)
            by_id = {str(chunk.id): (chunk, doc) for _, chunk, doc in survived}
            results = [
                _make_hit(*by_id[cid], score=score)
                for cid, score in reranked
            ]
        else:
            # Без reranker'а — порядок Qdrant, обрезаем до limit.
            results = [
                _make_hit(chunk, doc, score=score)
                for score, chunk, doc in survived[:limit]
            ]

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


def _make_hit(chunk: Chunk, document: Document, *, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.id,
        document_id=document.id,
        document_filename=document.filename,
        text=chunk.text,
        score=score,
        chunk_index=chunk.chunk_index,
        metadata=dict(chunk.chunk_metadata or {}),
    )


def _build_filter(document_ids: list[UUID] | None) -> Filter | None:
    """None / пустой список / непустой — три разных режима.
    Вызывающий код должен короткозамкнуть пустой список ДО этого вызова;
    здесь обрабатываем только None и непустой случай.
    """
    if document_ids is None:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="document_id",
                match=MatchAny(any=[str(d) for d in document_ids]),
            )
        ]
    )


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)