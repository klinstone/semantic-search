"""Пайплайн индексации документов: парсинг → чанкинг → векторизация → сохранение.

Сначала пишет в Postgres (источник истины), затем в Qdrant (векторный индекс). При сбое Qdrant компенсирует это удалением уже вставленных чанков из Postgres. chunk.id == qdrant point.id — этот UUID связывает два хранилища.
"""
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.embedding import Embedder
from app.ingestion.chunker import TextChunk, chunk_text
from app.ingestion.exceptions import (
    CorruptFileError,
    EmptyTextError,
    ParserError,
    UnsupportedFormatError,
)
from app.ingestion.parser import parse_file
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.storage.files import MIME_TO_EXT

logger = logging.getLogger(__name__)


# Ошибки, которые относятся к содержимому документа, а не к инфраструктуре.
# Их сообщения показываются пользователю в DocumentDetail.error_message.
_USER_FACING_ERRORS = (
    CorruptFileError,
    EmptyTextError,
    UnsupportedFormatError,
)


@dataclass(frozen=True)
class IndexResult:
    """Итог успешной индексации (для логов и тестов)."""
    document_id: UUID
    chunks_count: int
    text_length: int


class IndexingError(Exception):
    """Внутренняя ошибка пайплайна, не относящаяся к содержимому файла."""


class IngestionService:
    """Оркестратор индексации.

    Создаётся один на процесс. Тяжёлые зависимости (эмбеддер, токенайзер
    через эмбеддер) переиспользуются между документами. Postgres-сессия
    и Qdrant-клиент передаются на каждый вызов — они привязаны к scope
    конкретной фоновой задачи.
    """

    def __init__(
        self,
        embedder: Embedder,
        target_tokens: int,
        overlap_tokens: int,
        collection_name: str,
        upload_dir: Path,
    ) -> None:
        self._embedder = embedder
        self._target_tokens = target_tokens
        self._overlap_tokens = overlap_tokens
        self._collection = collection_name
        self._upload_dir = upload_dir

    def index_document(
        self,
        document_id: UUID,
        db: Session,
        qdrant: QdrantClient,
    ) -> IndexResult:
        """Индексирует документ. Переходы статусов: pending → processing → indexed|failed.

        Никогда не оставляет документ в статусе 'processing' после возврата.
        Выбрасывает IndexingError при сбое (статус уже установлен в 'failed').
        """
        document = db.get(Document, document_id)
        if document is None:
            logger.warning("document %s not found, nothing to index", document_id)
            raise IndexingError(f"document {document_id} not found")

        document.status = DocumentStatus.PROCESSING
        document.error_message = None
        db.commit()

        try:
            return self._run(document, db, qdrant)
        except _USER_FACING_ERRORS as exc:
            self._mark_failed(db, document_id, str(exc))
            raise IndexingError(str(exc)) from exc
        except Exception as exc:
            logger.exception("indexing failed for document %s", document_id)
            self._mark_failed(db, document_id, "internal error during indexing")
            raise IndexingError("internal error during indexing") from exc

    def _run(
        self,
        document: Document,
        db: Session,
        qdrant: QdrantClient,
    ) -> IndexResult:
        document_id = document.id

        # 1. Извлечение текста из файла.
        ext = MIME_TO_EXT[document.mime_type]
        file_path = self._upload_dir / f"{document_id}.{ext}"
        if not file_path.exists():
            # Отсутствие файла считаем ошибкой для пользователя
            raise CorruptFileError(f"file for document {document_id} not found on disk")
        text = parse_file(file_path, document.mime_type)
        text_length = len(text)

        # 2. Разбиение на чанки.
        chunks = chunk_text(
            text,
            tokenizer=self._embedder.tokenizer,
            target_tokens=self._target_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        if not chunks:
            raise EmptyTextError(f"no chunks produced from {document.filename}")

        logger.info(
            "document %s: %d chars, %d chunks",
            document_id, text_length, len(chunks),
        )

        # 3. Векторизуем до записи — чтобы не осталось сиротских чанков, если модель упадет.
        vectors = self._embedder.embed_passages([ch.text for ch in chunks])
        if len(vectors) != len(chunks):
            # Защита от тихой рассинхронизации между списками.
            raise IndexingError(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        # 4. Очистка любых предыдущих частичных данных (идемпотентная переиндексация).
        self._delete_existing(document_id, db, qdrant)

        # 5. Postgres сначала — это источник истины.
        chunk_ids = self._insert_chunks(document_id, chunks, db)

        # 6. Qdrant. При сбое — компенсация: чанки из БД удаляются,
        #    документ помечается failed (через внешний обработчик).
        try:
            self._upsert_points(document_id, chunk_ids, chunks, vectors, qdrant)
        except Exception as exc:
            logger.exception(
                "qdrant upsert failed for document %s, rolling back chunks",
                document_id,
            )
            self._compensate_chunks(document_id, db)
            raise IndexingError(f"qdrant upsert failed: {exc}") from exc

        # 7. Финализация документа.
        document.status = DocumentStatus.INDEXED
        document.chunks_count = len(chunks)
        document.text_length = text_length
        document.indexed_at = datetime.now(UTC)
        document.error_message = None
        db.commit()

        logger.info("document %s indexed (%d chunks)", document_id, len(chunks))
        return IndexResult(
            document_id=document_id,
            chunks_count=len(chunks),
            text_length=text_length,
        )

    def _delete_existing(
        self,
        document_id: UUID,
        db: Session,
        qdrant: QdrantClient,
    ) -> None:
        # Qdrant: удаление по фильтру, тип точек выбирается автоматически.
        qdrant.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
        )
        # Postgres: удаление через bulk DELETE с фиксацией изменений.
        # ORM-объекты с identity не нужны — мы работаем с массой строк.
        db.execute(delete(Chunk).where(Chunk.document_id == document_id))
        db.commit()

    def _insert_chunks(
        self,
        document_id: UUID,
        chunks: list[TextChunk],
        db: Session,
    ) -> list[UUID]:
        chunk_ids: list[UUID] = []
        rows: list[Chunk] = []
        for idx, ch in enumerate(chunks):
            chunk_id = uuid4()
            chunk_ids.append(chunk_id)
            rows.append(Chunk(
                id=chunk_id,
                document_id=document_id,
                chunk_index=idx,
                text=ch.text,
                char_start=ch.char_start,
                char_end=ch.char_end,
                chunk_metadata=dict(ch.metadata),
            ))
        db.add_all(rows)
        db.commit()
        return chunk_ids

    def _upsert_points(
        self,
        document_id: UUID,
        chunk_ids: list[UUID],
        chunks: list[TextChunk],
        vectors: list[list[float]],
        qdrant: QdrantClient,
    ) -> None:
        # chunks принимаем только для проверки длин в strict=True; кроме
        # количества из них здесь ничего не нужно — текст уже попал в
        # Postgres, в payload Qdrant он не дублируется.
        if not (len(chunks) == len(chunk_ids) == len(vectors)):
            raise IndexingError(
                f"length mismatch: chunks={len(chunks)} ids={len(chunk_ids)} "
                f"vectors={len(vectors)}"
            )
        points: list[PointStruct] = []
        for idx, (chunk_id, vector) in enumerate(zip(chunk_ids, vectors, strict=True)):
            payload = {
                "document_id": str(document_id),
                "chunk_index": idx,
            }
            points.append(PointStruct(
                id=str(chunk_id),
                vector=vector,
                payload=payload,
            ))
        qdrant.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )

    def _compensate_chunks(self, document_id: UUID, db: Session) -> None:
        """Откат вставленных в БД чанков при сбое Qdrant.

        Свои исключения изолируем: компенсация не должна потерять
        исходную ошибку из вызывающего кода.
        """
        try:
            db.rollback()
            db.execute(delete(Chunk).where(Chunk.document_id == document_id))
            db.commit()
        except Exception:
            logger.exception(
                "compensation rollback failed for document %s; "
                "manual cleanup may be required", document_id,
            )

    def _mark_failed(self, db: Session, document_id: UUID, message: str) -> None:
        """Атомарный перевод документа в failed.

        Используется как из user-facing веток (обрезанные сообщения видны
        пользователю), так и из catch-all (с замаскированным "internal
        error"). Откатывает любые незакоммиченные изменения от текущей
        транзакции до записи финального статуса.
        """
        try:
            db.rollback()
            # Перечитываем документ: ORM-объект мог стать stale после
            # rollback, особенно если перед этим были закоммичены чанки.
            doc = db.get(Document, document_id)
            if doc is None:
                return
            doc.status = DocumentStatus.FAILED
            doc.error_message = message[:1000]  # защита от мегабайтных traceback'ов
            db.commit()
        except Exception:
            logger.exception(
                "failed to mark document %s as FAILED", document_id,
            )

__all__ = [
    "IndexingError",
    "IndexResult",
    "IngestionService",
    "ParserError",
]