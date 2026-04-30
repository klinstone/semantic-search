"""HTTP API для документов."""
import logging
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, Response, UploadFile
from qdrant_client import QdrantClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_ingestion_service, get_qdrant
from app.api.errors import AppError
from app.config import settings
from app.ingestion.pipeline import IndexingError, IngestionService
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.schemas.document import (
    DocumentDetail,
    DocumentList,
    DocumentListItem,
    DocumentResponse,
)
from app.schemas.errors import ErrorResponse
from app.storage.database import SessionLocal, get_db
from app.storage.documents import delete_document_data
from app.storage.files import (
    ALLOWED_EXTENSIONS,
    EXT_TO_MIME,
    MIME_TO_EXT,
    check_magic_bytes,
    delete_file,
    save_upload,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])

# Лимиты пагинации фиксированы спецификацией.
LIST_DEFAULT_LIMIT = 20
LIST_MAX_LIMIT = 100


def _validate_upload(file: UploadFile) -> None:
    """Валидирует файл по цепочке: размер → имя → расширение → MIME → magic bytes.

    Кидает AppError на первой же проблеме. Не сохраняет файл.
    """
    # 1. Размер
    if file.size is None or file.size == 0:
        raise AppError(
            code="INVALID_FILE",
            message="File is empty",
            status_code=422,
        )
    if file.size > settings.max_upload_size_bytes:
        raise AppError(
            code="FILE_TOO_LARGE",
            message=f"File exceeds {settings.max_upload_size_mb}MB limit",
            status_code=413,
            details={
                "size_bytes": file.size,
                "max_bytes": settings.max_upload_size_bytes,
            },
        )

    # 2. Имя файла
    if not file.filename:
        raise AppError(
            code="INVALID_FILE",
            message="Filename is missing",
            status_code=422,
        )

    # 3. Расширение
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise AppError(
            code="UNSUPPORTED_FILE_TYPE",
            message=f"File extension '.{ext}' is not supported",
            status_code=415,
            details={"allowed_extensions": sorted(ALLOWED_EXTENSIONS)},
        )

    # 4. MIME-тип должен соответствовать расширению
    expected_mime = EXT_TO_MIME[ext]
    if file.content_type != expected_mime:
        raise AppError(
            code="UNSUPPORTED_FILE_TYPE",
            message=(
                f"MIME type '{file.content_type}' does not match "
                f"declared extension '.{ext}' (expected '{expected_mime}')"
            ),
            status_code=415,
            details={"declared_mime": file.content_type, "expected_mime": expected_mime},
        )

    # 5. Magic bytes — проверяем содержимое.
    # Читаем первые 8 байт и перематываем курсор обратно.
    file.file.seek(0)
    header = file.file.read(8)
    file.file.seek(0)

    if not check_magic_bytes(file.content_type, header):
        raise AppError(
            code="INVALID_FILE",
            message=f"File content does not match declared type '{file.content_type}'",
            status_code=422,
        )


def _run_indexing(
    document_id: UUID,
    ingestion: IngestionService,
    qdrant: QdrantClient,
) -> None:
    """Точка входа фоновой задачи.

    Создаёт собственную сессию (request scope уже закрыт к моменту запуска
    задачи) и делегирует работу IngestionService. IndexingError ловим
    тут — пайплайн уже зафиксировал статус failed в БД, дополнительных
    действий не требуется.
    """
    db = SessionLocal()
    try:
        ingestion.index_document(document_id, db, qdrant)
    except IndexingError:
        # Статус уже выставлен в FAILED внутри пайплайна.
        pass
    except Exception:
        # Подстраховка от исключений, которые пайплайн не превратил
        # в IndexingError.
        logger.exception("background indexing crashed for document %s", document_id)
    finally:
        db.close()


@router.post(
    "/documents",
    status_code=201,
    response_model=DocumentResponse,
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        422: {"model": ErrorResponse, "description": "Invalid or empty file"},
    },
    summary="Upload a document",
)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    ingestion: IngestionService = Depends(get_ingestion_service),
    qdrant: QdrantClient = Depends(get_qdrant),
) -> DocumentResponse:
    """Загружает документ. Возвращает 201 со статусом pending; индексация
    запускается в фоне. Опрос статуса — через GET /documents/{id}.
    """
    _validate_upload(file)

    document_id = uuid4()
    ext = MIME_TO_EXT[file.content_type]
    file_path = settings.upload_dir / f"{document_id}.{ext}"

    # Шаг 1: файл на диск.
    try:
        save_upload(file, file_path)
    except OSError:
        logger.exception("failed to save upload to %s", file_path)
        raise AppError(
            code="STORAGE_ERROR",
            message="Failed to save uploaded file",
            status_code=500,
        )

    # Шаг 2: запись в БД. При ошибке — откатываем файл с диска.
    try:
        document = Document(
            id=document_id,
            filename=file.filename[:255],  # обрезаем до длины VARCHAR(255)
            mime_type=file.content_type,
            size_bytes=file.size,
            status=DocumentStatus.PENDING,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception:
        logger.exception("failed to create document record, rolling back file")
        delete_file(file_path)
        raise AppError(
            code="STORAGE_ERROR",
            message="Failed to create document record",
            status_code=500,
        )

    # Шаг 3: фоновая индексация.
    background_tasks.add_task(_run_indexing, document_id, ingestion, qdrant)

    logger.info(
        "uploaded document %s: %s (%d bytes, %s)",
        document_id,
        document.filename,
        document.size_bytes,
        document.mime_type,
    )
    return DocumentResponse.model_validate(document)


@router.get(
    "/documents",
    response_model=DocumentList,
    summary="List documents",
)
def list_documents(
    limit: Annotated[int, Query(ge=1, le=LIST_MAX_LIMIT)] = LIST_DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: Annotated[DocumentStatus | None, Query()] = None,
    db: Session = Depends(get_db),
) -> DocumentList:
    """Возвращает страницу документов, отсортированных по uploaded_at DESC.

    Параметры пагинации жёсткие: limit ∈ [1, 100], offset ≥ 0. Невалидные
    значения отдают 422 от FastAPI до входа в обработчик. status, если
    задан, сужает выдачу до документов с этим статусом.
    """
    base = select(Document)
    count_base = select(func.count()).select_from(Document)

    if status is not None:
        base = base.where(Document.status == status)
        count_base = count_base.where(Document.status == status)

    # uploaded_at DESC — основная сортировка из спецификации; id — tie-breaker
    # для детерминированного порядка между документами с одинаковым
    # значением uploaded_at (например, batch-загрузка).
    stmt = (
        base.order_by(Document.uploaded_at.desc(), Document.id)
        .limit(limit)
        .offset(offset)
    )
    items = db.execute(stmt).scalars().all()
    total = db.execute(count_base).scalar_one()

    return DocumentList(
        items=[DocumentListItem.model_validate(d) for d in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetail,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
    summary="Get document details",
)
def get_document(
    document_id: UUID,
    db: Session = Depends(get_db),
) -> DocumentDetail:
    """Полная информация о документе включая статус индексации и сообщение
    об ошибке (если status=failed). Используется фронтом для polling'а.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise AppError(
            code="NOT_FOUND",
            message=f"Document {document_id} not found",
            status_code=404,
        )
    return DocumentDetail.model_validate(document)


@router.delete(
    "/documents/{document_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
    summary="Delete a document",
)
def delete_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant),
) -> Response:
    """Удаляет документ полностью: точки в Qdrant → запись в БД (CASCADE
    уберёт чанки) → файл с диска. Порядок зафиксирован в спецификации API.

    На повторный DELETE отвечает 404 (запись уже удалена). 500 возможен,
    если упал Qdrant — клиент может ретраить, операция идемпотентна.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise AppError(
            code="NOT_FOUND",
            message=f"Document {document_id} not found",
            status_code=404,
        )

    delete_document_data(
        document=document,
        db=db,
        qdrant=qdrant,
        collection_name=settings.qdrant_collection,
        upload_dir=settings.upload_dir,
    )
    return Response(status_code=204)