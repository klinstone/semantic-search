from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    """Ответ POST /documents — отдаётся сразу после загрузки.

    Не содержит chunks_count и indexed_at, потому что на момент ответа
    индексация ещё не запущена. Эти поля появятся в DocumentDetail
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    size_bytes: int
    mime_type: str
    status: str
    uploaded_at: datetime


class DocumentListItem(BaseModel):
    """Элемент в списке GET /documents."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    size_bytes: int
    mime_type: str
    status: str
    chunks_count: int
    uploaded_at: datetime
    indexed_at: datetime | None


class DocumentList(BaseModel):
    """Контейнер пагинированного ответа GET /documents."""

    items: list[DocumentListItem]
    total: int
    limit: int
    offset: int


class DocumentDetail(BaseModel):
    """Полная информация о документе для GET /documents/{id}."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    size_bytes: int
    mime_type: str
    status: str
    chunks_count: int
    text_length: int | None
    uploaded_at: datetime
    indexed_at: datetime | None
    error_message: str | None