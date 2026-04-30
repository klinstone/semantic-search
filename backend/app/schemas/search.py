"""Pydantic-схемы для эндпоинта /search."""
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Тело POST /search.

    document_ids: None — поиск по всем документам; пустой [] — заведомо
    пустой результат (не «по всем»). Различение даёт UI явный способ
    сказать «искать в выделенных».
    """

    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=50)
    document_ids: list[UUID] | None = None


class SearchHit(BaseModel):
    """Один результат поиска."""

    chunk_id: UUID
    document_id: UUID
    document_filename: str
    text: str
    score: float
    chunk_index: int
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Ответ POST /search."""

    query: str
    results: list[SearchHit]
    total_found: int
    took_ms: int