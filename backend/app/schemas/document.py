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