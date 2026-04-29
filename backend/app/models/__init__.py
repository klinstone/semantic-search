"""SQLAlchemy ORM models.

Импорт этого пакета регистрирует все модели в Base.metadata, что необходимо
для Alembic autogenerate, чтобы он увидел таблицы.
"""
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.enums import DocumentStatus

__all__ = ["Chunk", "Document", "DocumentStatus"]