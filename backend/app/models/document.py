from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.database import Base

if TYPE_CHECKING:
    from app.models.chunk import Chunk


class Document(Base):
    """Загруженный пользователем документ.

    UUID генерируется на стороне Python (default=uuid4), а не БД, потому что
    ID нужен ДО коммита: файл кладём на диск как {document_id}.{ext},
    а только потом коммитим запись.
    """

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Используем VARCHAR с CHECK constraint для статусов.
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Денормализация: количество чанков. Обновляется в ingestion-пайплайне
    # после успешной индексации, чтобы выдача списка не делала JOIN+COUNT.
    chunks_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

    # Длина извлечённого текста в символах. NULL пока не индексирован.
    text_length: Mapped[int | None] = mapped_column(Integer, nullable=True)

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # passive_deletes=True — полагаемся на ON DELETE CASCADE в БД,
    # ORM не делает отдельные DELETE для каждого чанка.
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'indexed', 'failed')",
            name="status",
        ),
        Index("ix_documents_status", "status"),
        Index("ix_documents_uploaded_at", "uploaded_at"),
    )

    def __repr__(self) -> str:
        return f"<Document {self.id} {self.filename!r} status={self.status!r}>"