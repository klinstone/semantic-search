from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    Uuid,
    func,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.database import Base

if TYPE_CHECKING:
    from app.models.document import Document


class Chunk(Base):
    """Чанк текста из документа.

    UUID этого чанка используется как ID точки в Qdrant —
    это ключевое проектное решение, связывающее два хранилища:
    один и тот же UUID живёт в PostgreSQL.chunks.id и в Qdrant.points[].id.
    """

    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Сам текст чанка — единственный источник истины,
    # в Qdrant тексты не дублируются.
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Смещения в исходном тексте, в символах. Для будущей подсветки.
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Гибкие метаданные (страница в PDF, заголовок раздела и т.п.).
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'{}'::jsonb")
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.id} doc={self.document_id} idx={self.chunk_index}>"