"""Межхранилищные операции с документами (Postgres + Qdrant + диск)."""
import logging
from pathlib import Path
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
from sqlalchemy.orm import Session

from app.models.document import Document
from app.storage.files import MIME_TO_EXT, delete_file

logger = logging.getLogger(__name__)


def delete_document_data(
    document: Document,
    db: Session,
    qdrant: QdrantClient,
    collection_name: str,
    upload_dir: Path,
) -> None:
    """Удаляет документ из всех хранилищ: Qdrant → Postgres (CASCADE) → диск.

    Каждый шаг идемпотентен. Исключения пробрасываются вызывающему коду.
    """
    # 1. Qdrant
    qdrant.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document.id)),
                    )
                ]
            )
        ),
    )

    # 2. PostgreSQL — CASCADE удалит chunks
    db.delete(document)
    db.commit()

    # 3. Файл с диска
    ext = MIME_TO_EXT.get(document.mime_type)
    if ext is not None:
        file_path = upload_dir / f"{document.id}.{ext}"
        delete_file(file_path)

    logger.info("deleted document %s", document.id)