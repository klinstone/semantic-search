import logging

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.config import settings
from app.storage.database import engine

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    """
    Liveness + readiness в одном эндпоинте.
    Возвращает 200 даже если зависимости лежат — статус каждой зависимости
    выдаётся в поле dependencies. Это намеренно: эндпоинт показывает СОСТОЯНИЕ,
    а не падает из-за зависимостей.
    """
    deps: dict[str, str] = {}

    # Postgres
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        deps["postgres"] = "ok"
    except Exception as e:
        logger.warning("postgres health check failed: %s", e)
        deps["postgres"] = "error"

    # Qdrant
    try:
        request.app.state.qdrant.get_collections()
        deps["qdrant"] = "ok"
    except Exception as e:
        logger.warning("qdrant health check failed: %s", e)
        deps["qdrant"] = "error"

    return {
        "status": "ok",
        "version": settings.app_version,
        "dependencies": deps,
    }