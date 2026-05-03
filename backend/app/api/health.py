import logging

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.config import settings
from app.storage.database import engine

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    """Возвращает 200 со статусом каждой зависимости, даже если она недоступна."""
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