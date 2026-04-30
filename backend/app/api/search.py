"""HTTP API семантического поиска."""
import logging

from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.api.deps import get_qdrant, get_search_service
from app.api.errors import AppError
from app.schemas.errors import ErrorResponse
from app.schemas.search import SearchRequest, SearchResponse
from app.search.service import SearchService
from app.storage.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={422: {"model": ErrorResponse, "description": "Invalid query"}},
    summary="Semantic search across indexed documents",
)
def search(
    body: SearchRequest,
    service: SearchService = Depends(get_search_service),
    db: Session = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant),
) -> SearchResponse:
    """Возвращает топ-K чанков, наиболее близких к запросу по косинусной мере.

    query валидируется pydantic-схемой на уровне маршрута (1..1000 символов,
    limit 1..50). Дополнительная проверка на whitespace-only запрос — здесь:
    после strip пустая строка не имеет смысла как поисковый запрос, а
    pydantic min_length=1 этого не отлавливает.
    """
    if not body.query.strip():
        raise AppError(
            code="INVALID_QUERY",
            message="query must not be empty or whitespace-only",
            status_code=422,
        )

    response = service.search(
        query=body.query,
        limit=body.limit,
        document_ids=body.document_ids,
        db=db,
        qdrant=qdrant,
    )
    logger.info(
        "search: query_len=%d limit=%d → %d results in %d ms",
        len(body.query), body.limit, response.total_found, response.took_ms,
    )
    return response