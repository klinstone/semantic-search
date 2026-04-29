import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.api.health import router as health_router
from app.config import settings

logging.basicConfig(
    level=settings.app_log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _ensure_qdrant_collection(client: QdrantClient) -> None:
    """
    Идемпотентно гарантирует, что коллекция document_chunks существует
    и имеет правильную размерность.

    Если коллекция уже есть, но dim не совпадает с конфигом — это значит,
    что модель эмбеддингов поменяли без переиндексации. Падаем с понятной
    ошибкой, а не молча принимаем чужие векторы.
    """
    name = settings.qdrant_collection
    expected_dim = settings.embedding_dim

    if not client.collection_exists(name):
        logger.info("creating qdrant collection %s (dim=%d)", name, expected_dim)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=expected_dim, distance=Distance.COSINE),
        )
        return

    info = client.get_collection(name)
    actual_dim = info.config.params.vectors.size
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"qdrant collection '{name}' has dim={actual_dim}, "
            f"but EMBEDDING_DIM={expected_dim}. "
            f"reindex required: drop collection or run reindex script."
        )
    logger.info("qdrant collection %s ready (dim=%d)", name, actual_dim)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting up (env=%s, version=%s)", settings.app_env, settings.app_version)

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _ensure_qdrant_collection(qdrant)
    app.state.qdrant = qdrant

    yield

    logger.info("shutting down")
    qdrant.close()


app = FastAPI(
    title="Semantic Search API",
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health_router, prefix="/api/v1")