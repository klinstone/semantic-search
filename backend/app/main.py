import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


from app.api.documents import router as documents_router
from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.config import settings
from app.embedding import Embedder
from app.ingestion.pipeline import IngestionService
from app.search.service import SearchService

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


def _build_embedder() -> Embedder:
    """Загружает модель и сверяет её фактическую размерность с конфигом.

    Несовпадение возможно при смене EMBEDDING_MODEL без обновления
    EMBEDDING_DIM — лучше упасть на старте, чем складывать в Qdrant
    векторы неправильной длины и ловить ошибки в рантайме.
    """
    embedder = Embedder(settings.embedding_model)
    if embedder.dim != settings.embedding_dim:
        raise RuntimeError(
            f"model {settings.embedding_model} produces dim={embedder.dim}, "
            f"but EMBEDDING_DIM={settings.embedding_dim}"
        )
    embedder.warmup()
    return embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting up (env=%s, version=%s)", settings.app_env, settings.app_version)

    # Создаём директорию для загрузок, если её нет.
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info("upload directory: %s", settings.upload_dir)

    # Эмбеддер до Qdrant: если модель не подходит к конфигу, нет смысла
    # трогать коллекцию — упасть нужно как можно раньше.
    embedder = _build_embedder()
    app.state.embedder = embedder

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _ensure_qdrant_collection(qdrant)
    app.state.qdrant = qdrant

    # Сервисы — синглтоны. Тяжёлых ресурсов сами не держат, тяжесть в
    # эмбеддере и Qdrant-клиенте. БД-сессия и qdrant-клиент передаются
    # на каждый вызов.
    app.state.ingestion = IngestionService(
        embedder=embedder,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        collection_name=settings.qdrant_collection,
        upload_dir=settings.upload_dir,
    )
    app.state.search = SearchService(
        embedder=embedder,
        collection_name=settings.qdrant_collection,
    )

    yield

    logger.info("shutting down")
    qdrant.close()


app = FastAPI(
    title="Semantic Search API",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS для разработки фронта. В проде origins сужают до конкретных доменов.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

register_exception_handlers(app)


app.include_router(health_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")