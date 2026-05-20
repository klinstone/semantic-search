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
from app.reranking import Reranker
from app.search.service import SearchService

logging.basicConfig(
    level=settings.app_log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _ensure_qdrant_collection(client: QdrantClient) -> None:
    """Убеждается, что коллекция Qdrant существует и имеет ожидаемую размерность векторов."""
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
    """Загружает модель эмбеддингов и сверяет ее размерность с конфигом."""
    embedder = Embedder(settings.embedding_model)
    if embedder.dim != settings.embedding_dim:
        raise RuntimeError(
            f"model {settings.embedding_model} produces dim={embedder.dim}, "
            f"but EMBEDDING_DIM={settings.embedding_dim}"
        )
    embedder.warmup()
    return embedder


def _build_reranker() -> Reranker | None:
    """Загружает cross-encoder, если включён в конфиге. Иначе None."""
    if not settings.reranking_enabled:
        logger.info("reranking disabled (RERANKING_ENABLED=false)")
        return None
    reranker = Reranker(settings.reranking_model)
    reranker.warmup()
    return reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting up (env=%s, version=%s)", settings.app_env, settings.app_version)

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info("upload directory: %s", settings.upload_dir)

    # Сначала эмбеддер — быстро падаем, если модель и конфиг не совпадают.
    embedder = _build_embedder()
    app.state.embedder = embedder

    # Реранкер опционален. Если выключен в конфиге — None, SearchService
    # принимает это и работает в исходном режиме без переранжирования.
    reranker = _build_reranker()
    app.state.reranker = reranker

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _ensure_qdrant_collection(qdrant)
    app.state.qdrant = qdrant

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
        reranker=reranker,
        rerank_pool_factor=settings.reranking_pool_factor,
    )

    yield

    logger.info("shutting down")
    qdrant.close()


app = FastAPI(
    title="Semantic Search API",
    version=settings.app_version,
    lifespan=lifespan,
)

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