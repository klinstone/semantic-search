"""Эмбеддинг чанков и поисковых запросов."""
from app.embedding.embedder import (
    DEFAULT_BATCH_SIZE,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    Embedder,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "Embedder",
    "PASSAGE_PREFIX",
    "QUERY_PREFIX",
]