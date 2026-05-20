"""Эмбеддинг чанков и поисковых запросов."""
from app.embedding.embedder import (
    DEFAULT_BATCH_SIZE,
    Embedder,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "Embedder",
]