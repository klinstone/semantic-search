"""Универсальная обёртка эмбеддингов.

Поддерживает настройку ролевых префиксов ("passage: " / "query: " и др.) через конфиг
и применяет L2-нормализацию к выходным векторам. Модели загружаются в FP16
для экономии видеопамяти.
"""
import logging
import torch

from sentence_transformers import SentenceTransformer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Консервативный размер батча для инференса на CPU/слабых GPU.
DEFAULT_BATCH_SIZE = 8


class Embedder:
    """Обёртка над SentenceTransformer. Один экземпляр на процесс, создается в lifespan."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        passage_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        logger.info("loading embedding model %s (FP16)", model_name)
        # Загружаем модель в float16, чтобы она занимала в 2 раза меньше видеопамяти
        self._model = SentenceTransformer(
            model_name,
            model_kwargs={"torch_dtype": torch.float16}
        )
        self._model_name = model_name
        self._batch_size = batch_size
        self._passage_prefix = passage_prefix
        self._query_prefix = query_prefix

    @property
    def dim(self) -> int:
        """Размерность выходного вектора модели."""
        # sentence-transformers переименовал метод в новых версиях;
        # старое имя оставлено как алиас, но даёт FutureWarning.
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        d = getter()
        if d is None:
            raise RuntimeError(
                f"model {self._model_name} did not report embedding dimension"
            )
        return d

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Токенайзер той же модели, что используется для эмбеддинга.

        Чанкер должен мерить длину тем же токенайзером, иначе можно
        переоценить или недооценить размер чанка относительно реального
        контекста модели.
        """
        return self._model.tokenizer

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Векторизует чанки документа.

        К каждому тексту добавляется префикс (если задан). Возвращает
        список L2-нормализованных векторов в порядке входных текстов.
        Пустой вход — пустой выход (без обращения к модели).
        """
        if not texts:
            return []
        prefixed = [self._passage_prefix + t for t in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Векторизует поисковый запрос.

        Пустой/whitespace-only запрос приводит к ValueError — пустой
        запрос на /search не должен доходить до модели.
        """
        if not text or not text.strip():
            raise ValueError("query text must be non-empty")
        prefixed = self._query_prefix + text
        vector = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_queries(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Векторизует список поисковых запросов батчем (для тестов/бенчмарков)."""
        if not texts:
            return []
        prefixed = [self._query_prefix + t for t in texts]
        bs = batch_size if batch_size is not None else self._batch_size
        vectors = self._model.encode(
            prefixed,
            batch_size=bs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def warmup(self) -> None:
        """Холостой прогон encode для ленивой инициализации PyTorch при старте."""
        logger.info("warming up embedder")
        self.embed_query("warmup")