"""Обёртка эмбеддингов для моделей intfloat/multilingual-e5-*.

Требует ролевых префиксов ("passage: " / "query: ") и применяет L2-нормализацию к выходным векторам.
"""
import logging

from sentence_transformers import SentenceTransformer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# Консервативный размер батча для инференса на CPU.
DEFAULT_BATCH_SIZE = 8


class Embedder:
    """Обёртка над SentenceTransformer. Один экземпляр на процесс, создается в lifespan."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        passage_prefix: str = PASSAGE_PREFIX,
        query_prefix: str = QUERY_PREFIX,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        logger.info("loading embedding model %s", model_name)
        self._model = SentenceTransformer(model_name)
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

        К каждому тексту добавляется префикс ``"passage: "``. Возвращает
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

    def warmup(self) -> None:
        """Холостой прогон encode для ленивой инициализации PyTorch при старте."""
        logger.info("warming up embedder")
        self.embed_query("warmup")