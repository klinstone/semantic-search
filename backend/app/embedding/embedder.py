"""Эмбеддер чанков и поисковых запросов на sentence-transformers.

Работает только с семейством intfloat/multilingual-e5-* (base/large/small).
Карточка этих моделей требует двух вещей, которые здесь зашиты:

1. Префикс роли. К документу добавляется ``"passage: "``, к запросу —
   ``"query: "``. Без префиксов качество поиска заметно падает: модель
   училась с разделением через эти маркеры. Для других семейств моделей
   (MiniLM, BGE, mE5-mistral) префиксы другие — этот эмбеддер для них
   не подойдёт без правок.

2. L2-нормализация выходных векторов. Делает Cosine-расстояние в Qdrant
   эквивалентным скалярному произведению, и сравнения косинусной близости
   между разными запусками устойчивы. Эндпоинт /search и тесты опираются
   на этот инвариант.
"""
import logging

from sentence_transformers import SentenceTransformer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# Консервативный размер батча для CPU-инференса. e5-base ~110M параметров;
# при batch=8 пиковое потребление памяти на батч укладывается в несколько
# сотен МБ. На GPU можно поднимать до 32-64 без ущерба.
DEFAULT_BATCH_SIZE = 8


class Embedder:
    """Stateful обёртка над SentenceTransformer.

    Тяжёлый объект — держит загруженную модель в памяти (~1.1 ГБ для
    e5-base). Создаётся один раз в lifespan приложения и переиспользуется
    для всех запросов. Параллельные вызовы из разных корутин/потоков
    допустимы: SentenceTransformer.encode под капотом работает с torch,
    который сам управляет блокировками в собственном backend.
    """

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
        """Прогрев: первый encode на CPU существенно медленнее последующих
        из-за ленивой инициализации ядер PyTorch. Вызывается из lifespan,
        чтобы первый пользовательский запрос не платил эту цену.
        """
        logger.info("warming up embedder")
        self.embed_query("warmup")