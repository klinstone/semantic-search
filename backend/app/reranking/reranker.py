"""Cross-encoder реранкинг для уточнения порядка семантической выдачи.

После dense retrieval из Qdrant получается top-N кандидатов; cross-encoder
переоценивает каждую пару (query, passage) совместно — это медленнее
независимых dense-векторов, но точнее. Применяется к узкому пулу
(обычно 30–50 кандидатов на запрос), финальный top-K возвращается
по скорам cross-encoder'а.

BAAI/bge-reranker-v2-m3 — мультиязычная модель, обученная на MIRACL
с русским в составе, подходит и для русскоязычных, и для англоязычных
корпусов без дополнительной настройки.

Один экземпляр на процесс, создаётся в lifespan, переиспользуется через
app.state.reranker.

Веса грузятся в fp16: для инференса разница в качестве пренебрежимо
мала (~0.5% на BEIR), а память и скорость инференса улучшаются вдвое.
Критично для CPU без AVX-512 и для GPU с 8GB VRAM, где fp32-веса крупных
моделей вроде bge-reranker-v2-m3 не помещаются и paging'уются в RAM.

max_length=512 — сегмент, на который cross-encoder режет пары
(query, passage) для совместного кодирования. Чанки в основном проекте
~450 токенов укладываются полностью; длинные пассажи внешних бенчмарков
обрезаются — это стандартное поведение IR-моделей.
"""
import logging

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 512


class Reranker:
    """Обёртка над CrossEncoder."""

    def __init__(self, model_name: str, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        logger.info("loading reranker model %s (max_length=%d)", model_name, max_length)
        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
            model_kwargs={"torch_dtype": torch.float16},
            trust_remote_code=True
        )
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Переранжирует пары (id, text) по релевантности к query.

        Возвращает top_k пар (id, score), отсортированных по убыванию
        score. Score cross-encoder'а — это логит, не вероятность,
        и не ограничен диапазоном [0, 1]. Корректно сравнивать только
        в рамках одного запроса.

        Пустой список кандидатов даёт пустой результат без обращения
        к модели.
        """
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def warmup(self) -> None:
        """Прогрев CUDA/PyTorch dummy-вызовом. Первый predict ленивый
        и заметно медленнее последующих — платим за инициализацию здесь,
        чтобы пользовательские запросы не зависели от прогрева.
        """
        logger.info("warming up reranker")
        self._model.predict([("warmup", "warmup")], show_progress_bar=False)