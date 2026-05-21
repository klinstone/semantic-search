"""Бенчмарк качества поиска на BEIR-форматных датасетах.

Поддерживаются:
  - RusBEIR (русский): rus-scifact, rus-xquad, rus-tydiqa
  - BEIR (английский): scifact

Запуск:
    docker compose exec backend python -m evaluation.benchmark \
        --datasets rus-scifact scifact

Скрипт обходит обычный пайплайн ingestion: пассажи бенчмарка уже
сегментированы и привязаны к qrels, поэтому не должны проходить через
чанкер. Пассажи кладутся в отдельную Qdrant-коллекцию,
тексты держатся в памяти процесса. Postgres не задействован.

Конфигурации:
  - semantic        — только dense retrieval
  - semantic+rerank — dense retrieval (top limit*pool) → cross-encoder
                      переранжирует → top limit
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass

# Глушит warning huggingface_hub про symlinks на Windows без прав. Кэш всё
# равно работает, просто без оптимизации дедупликации.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from datasets import load_dataset as hf_load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.embedding import Embedder
from app.reranking import Reranker

logger = logging.getLogger(__name__)

# Реестр поддерживаемых датасетов: имя → (hf_repo, hf_repo_qrels, preferred_split).
# Если preferred_split отсутствует в qrels-репозитории, берётся первый доступный.
# BEIR и RusBEIR используют одну и ту же структуру файлов (corpus/queries/qrels),
# поэтому загрузчик у них общий.
DATASETS = {
    # Русский (RusBEIR)
    "rus-scifact": ("kngrg/rus-scifact", "kngrg/rus-scifact-qrels", "test"),
    "rus-xquad":   ("kngrg/rus-xquad",   "kngrg/rus-xquad-qrels",   "dev"),
    "rus-tydiqa":  ("kngrg/rus-tydiqa",  "kngrg/rus-tydiqa-qrels",  "dev"),
    # Английский (оригинальный BEIR)
    "scifact":     ("BeIR/scifact",      "BeIR/scifact-qrels",      "test"),
}

ALLOWED_CONFIGS = ["semantic", "semantic+rerank"]

DEFAULT_FINAL_K = 10
DEFAULT_INDEX_BATCH_SIZE = 64
DEFAULT_QUERY_BATCH_SIZE = 64
DEFAULT_RERANK_POOL_FACTOR = 5

# time_ms — среднее время на запрос (эмбеддинг + retrieval + опционально
# reranking) в миллисекундах. Эмбеддинг считается батчем и амортизируется,
# что отражает поведение серверного режима: модель уже загружена,
# запросы идут потоком.
METRIC_KEYS = ["ndcg", "mrr", "p", "r", "time_ms"]
METRIC_LABELS = ["nDCG@10", "MRR", "P@10", "R@10", "Time(ms)"]


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass
class Dataset:
    name: str
    corpus: dict[str, Passage]
    queries: list[Query]
    qrels: dict[str, dict[str, int]]


# --- Загрузка датасетов ---

def _first_split(ds_dict) -> list:
    """Берёт первый split из DatasetDict. На разных репозиториях имена
    splits отличаются (corpus/queries vs train), хелпер унифицирует."""
    name = next(iter(ds_dict))
    return ds_dict[name]


def _load_qrels(qrels_repo: str, preferred_split: str):
    """Грузит qrels, пытаясь использовать preferred_split. Если такого
    split нет, берёт первый доступный."""
    full = hf_load_dataset(qrels_repo)
    if preferred_split in full:
        return full[preferred_split]
    available = list(full.keys())
    actual = available[0]
    logger.warning(
        "qrels split '%s' not found in %s; using '%s' (available: %s)",
        preferred_split, qrels_repo, actual, available,
    )
    return full[actual]


def _get_field(row, candidates: tuple[str, ...], context: str):
    """Возвращает значение по первому найденному имени поля. Если не найдено —
    выводит реальные ключи строки в ошибке."""
    for c in candidates:
        if c in row:
            return row[c]
    raise KeyError(
        f"none of {candidates} found in {context} row; "
        f"available keys: {list(row.keys())}"
    )


def load_beir_dataset(name: str) -> Dataset:
    """Грузит BEIR-форматный датасет с HuggingFace.
    Поддерживает и русские (RusBEIR), и английские (BEIR) датасеты."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}; available: {list(DATASETS)}")
    repo, qrels_repo, preferred_split = DATASETS[name]

    logger.info("loading dataset %s from HuggingFace", name)
    corpus_rows = _first_split(hf_load_dataset(repo, "corpus"))
    queries_rows = _first_split(hf_load_dataset(repo, "queries"))
    qrels_rows = _load_qrels(qrels_repo, preferred_split)

    # Имена колонок и типы id могут отличаться между репозиториями.
    # Приводим всё к строкам, пробуем несколько вариантов имён.
    corpus: dict[str, Passage] = {}
    for row in corpus_rows:
        did = str(_get_field(row, ("_id", "doc_id", "id", "docid"), "corpus"))
        text = _get_field(row, ("text", "passage"), "corpus")
        title = row.get("title") or ""
        corpus[did] = Passage(doc_id=did, title=title, text=text)

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_rows:
        qid = str(_get_field(row, ("query-id", "query_id", "qid"), "qrels"))
        did = str(_get_field(row, ("corpus-id", "corpus_id", "doc_id", "did"), "qrels"))
        score = int(_get_field(row, ("score", "relevance"), "qrels"))
        qrels.setdefault(qid, {})[did] = score

    queries: list[Query] = []
    for row in queries_rows:
        qid = str(_get_field(row, ("_id", "query_id", "id", "qid"), "queries"))
        text = _get_field(row, ("text", "query"), "queries")
        if qid in qrels and any(s > 0 for s in qrels[qid].values()):
            queries.append(Query(query_id=qid, text=text))

    logger.info(
        "loaded %s: %d passages, %d queries with positive qrels",
        name, len(corpus), len(queries),
    )
    return Dataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def passage_for_embedding(p: Passage) -> str:
    """Стандарт BEIR: title + text при индексации корпуса."""
    if p.title:
        return f"{p.title}. {p.text}"
    return p.text


# --- Индексация ---

def index_corpus(
    dataset: Dataset,
    embedder: Embedder,
    qdrant: QdrantClient,
    collection: str,
    batch_size: int,
) -> None:
    """Индексирует корпус в Qdrant. Если коллекция уже есть и совпадает по
    числу точек и размерности — пропускает (кэш между прогонами)."""
    expected = len(dataset.corpus)
    if qdrant.collection_exists(collection):
        info = qdrant.get_collection(collection)
        actual_dim = info.config.params.vectors.size
        actual_count = info.points_count or 0
        if actual_count == expected and actual_dim == embedder.dim:
            logger.info(
                "collection %s already indexed (%d points), skipping",
                collection, expected,
            )
            return
        logger.info("collection %s stale or wrong size, recreating", collection)
        qdrant.delete_collection(collection)

    logger.info("creating collection %s (dim=%d)", collection, embedder.dim)
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )

    doc_ids = list(dataset.corpus.keys())
    texts = [passage_for_embedding(dataset.corpus[d]) for d in doc_ids]
    total = len(texts)
    total_batches = math.ceil(total / batch_size)

    t0 = time.perf_counter()
    for batch_num, start in enumerate(range(0, total, batch_size), start=1):
        chunk_ids = doc_ids[start:start + batch_size]
        chunk_texts = texts[start:start + batch_size]
        vectors = embedder.embed_passages(chunk_texts)
        # ID точки в Qdrant — последовательный int (BEIR doc_id может быть
        # произвольной строкой, реальный doc_id храним в payload).
        points = [
            PointStruct(
                id=start + i,
                vector=vectors[i],
                payload={"doc_id": chunk_ids[i]},
            )
            for i in range(len(chunk_ids))
        ]
        qdrant.upsert(collection_name=collection, points=points, wait=True)
        indexed = min(start + batch_size, total)
        logger.info(
            "[indexing %s] batch %d/%d (%d/%d passages, %.1fs elapsed)",
            dataset.name, batch_num, total_batches, indexed, total,
            time.perf_counter() - t0,
        )

    logger.info(
        "indexed %d passages in %.1fs", total, time.perf_counter() - t0,
    )


# --- Эмбеддинг запросов батчем ---

def embed_queries_batch(
    embedder: Embedder,
    queries: list[Query],
    batch_size: int,
) -> tuple[list[list[float]], float]:
    """Эмбеддит все запросы одной партией. Возвращает векторы и среднее
    время на запрос в секундах.

    Имитирует поведение прод-сервера: модель загружена, запросы идут
    потоком. Single-query inference на GPU крайне неэффективен из-за
    kernel launch overhead — батч в десятки раз быстрее.
    """
    texts = [q.text for q in queries]
    t0 = time.perf_counter()
    vectors = embedder.embed_queries(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    return vectors, elapsed / len(queries)


# --- Метрики ---

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/позиция первого релевантного, 0 если ни одного нет в выдаче."""
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    """Normalized DCG@k. Числитель: (2^rel - 1) / log2(rank + 1) — стандарт BEIR."""
    dcg = sum(
        (2 ** qrel.get(d, 0) - 1) / math.log2(i + 1)
        for i, d in enumerate(retrieved[:k], start=1)
    )
    ideal = sorted((r for r in qrel.values() if r > 0), reverse=True)[:k]
    idcg = sum(
        (2 ** r - 1) / math.log2(i + 1)
        for i, r in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg > 0 else 0.0


# --- Оценка ---

def evaluate(
    dataset: Dataset,
    embedder: Embedder,
    qdrant: QdrantClient,
    collection: str,
    final_k: int,
    query_embed_batch_size: int,
    reranker: Reranker | None,
    rerank_pool_factor: int,
) -> dict[str, float]:
    """Прогоняет все запросы датасета через выбранную конфигурацию.
    Возвращает усреднённые по запросам метрики плюс среднее время запроса.

    Если reranker не None, из Qdrant забираем расширенный пул кандидатов
    (final_k * rerank_pool_factor), затем CE переранжирует и оставляет
    top final_k. Без reranker'а — стандартный flow с top-K из Qdrant.
    """
    metrics: dict[str, list[float]] = {k: [] for k in METRIC_KEYS if k != "time_ms"}
    retrieval_times: list[float] = []
    rerank_times: list[float] = []

    qdrant_pool = final_k * rerank_pool_factor if reranker else final_k

    # Шаг 1: пре-эмбеддинг всех запросов одной партией.
    total_q = len(dataset.queries)
    logger.info(
        "[%s] embedding %d queries (batch_size=%d)",
        dataset.name, total_q, query_embed_batch_size,
    )
    query_vectors, avg_embed_sec = embed_queries_batch(
        embedder, dataset.queries, query_embed_batch_size,
    )
    logger.info(
        "[%s] embedded %d queries: %.2fms per query (avg)",
        dataset.name, total_q, avg_embed_sec * 1000,
    )

    # Шаг 2: цикл по запросам — retrieval, опциональный rerank, метрики.
    log_every = max(50, total_q // 20)
    t0 = time.perf_counter()
    for i, (query, vector) in enumerate(zip(dataset.queries, query_vectors), start=1):
        qrel = dataset.qrels.get(query.query_id, {})
        relevant = {d for d, s in qrel.items() if s > 0}
        if not relevant:
            continue

        t_retrieval = time.perf_counter()
        points = qdrant.query_points(
            collection_name=collection,
            query=vector,
            limit=qdrant_pool,
            with_payload=True,
        ).points
        retrieval_times.append(time.perf_counter() - t_retrieval)

        candidate_ids = [p.payload["doc_id"] for p in points]

        if reranker is not None:
            # CE требует тексты пассажей. Тексты у нас в памяти из dataset.corpus.
            pairs = [(cid, passage_for_embedding(dataset.corpus[cid])) for cid in candidate_ids]
            t_rerank = time.perf_counter()
            reranked = reranker.rerank(query.text, pairs, top_k=final_k)
            rerank_times.append(time.perf_counter() - t_rerank)
            retrieved = [cid for cid, _ in reranked]
        else:
            retrieved = candidate_ids[:final_k]

        metrics["ndcg"].append(ndcg_at_k(retrieved, qrel, final_k))
        metrics["mrr"].append(reciprocal_rank(retrieved, relevant))
        metrics["p"].append(precision_at_k(retrieved, relevant, final_k))
        metrics["r"].append(recall_at_k(retrieved, relevant, final_k))

        if i % log_every == 0 or i == total_q:
            logger.info(
                "[querying %s] %d/%d queries (%.1fs elapsed)",
                dataset.name, i, total_q, time.perf_counter() - t0,
            )

    n = len(metrics["ndcg"])
    result = {k: (sum(v) / n if n else 0.0) for k, v in metrics.items()}
    avg_retrieval_sec = (sum(retrieval_times) / len(retrieval_times)) if retrieval_times else 0.0
    avg_rerank_sec = (sum(rerank_times) / len(rerank_times)) if rerank_times else 0.0
    result["time_ms"] = (avg_embed_sec + avg_retrieval_sec + avg_rerank_sec) * 1000

    if reranker is not None:
        logger.info(
            "[%s] timing — embed: %.2fms, retrieval: %.2fms, rerank: %.2fms, total: %.2fms",
            dataset.name,
            avg_embed_sec * 1000, avg_retrieval_sec * 1000,
            avg_rerank_sec * 1000, result["time_ms"],
        )
    else:
        logger.info(
            "[%s] timing — embed: %.2fms, retrieval: %.2fms, total: %.2fms",
            dataset.name,
            avg_embed_sec * 1000, avg_retrieval_sec * 1000, result["time_ms"],
        )
    return result


# --- Вывод ---

def _fmt_metric(key: str, value: float) -> str:
    """Время в мс — два знака после запятой; качественные метрики — четыре."""
    fmt = "9.2f" if key == "time_ms" else "9.4f"
    return f"{value:>{fmt}}"


def _row(label: str, label_w: int, metrics: dict[str, float]) -> str:
    parts = [f"{label:<{label_w}}"]
    parts.extend(_fmt_metric(k, metrics[k]) for k in METRIC_KEYS)
    return " | ".join(parts)


def _header(label: str, label_w: int) -> tuple[str, str]:
    parts = [f"{label:<{label_w}}"]
    parts.extend(f"{lbl:>9}" for lbl in METRIC_LABELS)
    head = " | ".join(parts)
    sep = "-" * len(head)
    return head, sep


def print_results(
    results: dict[str, dict[str, dict[str, float]]],
    datasets: list[str],
    configs: list[str],
) -> None:
    width = max(15, max(len(d) for d in datasets) + 2)

    for config in configs:
        print()
        print("=" * 80)
        print(f"Configuration: {config}")
        print("=" * 80)
        head, sep = _header("Dataset", width)
        print(head)
        print(sep)
        for ds in datasets:
            print(_row(ds, width, results[config][ds]))
        if len(datasets) > 1:
            avg = {
                k: sum(results[config][ds][k] for ds in datasets) / len(datasets)
                for k in METRIC_KEYS
            }
            print(sep)
            print(_row("Average", width, avg))

    if len(configs) > 1:
        print()
        print("=" * 80)
        print("Summary (averaged across datasets)")
        print("=" * 80)
        cw = max(15, max(len(c) for c in configs) + 2)
        head, sep = _header("Configuration", cw)
        print(head)
        print(sep)
        for config in configs:
            avg = {
                k: sum(results[config][ds][k] for ds in datasets) / len(datasets)
                for k in METRIC_KEYS
            }
            print(_row(config, cw, avg))


# --- Точка входа ---

def _configure_logging(level: str) -> None:
    """Настраивает логирование, заглушая болтливые сторонние библиотеки."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Бенчмарк качества поиска на BEIR-форматных датасетах",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["rus-scifact"],
        choices=list(DATASETS),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["semantic"],
        choices=ALLOWED_CONFIGS,
    )
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=DEFAULT_INDEX_BATCH_SIZE,
        help="Размер батча при индексации корпуса (эмбеддинг + upsert).",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=DEFAULT_QUERY_BATCH_SIZE,
        help="Размер батча для эмбеддинга запросов.",
    )
    parser.add_argument(
        "--rerank-pool-factor",
        type=int,
        default=DEFAULT_RERANK_POOL_FACTOR,
        help="Из Qdrant берётся top (final_k * pool_factor) кандидатов "
             "перед reranking'ом.",
    )
    args = parser.parse_args()

    _configure_logging(settings.app_log_level)

    # Инициализируем эмбеддер с префиксами из конфига
    embedder = Embedder(
        model_name=settings.embedding_model,
        batch_size=args.index_batch_size,
        passage_prefix=settings.embedding_passage_prefix,
        query_prefix=settings.embedding_query_prefix,
    )
    embedder.warmup()

    # Reranker берём из настроек .env
    reranker: Reranker | None = None
    if any("rerank" in c for c in args.configs):
        reranker = Reranker(settings.reranking_model)
        reranker.warmup()

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # results[config][dataset_name] = {ndcg, mrr, p, r, time_ms}
    results: dict[str, dict[str, dict[str, float]]] = {c: {} for c in args.configs}

    for ds_name in args.datasets:
        dataset = load_beir_dataset(ds_name)

        # Экранируем спецсимволы в имени модели, чтобы Qdrant не ругался
        safe_model_name = settings.embedding_model.replace("/", "_").replace("-", "_")
        collection = f"bench_{ds_name}_{safe_model_name}_{embedder.dim}"

        index_corpus(dataset, embedder, qdrant, collection, args.index_batch_size)

        for config in args.configs:
            use_rerank = "rerank" in config
            logger.info("evaluating %s on %s", config, ds_name)
            metrics = evaluate(
                dataset, embedder, qdrant, collection,
                args.final_k, args.query_batch_size,
                reranker=reranker if use_rerank else None,
                rerank_pool_factor=args.rerank_pool_factor,
            )
            results[config][ds_name] = metrics
            logger.info(
                "%s on %s: nDCG=%.4f MRR=%.4f P@10=%.4f R@10=%.4f time=%.2fms",
                config, ds_name,
                metrics["ndcg"], metrics["mrr"], metrics["p"], metrics["r"],
                metrics["time_ms"],
            )

    print_results(results, args.datasets, args.configs)
    qdrant.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())