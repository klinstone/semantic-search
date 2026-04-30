"""Минимальная in-memory замена SQLAlchemy Session для тестов пайплайна.

Эмулирует только те операции, которые использует IngestionService:
get(Model, pk), add(obj), add_all(objs), commit(), rollback(),
execute(delete(...).where(...)). Этого достаточно для интеграционных
тестов оркестрации; полное поведение SQLAlchemy не воспроизводится.
"""
from __future__ import annotations

from collections import defaultdict

from sqlalchemy.sql import Delete

from app.models.chunk import Chunk
from app.models.document import Document


class FakeSession:
    def __init__(self) -> None:
        # storage[Model] = {pk: obj}
        self._storage: dict[type, dict] = defaultdict(dict)
        # Незакоммиченные изменения за текущую "транзакцию".
        # На MVP это просто список tombstone-операций add/delete для
        # отката через rollback().
        self._pending_adds: list[object] = []
        self._pending_deletes: list[tuple[type, object]] = []
        self.commits = 0
        self.rollbacks = 0

    # ---- public Session API used by pipeline ----

    def get(self, model: type, pk):
        return self._storage[model].get(pk)

    def add(self, obj) -> None:
        self._pending_adds.append(obj)

    def add_all(self, objs) -> None:
        self._pending_adds.extend(objs)

    def execute(self, stmt):
        if not isinstance(stmt, Delete):
            raise NotImplementedError("FakeSession.execute supports only DELETE")
        table = stmt.table
        model = _model_for_table(table)
        # Простейшая поддержка: WHERE column == value.
        # Снимаем фильтр, перебирая текущие строки и сверяя поля.
        criteria = _extract_eq_criteria(stmt)
        for pk, obj in list(self._storage[model].items()):
            if all(getattr(obj, col) == val for col, val in criteria.items()):
                self._pending_deletes.append((model, pk))

    def commit(self) -> None:
        for obj in self._pending_adds:
            model = type(obj)
            self._storage[model][obj.id] = obj
        for model, pk in self._pending_deletes:
            self._storage[model].pop(pk, None)
        self._pending_adds.clear()
        self._pending_deletes.clear()
        self.commits += 1

    def rollback(self) -> None:
        self._pending_adds.clear()
        self._pending_deletes.clear()
        self.rollbacks += 1

    def close(self) -> None:
        self.rollback()

    # ---- helpers for tests ----

    def all(self, model: type) -> list:
        return list(self._storage[model].values())

    def count(self, model: type) -> int:
        return len(self._storage[model])

    def seed(self, obj) -> None:
        """Кладёт объект в storage в обход pending — для арenge-фазы тестов."""
        self._storage[type(obj)][obj.id] = obj


_TABLE_TO_MODEL = {
    Chunk.__table__: Chunk,
    Document.__table__: Document,
}


def _model_for_table(table) -> type:
    model = _TABLE_TO_MODEL.get(table)
    if model is None:
        raise NotImplementedError(f"FakeSession: unknown table {table}")
    return model


def _extract_eq_criteria(stmt: Delete) -> dict[str, object]:
    """Распаковывает простые WHERE col == value из DELETE-запроса."""
    criteria: dict[str, object] = {}
    where = stmt.whereclause
    if where is None:
        return criteria
    # SQLAlchemy представляет цепочку AND как BooleanClauseList; в нашем
    # пайплайне фильтр всегда одиночный (document_id == X).
    clauses = getattr(where, "clauses", [where])
    for clause in clauses:
        # BinaryExpression: clause.left = Column, clause.right = BindParameter
        col = getattr(clause.left, "key", None)
        right = clause.right
        val = getattr(right, "value", None)
        if col is None:
            raise NotImplementedError(f"FakeSession: cannot interpret {clause!r}")
        criteria[col] = val
    return criteria