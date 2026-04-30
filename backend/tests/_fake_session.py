"""Минимальная in-memory замена SQLAlchemy Session для тестов.

Поддерживает только те операции, которые использует прикладной код:
get / add / add_all / delete / commit / rollback / close,
а также execute() для DELETE и SELECT-запросов вида
``select(Model).where(predicate).order_by(col[.desc()]).limit(L).offset(O)``
и ``select(func.count()).select_from(Model).where(predicate)``.

В where поддерживается конъюнкция из ``col == value`` и ``col.in_(values)``.
Этого достаточно для всех текущих SELECT/DELETE прикладного кода.

Назначение — unit-тесты ОРКЕСТРАЦИИ (handlers, pipeline, search).
Полное поведение SQLAlchemy не воспроизводится: транзакционная семантика
упрощена до уровня "pending до commit, потом отражено в storage".
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy.sql import Delete, Select
from sqlalchemy.sql import operators as sql_ops
from sqlalchemy.sql.functions import count as func_count

from app.models.chunk import Chunk
from app.models.document import Document


class _Result:
    """Эмуляция Result/ScalarResult, достаточная для .scalars().all(),
    .scalar_one(), .scalar(), .all()."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[Any]:
        return list(self._rows)

    def scalar_one(self) -> Any:
        if len(self._rows) != 1:
            raise RuntimeError(f"scalar_one expected 1 row, got {len(self._rows)}")
        return self._rows[0]

    def scalar(self) -> Any:
        return self._rows[0] if self._rows else None


class FakeSession:
    def __init__(self) -> None:
        # storage[Model] = {pk: obj}
        self._storage: dict[type, dict] = defaultdict(dict)
        # Незакоммиченные изменения за текущую "транзакцию".
        self._pending_adds: list[object] = []
        self._pending_deletes: list[tuple[type, object]] = []
        self.commits = 0
        self.rollbacks = 0

    # ---- Session API used by application code ----

    def get(self, model: type, pk):
        return self._storage[model].get(pk)

    def add(self, obj) -> None:
        self._pending_adds.append(obj)

    def add_all(self, objs) -> None:
        self._pending_adds.extend(objs)

    def delete(self, obj) -> None:
        # Аналог Session.delete(obj) — помечает на удаление до commit.
        self._pending_deletes.append((type(obj), obj.id))
        # Эмуляция CASCADE через FK для связки Document → Chunk.
        if isinstance(obj, Document):
            for chunk_id, chunk in list(self._storage[Chunk].items()):
                if chunk.document_id == obj.id:
                    self._pending_deletes.append((Chunk, chunk_id))

    def execute(self, stmt):
        if isinstance(stmt, Delete):
            return self._execute_delete(stmt)
        if isinstance(stmt, Select):
            return self._execute_select(stmt)
        raise NotImplementedError(f"FakeSession.execute: unsupported {type(stmt).__name__}")

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

    def refresh(self, obj) -> None:
        # No-op: после add+commit объект уже идентичен записи в storage.
        return None

    def close(self) -> None:
        self.rollback()

    # ---- internal: SELECT/DELETE evaluators ----

    def _execute_delete(self, stmt: Delete) -> _Result:
        table = stmt.table
        model = _model_for_table(table)
        predicate = _build_predicate(stmt.whereclause)
        for pk, obj in list(self._storage[model].items()):
            if predicate(obj):
                self._pending_deletes.append((model, pk))
        return _Result([])

    def _execute_select(self, stmt: Select) -> _Result:
        # Выбираем источник: либо select(Model), либо select(count()).select_from(Model).
        col_descs = stmt.column_descriptions
        is_count = (
            len(col_descs) == 1
            and isinstance(col_descs[0]["expr"], func_count)
        )

        if is_count:
            from_tables = stmt.get_final_froms()
            if len(from_tables) != 1:
                raise NotImplementedError(
                    "FakeSession select-count expects exactly one FROM"
                )
            model = _model_for_table(from_tables[0])
        else:
            if len(col_descs) != 1 or col_descs[0]["entity"] is None:
                raise NotImplementedError(
                    "FakeSession select expects a single ORM entity"
                )
            model = col_descs[0]["entity"]

        predicate = _build_predicate(stmt.whereclause)
        rows = [obj for obj in self._storage[model].values() if predicate(obj)]

        if is_count:
            return _Result([len(rows)])

        # ORDER BY: поддерживаем UnaryExpression (col.desc()) и обычные Column
        # (по возрастанию). Множественные order_by обрабатываются от последнего
        # к первому, благодаря стабильности sort.
        for clause in reversed(stmt._order_by_clauses):
            col_name, descending = _order_key(clause)
            rows.sort(key=lambda o: getattr(o, col_name), reverse=descending)

        offset = _scalar_clause(stmt._offset_clause) or 0
        limit = _scalar_clause(stmt._limit_clause)
        if limit is not None:
            rows = rows[offset:offset + limit]
        elif offset:
            rows = rows[offset:]

        return _Result(rows)

    # ---- helpers for tests ----

    def all(self, model: type) -> list:
        return list(self._storage[model].values())

    def count(self, model: type) -> int:
        return len(self._storage[model])

    def seed(self, obj) -> None:
        """Кладёт объект в storage в обход pending — для arrange-фазы тестов."""
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


def _build_predicate(where):
    """Превращает whereclause SQLAlchemy в callable: object → bool.

    Поддерживается конъюнкция простых сравнений: col == val и col IN (...).
    """
    if where is None:
        return lambda _obj: True

    clauses = getattr(where, "clauses", [where])
    predicates = []
    for clause in clauses:
        col = getattr(clause.left, "key", None)
        if col is None:
            raise NotImplementedError(f"FakeSession: cannot interpret {clause!r}")
        op = getattr(clause, "operator", None)
        right = clause.right
        val = getattr(right, "value", None)

        if op is sql_ops.in_op:
            allowed = set(val) if val is not None else set()
            predicates.append(lambda obj, c=col, s=allowed: getattr(obj, c) in s)
        else:
            # eq и всё остальное обрабатываем как ==.
            predicates.append(lambda obj, c=col, v=val: getattr(obj, c) == v)

    def combined(obj):
        return all(p(obj) for p in predicates)

    return combined


def _order_key(clause) -> tuple[str, bool]:
    """Возвращает (имя_колонки, descending) для разных форм ORDER BY."""
    descending = False
    element = clause
    # UnaryExpression от .desc()/.asc().
    if hasattr(clause, "modifier") and clause.modifier is not None:
        modifier_name = getattr(clause.modifier, "__name__", "")
        descending = modifier_name == "desc_op"
        element = clause.element
    col_name = getattr(element, "key", None)
    if col_name is None:
        raise NotImplementedError(f"FakeSession: cannot order by {clause!r}")
    return col_name, descending


def _scalar_clause(clause) -> int | None:
    """LIMIT/OFFSET у SQLA — это BindParameter с .value либо None."""
    if clause is None:
        return None
    val = getattr(clause, "value", None)
    if val is None:
        return None
    return int(val)