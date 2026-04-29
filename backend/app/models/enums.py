from enum import StrEnum


class DocumentStatus(StrEnum):
    """Статус обработки документа.

    Хранится в БД как VARCHAR(20) с CHECK-constraint, а не как PG ENUM —
    ALTER ENUM в миграциях работает плохо, проще менять CHECK.

    StrEnum наследуется от str, поэтому `document.status = DocumentStatus.PENDING`
    автоматически попадает в БД как строка "pending".
    """

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"