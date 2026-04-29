"""Низкоуровневые операции с файлами и константы поддерживаемых форматов."""
import shutil
from pathlib import Path

from fastapi import UploadFile

# Маппинг MIME → расширение для построения пути файла на диске.
# Имя файла на диске: {document_id}.{ext} — оригинальное имя пользователя
# на диск НЕ попадает (защита от path traversal и проблем с кодировками).
MIME_TO_EXT: dict[str, str] = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

# Обратный маппинг — для проверки соответствия расширения и MIME-типа.
EXT_TO_MIME: dict[str, str] = {v: k for k, v in MIME_TO_EXT.items()}

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(MIME_TO_EXT.values())

# Сигнатуры (magic bytes) для проверки соответствия содержимого заявленному типу.
PDF_SIGNATURE = b"%PDF-"
DOCX_SIGNATURE = b"PK\x03\x04"  # ZIP local file header — DOCX это zip


def check_magic_bytes(mime_type: str, header: bytes) -> bool:
    """Проверяет, что первые байты файла соответствуют заявленному MIME-типу.

    Для TXT строго требуется UTF-8 (включая BOM) без null-байтов.
    Если у пользователя файл в cp1251 — пусть конвертирует в UTF-8.
    """
    if mime_type == "application/pdf":
        return header.startswith(PDF_SIGNATURE)
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return header.startswith(DOCX_SIGNATURE)
    if mime_type == "text/plain":
        if b"\x00" in header:
            return False
        try:
            header.decode("utf-8", errors="strict")
            return True
        except UnicodeDecodeError:
            return False
    return False


def save_upload(file: UploadFile, dest: Path) -> None:
    """Сохраняет загруженный файл на диск.

    Перематывает курсор на начало (на случай, если кто-то читал заголовок).
    Использует chunked copy через shutil — не загружает весь файл в память.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    file.file.seek(0)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)


def delete_file(path: Path) -> None:
    """Удаляет файл, не падая если его нет (idempotent)."""
    path.unlink(missing_ok=True)