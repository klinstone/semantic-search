"""Низкоуровневые операции с файлами и константы поддерживаемых форматов."""
import shutil
from pathlib import Path

from fastapi import UploadFile

# Файлы на диске называются {document_id}.{ext} — оригинальные имена не используются.
MIME_TO_EXT: dict[str, str] = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

EXT_TO_MIME: dict[str, str] = {v: k for k, v in MIME_TO_EXT.items()}

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(MIME_TO_EXT.values())

PDF_SIGNATURE = b"%PDF-"
DOCX_SIGNATURE = b"PK\x03\x04"  # ZIP local file header — DOCX это zip


def check_magic_bytes(mime_type: str, header: bytes) -> bool:
    """Проверяет, что заголовок файла соответствует заявленному MIME-типу."""
    if mime_type == "application/pdf":
        return header.startswith(PDF_SIGNATURE)
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return header.startswith(DOCX_SIGNATURE)
    if mime_type == "text/plain":
        return b"\x00" not in header


def save_upload(file: UploadFile, dest: Path) -> None:
    """Сохраняет загруженный файл на диск порциями (chunked copy)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    file.file.seek(0)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)


def delete_file(path: Path) -> None:
    """Удаляет файл, не падая если его нет (idempotent)."""
    path.unlink(missing_ok=True)