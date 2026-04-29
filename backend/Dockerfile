FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# libmagic1 — пригодится на этапе 4 для валидации файлов по magic bytes.
# curl — для healthcheck'а из docker compose.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Берём бинарник uv из официального образа.
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /usr/local/bin/


WORKDIR /app

# Сначала только метаданные — это даёт кэш слоя с зависимостями.
COPY pyproject.toml ./
COPY uv.lock* ./

RUN uv sync --no-dev

# Код приложения и миграции
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

EXPOSE 8000

# В compose эта команда переопределяется на запуск миграций + uvicorn с reload.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]