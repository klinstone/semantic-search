"""Alembic migration environment."""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings
from app.storage.database import Base

# Register models for Alembic autogenerate
from app.models import Chunk, Document  # noqa: F401

config = context.config

# Настройки логирования из alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override DSN from application settings
config.set_main_option("sqlalchemy.url", settings.postgres_dsn)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """'Offline' mode: генерирует SQL без подключения к БД."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Включает отслеживание изменений типов колонок и server_default —
        # без этого autogenerate пропустит часть изменений.
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """'Online' mode: подключается к БД и применяет миграции."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()