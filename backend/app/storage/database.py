from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# Naming convention для constraints — даёт предсказуемые имена в миграциях.
# Без неё Alembic может сгенерировать случайные суффиксы, которые усложняют
# чтение миграций и сравнение между ревизиями.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# pool_pre_ping=True — отлавливает разорванные соединения (например, после
# рестарта postgres-контейнера), пересоздавая их прозрачно для приложения.
engine = create_engine(
    settings.postgres_dsn,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency. Используется через Depends(get_db)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()