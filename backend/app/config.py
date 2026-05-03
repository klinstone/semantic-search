from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_version: str = "0.1.0"
    app_env: str = Field(default="development")
    app_log_level: str = Field(default="INFO")

    # --- PostgreSQL ---
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # --- Qdrant ---
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "document_chunks"

    # --- Embedding model ---
    # Должно совпадать с размерностью модели; проверяется при запуске.
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768

    # --- Chunking ---
    # Размер чанка в токенах модели. Запас от 512 покрывает префикс "passage: " и спецтокены.
    chunk_target_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # --- Storage ---
    upload_dir: Path = Path("/data/uploads")
    max_upload_size_mb: int = 10

    # --- CORS ---
    # JSON-list в env, например: CORS_ORIGINS=["http://localhost:5173"]
    cors_origins: list[str] = ["http://localhost:5173"]

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
