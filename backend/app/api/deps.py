"""Общие зависимости FastAPI-эндпоинтов.

Доступ к синглтонам, инициализированным в lifespan (app.state).
Использование через Depends() позволяет тестам подменять реализации.
"""
from fastapi import Request
from qdrant_client import QdrantClient

from app.embedding import Embedder
from app.ingestion.pipeline import IngestionService
from app.search.service import SearchService


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_qdrant(request: Request) -> QdrantClient:
    return request.app.state.qdrant


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search