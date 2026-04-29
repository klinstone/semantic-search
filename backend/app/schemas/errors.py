from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Единый формат ошибок API.

    Используется в декораторах эндпоинтов через responses={...: {"model": ErrorResponse}}
    для генерации корректной OpenAPI-документации.
    """

    error: ErrorDetail