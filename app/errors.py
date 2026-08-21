"""Unified error envelope: code, message, retryable, details, correlation_id."""

import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code: int = 400
    code: str = "app_error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        if retryable is not None:
            self.retryable = retryable
        self.details = details or {}


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class Conflict(AppError):
    status_code = 409
    code = "conflict"


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"


class PolicyBlocked(AppError):
    status_code = 422
    code = "policy_blocked"


class CostCapExceeded(AppError):
    status_code = 402
    code = "cost_cap_exceeded"


class UpstreamError(AppError):
    status_code = 502
    code = "upstream_error"
    retryable = True


def error_envelope(exc: AppError, correlation_id: str | None = None) -> dict[str, Any]:
    return {
        "code": exc.code,
        "message": exc.message,
        "retryable": exc.retryable,
        "details": exc.details,
        "correlation_id": correlation_id or str(uuid.uuid4()),
    }


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    correlation_id = getattr(request.state, "correlation_id", None)
    return JSONResponse(status_code=exc.status_code, content=error_envelope(exc, correlation_id))
