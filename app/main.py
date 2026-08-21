"""FastAPI application entrypoint."""

import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from app.config import get_model_config, get_settings
from app.db import Base, engine
from app.errors import AppError, app_error_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Alembic owns production schema; create_all keeps dev/test friction-free.
    if settings.app_env in {"dev", "test"}:
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Video AI", version="0.1.0", lifespan=lifespan)
app.add_exception_handler(AppError, app_error_handler)


@app.middleware("http")
async def correlation_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request.state.correlation_id = request.headers.get("X-Correlation-Id", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Correlation-Id"] = request.state.correlation_id
    return response


@app.get("/healthz")
async def healthz() -> dict:
    cfg = get_model_config()
    return {
        "status": "ok",
        "model_config_version": cfg.config_version,
        "master_duration_seconds": cfg.master_duration_seconds,
    }


def _include_routers() -> None:
    # Imported late so optional modules (built incrementally per the plan) don't
    # break app startup while earlier weeks are being implemented.
    from importlib import import_module

    for module_path, attr in [
        ("app.api.routes", "router"),
        ("app.api.oauth", "router"),
        ("app.api.webhooks", "router"),
        ("app.web.routes", "router"),
    ]:
        try:
            module = import_module(module_path)
            app.include_router(getattr(module, attr))
        except ModuleNotFoundError:
            continue


_include_routers()
