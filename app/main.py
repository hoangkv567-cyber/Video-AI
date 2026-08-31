"""FastAPI application entrypoint."""

import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app import readiness
from app.config import get_model_config, get_settings
from app.db import Base, engine
from app.errors import AppError, app_error_handler

logger = logging.getLogger("videoai.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Alembic owns production schema; create_all keeps dev/test friction-free.
    if settings.app_env in {"dev", "test"}:
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Video AI", version="0.1.0", lifespan=lifespan)
app.exception_handler(AppError)(app_error_handler)


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


@app.get("/readyz", include_in_schema=False)
def readyz() -> JSONResponse:
    """Report whether the internal infrastructure needed for work is reachable."""
    results: dict[str, dict[str, str]] = {}
    for name, probe in readiness.get_readiness_checks().items():
        try:
            probe()
        except Exception as exc:
            # Do not return exception messages: they can contain endpoints or credentials.
            logger.warning("readiness check failed: %s (%s)", name, type(exc).__name__)
            results[name] = {"status": "failed"}
        else:
            results[name] = {"status": "ok"}

    is_ready = bool(results) and all(item["status"] == "ok" for item in results.values())
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "checks": results,
        },
        headers={"Cache-Control": "no-store"},
    )


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
