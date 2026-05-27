"""FastAPI entrypoint, request logging, and global error handling."""

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from gradio.routes import mount_gradio_app

from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.api.routes_indexing import router as indexing_router
from app.api.routes_retrieval import router as retrieval_router
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.core.tracing import configure_langsmith_environment
from app.ui.gradio_app import build_gradio_app

configure_logging()
configure_langsmith_environment()
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")


@app.on_event("startup")
# Log trang thai startup va canh bao neu thieu config quan trong.
async def startup_event() -> None:
    """Log startup state and warn when optional runtime config is missing."""
    logger.info("Starting Vietnamese Tax Legal RAG API")
    if not settings.groq_api_key:
        logger.error("Missing GROQ_API_KEY; /chat will return a configuration error")


@app.middleware("http")
# Ghi log request id, method, path, status va latency cho moi HTTP request.
async def request_logging_middleware(request: Request, call_next):
    """Attach request id and log latency/status for every HTTP request."""
    request_id = request.headers.get("x-request-id", str(uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request error request_id=%s path=%s", request_id, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "HTTP request request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(AppError)
# Chuyen AppError thanh JSON response on dinh cho client.
async def app_error_handler(_: Request, exc: AppError):
    """Convert application-level errors into stable JSON API responses."""
    logger.error("Application error code=%s message=%s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "message": exc.message})


app.include_router(health_router)
app.include_router(indexing_router)
app.include_router(retrieval_router)
app.include_router(chat_router)
app = mount_gradio_app(app, build_gradio_app(), path="/ui")
