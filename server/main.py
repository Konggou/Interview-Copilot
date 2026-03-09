import time
import uuid
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request

from api.routes import router
from core.llm_service import close_http_client
from core.semantic_cache import close_redis_client
from core.vector_database import initialize_empty_vectorstores
from utils.logger import logger


@asynccontextmanager
async def lifespan(_app: FastAPI):
  logger.info("Starting application")
  await initialize_empty_vectorstores()
  logger.info("Application startup complete")
  try:
    yield
  finally:
    await close_http_client()
    await close_redis_client()
    logger.info("Application shutdown complete")


app = FastAPI(
  title="AI Interview Copilot",
  description="Resume-grounded interview assistant service.",
  lifespan=lifespan,
)
app.include_router(router)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
  request_id = str(uuid.uuid4())
  structlog.contextvars.clear_contextvars()
  structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)

  started_at = time.perf_counter()
  try:
    response = await call_next(request)
  except Exception:
    latency_seconds = round(time.perf_counter() - started_at, 4)
    logger.exception(
      "HTTP request failed",
      method=request.method,
      path=request.url.path,
      latency_seconds=latency_seconds,
    )
    raise

  latency_seconds = round(time.perf_counter() - started_at, 4)
  response.headers["X-Request-ID"] = request_id
  logger.info(
    "HTTP request completed",
    method=request.method,
    path=request.url.path,
    status_code=response.status_code,
    latency_seconds=latency_seconds,
  )
  return response


if __name__ == "__main__":
  logger.info("Running application")
  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
