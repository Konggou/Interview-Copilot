import uvicorn

from fastapi import FastAPI

from api.routes import router
from core.vector_database import initialize_empty_vectorstores
from utils.logger import logger


app = FastAPI(
  title="AI Interview Copilot",
  description="Resume-grounded interview assistant service.",
)
app.include_router(router)


@app.on_event("startup")
async def startup_event():
  logger.info("Starting application.")
  initialize_empty_vectorstores()
  logger.info("Application startup complete.")


if __name__ == "__main__":
  logger.info("Running application.")
  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
