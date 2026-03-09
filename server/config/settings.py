import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "temp"


def _load_environment() -> None:
  # Load the repo-level .env deterministically so runtime behavior does not
  # depend on which directory uvicorn was started from.
  root_env = ROOT_DIR / ".env"
  server_env = BASE_DIR / ".env"

  if root_env.exists():
    load_dotenv(root_env)

  if server_env.exists():
    load_dotenv(server_env)


_load_environment()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT_SECONDS = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "45"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = float(
  os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.95"),
)
MAX_CONCURRENT_INTERVIEWS = int(os.getenv("MAX_CONCURRENT_INTERVIEWS", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

TEMPFILE_UPLOAD_DIRECTORY = os.getenv(
  "TEMPFILE_UPLOAD_DIRECTORY",
  str(TEMP_DIR / "uploaded_files"),
)
INTERVIEW_SESSION_STORE_PATH = os.getenv(
  "INTERVIEW_SESSION_STORE_PATH",
  str(TEMP_DIR / "interview_sessions.json"),
)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.4"))
SOURCE_SNIPPET_LENGTH = int(os.getenv("SOURCE_SNIPPET_LENGTH", "240"))
INTERVIEW_CONTEXT_SNIPPET_LENGTH = int(
  os.getenv("INTERVIEW_CONTEXT_SNIPPET_LENGTH", "400"),
)
DEFAULT_INTERVIEW_QUESTION_COUNT = int(
  os.getenv("DEFAULT_INTERVIEW_QUESTION_COUNT", "3"),
)
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
CLIENT_API_URL = os.getenv("CLIENT_API_URL", "http://127.0.0.1:8000")

MODEL_OPTIONS = {
  "deepseek": {
    "models": list(dict.fromkeys([
      DEEPSEEK_MODEL,
      "deepseek-chat",
      "deepseek-reasoner",
    ])),
  },
}

VECTORSTORE_DIRECTORY = {
  key.lower(): str(DATA_DIR / f"{key.lower()}_vector_store")
  for key in MODEL_OPTIONS.keys()
}
