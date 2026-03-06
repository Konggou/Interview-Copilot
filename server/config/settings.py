import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

TEMPFILE_UPLOAD_DIRECTORY = "./temp/uploaded_files"
INTERVIEW_SESSION_STORE_PATH = os.getenv("INTERVIEW_SESSION_STORE_PATH", "./temp/interview_sessions.json")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.4"))
SOURCE_SNIPPET_LENGTH = int(os.getenv("SOURCE_SNIPPET_LENGTH", "240"))
INTERVIEW_CONTEXT_SNIPPET_LENGTH = int(os.getenv("INTERVIEW_CONTEXT_SNIPPET_LENGTH", "400"))
DEFAULT_INTERVIEW_QUESTION_COUNT = int(os.getenv("DEFAULT_INTERVIEW_QUESTION_COUNT", "3"))

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
  key.lower(): f"./data/{key.lower()}_vector_store"
  for key in MODEL_OPTIONS.keys()
}
