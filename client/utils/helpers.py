from utils.api import (
  answer_interview,
  chat,
  end_interview,
  get_interview_report,
  get_supported_llm,
  get_supported_models,
  get_vectorstore_colllection_count,
  get_vectorstore_similarity_search,
  start_interview,
  upload_and_process_pdfs,
)


def get_model_providers() -> list[str]:
  return get_supported_llm()


def get_models(model_provider) -> list[str]:
  if not model_provider:
    return []
  return get_supported_models(model_provider)


def process_uploaded_pdfs(model_provider, uploaded_files) -> str:
  return upload_and_process_pdfs(model_provider, uploaded_files)


def process_user_input(model_provider, model_name, user_input) -> dict:
  return chat(model_provider, model_name, user_input)


def start_interview_session(
  model_provider,
  model_name,
  jd_text="",
  opening_style="",
) -> dict:
  return start_interview(model_provider, model_name, jd_text, opening_style)


def score_interview_answer(
  model_provider,
  model_name,
  session_id,
  question_id,
  user_answer,
) -> dict:
  return answer_interview(
    model_provider,
    session_id,
    question_id,
    user_answer,
    model_name,
  )


def end_interview_session(session_id) -> dict:
  return end_interview(session_id)


def load_interview_report(session_id, report_format="json"):
  return get_interview_report(session_id, report_format)


def get_documents_count(model_provider) -> int:
  return get_vectorstore_colllection_count(model_provider)


def get_similar_chunks(model_provider, query) -> list[dict]:
  return get_vectorstore_similarity_search(model_provider, query)
