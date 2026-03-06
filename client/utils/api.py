from io import BytesIO

import requests

from utils.config import API_URL


def handle_response(response):
  try:
    json_data = response.json()
    if json_data["status"] == "success":
      return json_data.get("data")
    raise Exception(json_data.get("message", "发生未知错误。"))
  except Exception as e:
    raise Exception(f"接口错误：{str(e)}")


def get_supported_llm() -> list[str]:
  response = requests.get(f"{API_URL}/llm", timeout=30)
  return handle_response(response)


def get_supported_models(model_provider) -> list[str]:
  response = requests.get(f"{API_URL}/llm/{model_provider}", timeout=30)
  return handle_response(response)


def get_vectorstore_colllection_count(model_provider) -> int:
  response = requests.get(f"{API_URL}/vector_store/count/{model_provider}", timeout=30)
  return handle_response(response)


def get_vectorstore_similarity_search(model_provider, query) -> list[dict]:
  payload = {
    "model_provider": model_provider,
    "query": query,
  }
  response = requests.post(
    f"{API_URL}/vector_store/search",
    json=payload,
    timeout=60,
  )
  return handle_response(response)


def upload_and_process_pdfs(model_provider, uploaded_files) -> str:
  files = []
  for file in uploaded_files:
    if hasattr(file, "data"):
      files.append(("files", (file.name, BytesIO(file.data), file.type)))
    else:
      files.append(("files", (file.name, file.read(), file.type)))

  response = requests.post(
    f"{API_URL}/upload_and_process_pdfs",
    files=files,
    data={"model_provider": model_provider},
    timeout=180,
  )
  return handle_response(response)


def chat(model_provider, model_name, user_input) -> dict:
  payload = {
    "model_provider": model_provider,
    "model_name": model_name,
    "message": user_input,
  }
  response = requests.post(
    f"{API_URL}/chat",
    json=payload,
    timeout=180,
  )
  return handle_response(response)


def start_interview(model_provider, model_name=None, jd_text="", opening_style="") -> dict:
  payload = {
    "model_provider": model_provider,
  }
  if model_name:
    payload["model_name"] = model_name
  if jd_text:
    payload["jd_text"] = jd_text
  if opening_style:
    payload["opening_style"] = opening_style

  response = requests.post(
    f"{API_URL}/interview/start",
    json=payload,
    timeout=180,
  )
  return handle_response(response)


def answer_interview(
  model_provider,
  session_id,
  question_id,
  user_answer,
  model_name=None,
) -> dict:
  payload = {
    "model_provider": model_provider,
    "session_id": session_id,
    "question_id": question_id,
    "user_answer": user_answer,
  }
  if model_name:
    payload["model_name"] = model_name

  response = requests.post(
    f"{API_URL}/interview/answer",
    json=payload,
    timeout=180,
  )
  return handle_response(response)


def end_interview(session_id) -> dict:
  response = requests.post(
    f"{API_URL}/interview/end",
    json={"session_id": session_id},
    timeout=60,
  )
  return handle_response(response)


def get_interview_report(session_id, report_format="json"):
  response = requests.get(
    f"{API_URL}/interview/report/{session_id}",
    params={"report_format": report_format},
    timeout=60,
  )
  return handle_response(response)
