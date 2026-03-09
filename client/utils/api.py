from io import BytesIO
import json

import httpx

from utils.config import API_URL


DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
STREAM_TIMEOUT = httpx.Timeout(None, connect=10.0)


def _handle_response(response: httpx.Response):
  try:
    payload = response.json()
  except ValueError as exc:
    response.raise_for_status()
    raise Exception("后端返回了无法解析的响应。") from exc

  if response.is_error:
    raise Exception(payload.get("message", f"接口调用失败：HTTP {response.status_code}"))

  if payload["status"] == "success":
    return payload.get("data")
  raise Exception(payload.get("message", "接口调用失败。"))


def _build_client(timeout=DEFAULT_TIMEOUT):
  # Local app-to-app calls should not inherit workstation proxy settings.
  return httpx.Client(base_url=API_URL, timeout=timeout, trust_env=False)


def _iter_sse_events(response: httpx.Response):
  event_name = "message"
  data_lines = []

  for line in response.iter_lines():
    if line is None:
      continue
    if line == "":
      if data_lines:
        yield {
          "event": event_name,
          "data": json.loads("".join(data_lines)),
        }
      event_name = "message"
      data_lines = []
      continue
    if line.startswith("event: "):
      event_name = line[7:]
    elif line.startswith("data: "):
      data_lines.append(line[6:])

  if data_lines:
    yield {
      "event": event_name,
      "data": json.loads("".join(data_lines)),
    }


def get_supported_llm() -> list[str]:
  with _build_client() as client:
    response = client.get("/llm")
    return _handle_response(response)


def get_supported_models(model_provider) -> list[str]:
  with _build_client() as client:
    response = client.get(f"/llm/{model_provider}")
    return _handle_response(response)


def get_vectorstore_colllection_count(model_provider) -> int:
  with _build_client() as client:
    response = client.get(f"/vector_store/count/{model_provider}")
    return _handle_response(response)


def get_vectorstore_similarity_search(model_provider, query) -> list[dict]:
  with _build_client() as client:
    response = client.post(
      "/vector_store/search",
      json={
        "model_provider": model_provider,
        "query": query,
      },
    )
    return _handle_response(response)


def upload_and_process_pdfs(model_provider, uploaded_files) -> str:
  files = []
  for file in uploaded_files:
    if hasattr(file, "data"):
      files.append(("files", (file.name, BytesIO(file.data), file.type)))
    else:
      files.append(("files", (file.name, file.read(), file.type)))

  with _build_client(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
    response = client.post(
      "/upload_and_process_pdfs",
      files=files,
      data={"model_provider": model_provider},
    )
    return _handle_response(response)


def chat(model_provider, model_name, user_input) -> dict:
  with _build_client(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
    response = client.post(
      "/chat",
      json={
        "model_provider": model_provider,
        "model_name": model_name,
        "message": user_input,
      },
    )
    return {
      "data": _handle_response(response),
      "headers": dict(response.headers),
    }


def stream_chat(model_provider, model_name, user_input):
  payload = {
    "model_provider": model_provider,
    "model_name": model_name,
    "message": user_input,
  }
  with _build_client(timeout=STREAM_TIMEOUT) as client:
    with client.stream("POST", "/chat/stream", json=payload) as response:
      response.raise_for_status()
      for event in _iter_sse_events(response):
        yield event


def start_interview(model_provider, model_name=None, jd_text="", opening_style="") -> dict:
  payload = {"model_provider": model_provider}
  if model_name:
    payload["model_name"] = model_name
  if jd_text:
    payload["jd_text"] = jd_text
  if opening_style:
    payload["opening_style"] = opening_style

  with _build_client(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
    response = client.post("/interview/start", json=payload)
    return {
      "data": _handle_response(response),
      "headers": dict(response.headers),
    }


def stream_start_interview(model_provider, model_name=None, jd_text="", opening_style=""):
  payload = {"model_provider": model_provider}
  if model_name:
    payload["model_name"] = model_name
  if jd_text:
    payload["jd_text"] = jd_text
  if opening_style:
    payload["opening_style"] = opening_style

  with _build_client(timeout=STREAM_TIMEOUT) as client:
    with client.stream("POST", "/interview/start/stream", json=payload) as response:
      response.raise_for_status()
      for event in _iter_sse_events(response):
        yield event


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

  with _build_client(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
    response = client.post("/interview/answer", json=payload)
    return {
      "data": _handle_response(response),
      "headers": dict(response.headers),
    }


def stream_answer_interview(
  model_provider,
  session_id,
  question_id,
  user_answer,
  model_name=None,
):
  payload = {
    "model_provider": model_provider,
    "session_id": session_id,
    "question_id": question_id,
    "user_answer": user_answer,
  }
  if model_name:
    payload["model_name"] = model_name

  with _build_client(timeout=STREAM_TIMEOUT) as client:
    with client.stream("POST", "/interview/answer/stream", json=payload) as response:
      response.raise_for_status()
      for event in _iter_sse_events(response):
        yield event


def end_interview(session_id) -> dict:
  with _build_client() as client:
    response = client.post("/interview/end", json={"session_id": session_id})
    return _handle_response(response)


def get_interview_report(session_id, report_format="json"):
  with _build_client() as client:
    response = client.get(
      f"/interview/report/{session_id}",
      params={"report_format": report_format},
    )
    payload = _handle_response(response)
    if report_format == "markdown" and isinstance(payload, dict):
      return payload.get("markdown", "")
    return payload
