from copy import deepcopy
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
  sys.path.insert(0, str(SERVER_DIR))

import main  # noqa: E402
from api import routes  # noqa: E402


@pytest.fixture
def client(monkeypatch):
  async def noop_startup():
    return None

  monkeypatch.setattr(main, "initialize_empty_vectorstores", noop_startup)
  monkeypatch.setattr(routes, "ping_redis", lambda: noop_startup())

  store = {}

  async def save_session(session_id, session_data):
    payload = deepcopy(session_data)
    payload.setdefault("created_at", "now")
    payload["updated_at"] = "now"
    store[session_id] = payload

  async def get_session(session_id):
    session = store.get(session_id)
    return deepcopy(session) if session else None

  async def append_turn(session_id, turn_payload):
    session = store.get(session_id)
    if not session:
      return None
    session.setdefault("turns", []).append(deepcopy(turn_payload))
    session["updated_at"] = "now"
    return deepcopy(session)

  async def set_current_question(session_id, question_id):
    session = store.get(session_id)
    if not session:
      return None
    session["current_question_id"] = question_id
    session["updated_at"] = "now"
    return deepcopy(session)

  async def mark_session_status(session_id, status):
    session = store.get(session_id)
    if not session:
      return None
    session["status"] = status
    session["updated_at"] = "now"
    return deepcopy(session)

  async def get_current_turn(session_id):
    session = store.get(session_id)
    if not session:
      return None
    current_question_id = session.get("current_question_id")
    for turn in session.get("turns", []):
      if turn.get("question_id") == current_question_id:
        return deepcopy(turn)
    return None

  async def get_completed_turns(session_id):
    session = store.get(session_id)
    if not session:
      return []
    return [
      deepcopy(turn)
      for turn in session.get("turns", [])
      if turn.get("user_answer")
    ]

  async def update_session_answer(session_id, question_id, user_answer, feedback):
    session = store.get(session_id)
    if not session:
      return None
    for turn in session.get("turns", []):
      if turn.get("question_id") == question_id:
        turn["user_answer"] = user_answer
        turn["feedback"] = feedback
        turn["answered_at"] = "now"
        break
    session["updated_at"] = "now"
    return deepcopy(session)

  async def update_session_report(session_id, report):
    session = store.get(session_id)
    if not session:
      return None
    session["report"] = deepcopy(report)
    session["updated_at"] = "now"
    return deepcopy(session)

  monkeypatch.setattr(routes, "save_session", save_session)
  monkeypatch.setattr(routes, "get_session", get_session)
  monkeypatch.setattr(routes, "append_turn", append_turn)
  monkeypatch.setattr(routes, "set_current_question", set_current_question)
  monkeypatch.setattr(routes, "mark_session_status", mark_session_status)
  monkeypatch.setattr(routes, "get_current_turn", get_current_turn)
  monkeypatch.setattr(routes, "get_completed_turns", get_completed_turns)
  monkeypatch.setattr(routes, "update_session_answer", update_session_answer)
  monkeypatch.setattr(routes, "update_session_report", update_session_report)

  with TestClient(main.app) as test_client:
    yield test_client, store


def test_health_check(client):
  test_client, _store = client
  response = test_client.get("/health")

  assert response.status_code == 200
  assert response.json()["status"] == "success"
  assert response.json()["data"]["app"] == "ok"
  assert "redis" in response.json()["data"]


def test_metrics_endpoint(client):
  test_client, _store = client
  response = test_client.get("/metrics")

  assert response.status_code == 200
  assert "ragbot_llm_requests_total" in response.text


def test_chat_returns_answer_and_sources_when_threshold_passes(client, monkeypatch):
  test_client, _store = client

  async def fake_retrieve(*_args, **_kwargs):
    return {
      "results": [("doc", 0.2)],
      "top_score": 0.2,
      "passes_threshold": True,
    }

  async def fake_invoke(*_args, **_kwargs):
    return type("Result", (), {
      "raw_text": "structured answer",
      "cache_hit": False,
    })()

  monkeypatch.setattr(routes, "retrieve_scored_chunks", fake_retrieve)
  monkeypatch.setattr(
    routes,
    "serialize_search_results",
    lambda *_args, **_kwargs: [{
      "source": "resume.pdf",
      "page": 1,
      "score": 0.2,
      "snippet": "retrieved context",
    }],
  )
  monkeypatch.setattr(routes, "invoke_completion", fake_invoke)

  response = test_client.post(
    "/chat",
    json={
      "model_provider": "deepseek",
      "model_name": "deepseek-chat",
      "message": "Summarize my project",
    },
  )

  assert response.status_code == 200
  assert response.json()["status"] == "success"
  assert response.headers["X-Cache"] == "MISS"
  assert response.json()["data"]["answer"] == "structured answer"


def test_chat_stream_returns_sse_events(client, monkeypatch):
  test_client, _store = client

  async def fake_retrieve(*_args, **_kwargs):
    return {
      "results": [("doc", 0.2)],
      "top_score": 0.2,
      "passes_threshold": True,
    }

  async def fake_cache_lookup(*_args, **_kwargs):
    return {"prompt_text": "prompt", "prompt_tokens": 10, "cached": None}

  async def fake_stream(*_args, **_kwargs):
    yield {"event": "meta", "data": {"phase": "answer", "cache": "MISS"}}
    yield {"event": "delta", "data": {"phase": "answer", "text": "hello"}}
    yield {
      "event": "done",
      "data": {
        "raw_text": "hello world",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "cache": "MISS",
      },
    }

  monkeypatch.setattr(routes, "retrieve_scored_chunks", fake_retrieve)
  monkeypatch.setattr(routes, "inspect_completion_cache", fake_cache_lookup)
  monkeypatch.setattr(routes, "stream_completion", fake_stream)
  monkeypatch.setattr(
    routes,
    "serialize_search_results",
    lambda *_args, **_kwargs: [{
      "source": "resume.pdf",
      "page": 1,
      "score": 0.2,
      "snippet": "retrieved context",
    }],
  )

  response = test_client.post(
    "/chat/stream",
    json={
      "model_provider": "deepseek",
      "model_name": "deepseek-chat",
      "message": "stream it",
    },
  )

  assert response.status_code == 200
  assert response.headers["X-Cache"] == "MISS"
  assert "event: delta" in response.text
  assert "event: done" in response.text


def test_interview_start_returns_first_question(client, monkeypatch):
  test_client, store = client

  async def fake_prepare_context(*_args, **_kwargs):
    return (
      [{"source": "resume.pdf", "page": 1, "score": 0.2, "snippet": "resume evidence"}],
      "resume context",
    )

  async def fake_invoke(*_args, **_kwargs):
    return type("Result", (), {
      "parsed_payload": {
        "opening_message": "Let us begin.",
        "question": {
          "id": "q1",
          "question": "Introduce your main project.",
          "focus": "Project overview",
          "difficulty": "easy",
        },
        "rubric": {"score_scale": "1-10"},
      },
      "cache_hit": False,
    })()

  async def fake_retrieve_sources(*_args, **_kwargs):
    return [{"source": "resume.pdf", "page": 1, "score": 0.2, "snippet": "resume evidence"}]

  monkeypatch.setattr(routes, "_prepare_interview_context", fake_prepare_context)
  monkeypatch.setattr(routes, "invoke_completion", fake_invoke)
  monkeypatch.setattr(routes, "_retrieve_question_specific_sources", fake_retrieve_sources)

  response = test_client.post(
    "/interview/start",
    json={
      "model_provider": "deepseek",
      "model_name": "deepseek-chat",
      "jd_text": "AI application engineer",
    },
  )

  payload = response.json()["data"]
  assert response.status_code == 200
  assert payload["status"] == "active"
  assert payload["current_question"]["id"] == "q1"
  assert payload["opening_message"] == "Let us begin."
  assert len(store[payload["session_id"]]["turns"]) == 1


def test_interview_answer_appends_next_question(client, monkeypatch):
  test_client, store = client
  session_id = "session-1"
  store[session_id] = {
    "session_id": session_id,
    "status": "active",
    "model_provider": "deepseek",
    "model_name": "deepseek-chat",
    "jd_text": "AI application engineer",
    "rubric": {"score_scale": "1-10"},
    "resume_sources": [{
      "source": "resume.pdf",
      "page": 1,
      "score": 0.2,
      "snippet": "resume evidence",
    }],
    "turns": [{
      "turn_id": "t1",
      "question_id": "q1",
      "question": "Introduce your main project.",
      "focus": "Project overview",
      "difficulty": "easy",
      "sources": [{
        "source": "resume.pdf",
        "page": 1,
        "score": 0.2,
        "snippet": "resume evidence",
      }],
      "user_answer": "",
      "feedback": {},
      "created_at": "now",
      "answered_at": None,
    }],
    "current_question_id": "q1",
    "report": None,
    "created_at": "now",
    "updated_at": "now",
  }

  async def fake_retrieve_sources(*_args, **_kwargs):
    return [{
      "source": "resume.pdf",
      "page": 1,
      "score": 0.2,
      "snippet": "resume evidence",
    }]

  results = [
    type("Result", (), {
      "parsed_payload": {
        "score": 8,
        "summary": "Solid answer.",
        "strengths": ["Clear structure"],
        "weaknesses": ["Could quantify impact more"],
        "suggestions": ["Add metrics"],
        "followup_question": "What was the scale?",
      },
      "cache_hit": False,
    })(),
    type("Result", (), {
      "parsed_payload": {
        "question": {
          "id": "q2",
          "question": "How did you handle retrieval quality?",
          "focus": "RAG retrieval",
          "difficulty": "mid",
        },
      },
      "cache_hit": False,
    })(),
  ]

  async def fake_invoke(*_args, **_kwargs):
    return results.pop(0)

  monkeypatch.setattr(routes, "_retrieve_question_specific_sources", fake_retrieve_sources)
  monkeypatch.setattr(routes, "invoke_completion", fake_invoke)

  response = test_client.post(
    "/interview/answer",
    json={
      "model_provider": "deepseek",
      "model_name": "deepseek-chat",
      "session_id": session_id,
      "question_id": "q1",
      "user_answer": "I built the core RAG flow and tuned retrieval quality.",
    },
  )

  payload = response.json()["data"]
  assert response.status_code == 200
  assert payload["status"] == "active"
  assert payload["next_question"]["id"] == "q2"
  assert store[session_id]["current_question_id"] == "q2"
  assert len(store[session_id]["turns"]) == 2


def test_interview_end_marks_session_ended(client):
  test_client, store = client
  store["session-1"] = {
    "session_id": "session-1",
    "status": "active",
    "turns": [],
    "current_question_id": "q1",
  }

  response = test_client.post("/interview/end", json={"session_id": "session-1"})

  assert response.status_code == 200
  assert response.json()["data"]["status"] == "ended"
  assert store["session-1"]["status"] == "ended"
  assert store["session-1"]["current_question_id"] is None


def test_interview_report_requires_ended_session(client):
  test_client, store = client
  store["session-1"] = {
    "session_id": "session-1",
    "status": "active",
    "turns": [],
    "current_question_id": "q1",
  }

  response = test_client.get("/interview/report/session-1")

  assert response.status_code == 409
  assert response.json()["status"] == "error"


def test_interview_report_returns_after_end(client):
  test_client, store = client
  store["session-1"] = {
    "session_id": "session-1",
    "status": "ended",
    "jd_text": "AI application engineer",
    "rubric": {"score_scale": "1-10"},
    "turns": [{
      "turn_id": "t1",
      "question_id": "q1",
      "question": "Introduce your main project.",
      "focus": "Project overview",
      "difficulty": "easy",
      "user_answer": "I built a resume-grounded interview copilot.",
      "feedback": {
        "score": 8,
        "summary": "Solid answer.",
        "strengths": ["Clear structure"],
        "weaknesses": ["Could quantify impact more"],
        "suggestions": ["Add metrics"],
      },
    }],
    "current_question_id": None,
    "report": None,
  }

  response = test_client.get("/interview/report/session-1")

  assert response.status_code == 200
  assert response.json()["status"] == "success"
  assert response.json()["data"]["average_score"] == 8.0
