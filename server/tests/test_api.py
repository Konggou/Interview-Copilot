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


class DummyChain:
  def __init__(self, answer: str):
    self.answer = answer

  def invoke(self, _payload):
    return {"answer": self.answer}


@pytest.fixture
def client(monkeypatch):
  monkeypatch.setattr(main, "initialize_empty_vectorstores", lambda: None)

  store = {}

  def save_session(session_id, session_data):
    payload = deepcopy(session_data)
    payload.setdefault("created_at", "now")
    payload["updated_at"] = "now"
    store[session_id] = payload

  def get_session(session_id):
    session = store.get(session_id)
    return deepcopy(session) if session else None

  def append_turn(session_id, turn_payload):
    session = store.get(session_id)
    if not session:
      return None
    session.setdefault("turns", []).append(deepcopy(turn_payload))
    session["updated_at"] = "now"
    return deepcopy(session)

  def set_current_question(session_id, question_id):
    session = store.get(session_id)
    if not session:
      return None
    session["current_question_id"] = question_id
    session["updated_at"] = "now"
    return deepcopy(session)

  def mark_session_status(session_id, status):
    session = store.get(session_id)
    if not session:
      return None
    session["status"] = status
    session["updated_at"] = "now"
    return deepcopy(session)

  def get_current_turn(session_id):
    session = store.get(session_id)
    if not session:
      return None
    current_question_id = session.get("current_question_id")
    for turn in session.get("turns", []):
      if turn.get("question_id") == current_question_id:
        return deepcopy(turn)
    return None

  def get_completed_turns(session_id):
    session = store.get(session_id)
    if not session:
      return []
    return [
      deepcopy(turn)
      for turn in session.get("turns", [])
      if turn.get("user_answer")
    ]

  def update_session_answer(session_id, question_id, user_answer, feedback):
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

  def update_session_report(session_id, report):
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
  assert response.json()["data"] == "ok"


def test_chat_returns_answer_and_sources_when_threshold_passes(client, monkeypatch):
  test_client, _store = client
  monkeypatch.setattr(routes, "load_vectorstore", lambda *_args, **_kwargs: object())
  monkeypatch.setattr(
    routes,
    "retrieve_scored_chunks",
    lambda *_args, **_kwargs: {
      "results": [("doc", 1.6)],
      "top_score": 1.6,
      "passes_threshold": True,
    },
  )
  monkeypatch.setattr(
    routes,
    "serialize_search_results",
    lambda *_args, **_kwargs: [{
      "source": "resume.pdf",
      "page": 1,
      "score": 1.6,
      "snippet": "retrieved context",
    }],
  )
  monkeypatch.setattr(
    routes,
    "build_llm_chain",
    lambda *_args, **_kwargs: DummyChain("structured answer"),
  )

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
  assert response.json()["data"]["answer"] == "structured answer"


def test_interview_start_returns_only_first_question(client, monkeypatch):
  test_client, store = client
  monkeypatch.setattr(
    routes,
    "retrieve_scored_chunks",
    lambda *_args, **_kwargs: {
      "results": [("doc", 1.8)],
      "top_score": 1.8,
      "passes_threshold": True,
    },
  )
  monkeypatch.setattr(
    routes,
    "serialize_search_results",
    lambda *_args, **_kwargs: [{
      "source": "resume.pdf",
      "page": 1,
      "score": 1.8,
      "snippet": "resume evidence",
    }],
  )
  monkeypatch.setattr(
    routes,
    "generate_initial_interview_turn",
    lambda *_args, **_kwargs: {
      "opening_message": "Let us begin.",
      "question": {
        "id": "q1",
        "question": "Introduce your main project.",
        "focus": "Project overview",
        "difficulty": "easy",
      },
      "rubric": {"score_scale": "1-10"},
    },
  )

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


def test_interview_answer_appends_next_question_without_ending(client, monkeypatch):
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
      "score": 1.9,
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
        "score": 1.9,
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

  monkeypatch.setattr(
    routes,
    "evaluate_interview_answer",
    lambda *_args, **_kwargs: {
      "score": 8,
      "summary": "Solid answer.",
      "strengths": ["Clear structure"],
      "weaknesses": ["Could quantify impact more"],
      "suggestions": ["Add metrics"],
      "followup_question": "What was the scale?",
    },
  )
  monkeypatch.setattr(
    routes,
    "generate_next_interview_question",
    lambda *_args, **_kwargs: {
      "question": {
        "id": "q2",
        "question": "How did you handle retrieval quality?",
        "focus": "RAG retrieval",
        "difficulty": "mid",
      },
    },
  )

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
  assert payload["is_finished"] is False
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

  response = test_client.post(
    "/interview/end",
    json={"session_id": "session-1"},
  )

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

  assert response.status_code == 200
  assert response.json()["status"] == "error"
  assert "结束" in response.json()["message"]


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
