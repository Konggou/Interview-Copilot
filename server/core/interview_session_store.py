import json
import os
from copy import deepcopy
from datetime import datetime, timezone

from config.settings import INTERVIEW_SESSION_STORE_PATH
from utils.logger import logger


def _ensure_store_file():
  store_dir = os.path.dirname(INTERVIEW_SESSION_STORE_PATH)
  if store_dir:
    os.makedirs(store_dir, exist_ok=True)
  if not os.path.exists(INTERVIEW_SESSION_STORE_PATH):
    with open(INTERVIEW_SESSION_STORE_PATH, "w", encoding="utf-8") as store_file:
      json.dump({}, store_file)


def _read_store() -> dict:
  _ensure_store_file()
  with open(INTERVIEW_SESSION_STORE_PATH, "r", encoding="utf-8") as store_file:
    try:
      return json.load(store_file)
    except json.JSONDecodeError:
      logger.warning("Interview session store is corrupted. Resetting it.")
      return {}


def _write_store(store: dict):
  _ensure_store_file()
  with open(INTERVIEW_SESSION_STORE_PATH, "w", encoding="utf-8") as store_file:
    json.dump(store, store_file, ensure_ascii=False, indent=2)


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def save_session(session_id: str, session_data: dict):
  store = _read_store()
  payload = deepcopy(session_data)
  payload["updated_at"] = _now_iso()
  if "created_at" not in payload:
    payload["created_at"] = payload["updated_at"]
  store[session_id] = payload
  _write_store(store)


def get_session(session_id: str) -> dict | None:
  store = _read_store()
  session = store.get(session_id)
  return deepcopy(session) if session else None


def append_turn(session_id: str, turn_payload: dict):
  session = get_session(session_id)
  if not session:
    return None

  turns = session.setdefault("turns", [])
  turn = deepcopy(turn_payload)
  turn.setdefault("created_at", _now_iso())
  turn.setdefault("answered_at", None)
  turns.append(turn)
  save_session(session_id, session)
  return get_session(session_id)


def set_current_question(session_id: str, question_id: str | None):
  session = get_session(session_id)
  if not session:
    return None

  session["current_question_id"] = question_id
  save_session(session_id, session)
  return get_session(session_id)


def mark_session_status(session_id: str, status: str):
  session = get_session(session_id)
  if not session:
    return None

  session["status"] = status
  save_session(session_id, session)
  return get_session(session_id)


def get_current_turn(session_id: str) -> dict | None:
  session = get_session(session_id)
  if not session:
    return None

  current_question_id = session.get("current_question_id")
  if not current_question_id:
    return None

  for turn in session.get("turns", []):
    if turn.get("question_id") == current_question_id:
      return deepcopy(turn)
  return None


def get_completed_turns(session_id: str) -> list[dict]:
  session = get_session(session_id)
  if not session:
    return []

  completed_turns = []
  for turn in session.get("turns", []):
    if turn.get("user_answer"):
      completed_turns.append(deepcopy(turn))
  return completed_turns


def update_session_answer(session_id: str, question_id: str, user_answer: str, feedback: dict):
  session = get_session(session_id)
  if not session:
    return None

  updated = False
  for turn in session.get("turns", []):
    if turn.get("question_id") == question_id:
      turn["user_answer"] = user_answer
      turn["feedback"] = feedback
      turn["answered_at"] = _now_iso()
      updated = True
      break

  if not updated:
    return None

  save_session(session_id, session)
  return get_session(session_id)


def update_session_report(session_id: str, report: dict):
  session = get_session(session_id)
  if not session:
    return None

  session["report"] = report
  save_session(session_id, session)
  return get_session(session_id)
