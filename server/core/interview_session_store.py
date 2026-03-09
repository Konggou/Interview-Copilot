import asyncio
import json
import os
from copy import deepcopy
from datetime import datetime, timezone

import aiofiles

from config.settings import INTERVIEW_SESSION_STORE_PATH
from utils.logger import logger


_STORE_LOCK = asyncio.Lock()


async def _ensure_store_file():
  store_dir = os.path.dirname(INTERVIEW_SESSION_STORE_PATH)
  if store_dir:
    os.makedirs(store_dir, exist_ok=True)
  if not os.path.exists(INTERVIEW_SESSION_STORE_PATH):
    async with aiofiles.open(INTERVIEW_SESSION_STORE_PATH, "w", encoding="utf-8") as store_file:
      await store_file.write("{}")


async def _read_store() -> dict:
  await _ensure_store_file()
  async with aiofiles.open(INTERVIEW_SESSION_STORE_PATH, "r", encoding="utf-8") as store_file:
    try:
      content = await store_file.read()
      return json.loads(content or "{}")
    except json.JSONDecodeError:
      logger.warning("Interview session store is corrupted. Resetting it.")
      return {}


async def _write_store(store: dict):
  await _ensure_store_file()
  async with aiofiles.open(INTERVIEW_SESSION_STORE_PATH, "w", encoding="utf-8") as store_file:
    await store_file.write(json.dumps(store, ensure_ascii=False, indent=2))


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


async def save_session(session_id: str, session_data: dict):
  async with _STORE_LOCK:
    store = await _read_store()
    payload = deepcopy(session_data)
    payload["updated_at"] = _now_iso()
    if "created_at" not in payload:
      payload["created_at"] = payload["updated_at"]
    store[session_id] = payload
    await _write_store(store)


async def get_session(session_id: str) -> dict | None:
  async with _STORE_LOCK:
    store = await _read_store()
    session = store.get(session_id)
    return deepcopy(session) if session else None


async def append_turn(session_id: str, turn_payload: dict):
  async with _STORE_LOCK:
    store = await _read_store()
    session = store.get(session_id)
    if not session:
      return None

    turns = session.setdefault("turns", [])
    turn = deepcopy(turn_payload)
    turn.setdefault("created_at", _now_iso())
    turn.setdefault("answered_at", None)
    turns.append(turn)
    session["updated_at"] = _now_iso()
    store[session_id] = session
    await _write_store(store)
    return deepcopy(session)


async def set_current_question(session_id: str, question_id: str | None):
  async with _STORE_LOCK:
    store = await _read_store()
    session = store.get(session_id)
    if not session:
      return None

    session["current_question_id"] = question_id
    session["updated_at"] = _now_iso()
    store[session_id] = session
    await _write_store(store)
    return deepcopy(session)


async def mark_session_status(session_id: str, status: str):
  async with _STORE_LOCK:
    store = await _read_store()
    session = store.get(session_id)
    if not session:
      return None

    session["status"] = status
    session["updated_at"] = _now_iso()
    store[session_id] = session
    await _write_store(store)
    return deepcopy(session)


async def get_current_turn(session_id: str) -> dict | None:
  session = await get_session(session_id)
  if not session:
    return None

  current_question_id = session.get("current_question_id")
  if not current_question_id:
    return None

  for turn in session.get("turns", []):
    if turn.get("question_id") == current_question_id:
      return deepcopy(turn)
  return None


async def get_completed_turns(session_id: str) -> list[dict]:
  session = await get_session(session_id)
  if not session:
    return []

  completed_turns = []
  for turn in session.get("turns", []):
    if turn.get("user_answer"):
      completed_turns.append(deepcopy(turn))
  return completed_turns


async def update_session_answer(
  session_id: str,
  question_id: str,
  user_answer: str,
  feedback: dict,
):
  async with _STORE_LOCK:
    store = await _read_store()
    session = store.get(session_id)
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

    session["updated_at"] = _now_iso()
    store[session_id] = session
    await _write_store(store)
    return deepcopy(session)


async def update_session_report(session_id: str, report: dict):
  async with _STORE_LOCK:
    store = await _read_store()
    session = store.get(session_id)
    if not session:
      return None

    session["report"] = report
    session["updated_at"] = _now_iso()
    store[session_id] = session
    await _write_store(store)
    return deepcopy(session)
