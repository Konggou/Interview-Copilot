import streamlit as st

from state.session import is_interview_active
from utils.helpers import score_interview_answer


MAX_STORED_SOURCES = 2


def _compact_sources(sources: list[dict]) -> list[dict]:
  compact_sources = []
  for src in sources[:MAX_STORED_SOURCES]:
    compact_sources.append({
      "source": src.get("source", "\u672a\u77e5\u6587\u4ef6"),
      "page": src.get("page", "\u672a\u77e5\u9875\u7801"),
      "score": src.get("score", 0.0),
      "snippet": src.get("snippet", "")[:220].replace("\n", " ").strip(),
    })
  return compact_sources


def append_interview_message(role: str, text: str, sources: list[dict] | None = None):
  transcript = st.session_state.get("interview_transcript", [])
  transcript.append({
    "role": role,
    "text": text,
    "sources": _compact_sources(sources or []),
  })
  st.session_state.interview_transcript = transcript[-80:]


def reset_interview_view_state():
  st.session_state.update(
    interview_session_id="",
    interview_status="draft",
    interview_current_question=None,
    interview_progress={},
    interview_transcript=[],
    interview_turns=[],
    interview_rubric={},
    interview_report={},
    interview_report_markdown="",
  )


def seed_interview_session(result: dict):
  reset_interview_view_state()
  st.session_state.interview_session_id = result.get("session_id", "")
  st.session_state.interview_status = result.get("status", "active")
  st.session_state.interview_rubric = result.get("rubric", {})
  st.session_state.interview_progress = result.get("progress", {})

  opening_message = result.get("opening_message", "")
  if opening_message:
    append_interview_message("assistant", opening_message)

  current_question = result.get("current_question")
  st.session_state.interview_current_question = current_question
  if current_question:
    turn = {
      "question_id": current_question.get("id", ""),
      "question": current_question.get("question", ""),
      "answer": "",
      "feedback": {},
    }
    st.session_state.interview_turns = [turn]
    append_interview_message(
      "assistant",
      f"\u9762\u8bd5\u5b98\uff1a{current_question.get('question', '')}",
    )


def render_sources(sources: list[dict]):
  if not sources:
    return

  with st.expander("\u67e5\u770b\u6765\u6e90\u7247\u6bb5\u4e0e\u76f8\u4f3c\u5ea6"):
    for idx, src in enumerate(sources, start=1):
      filename = src.get("source", "\u672a\u77e5\u6587\u4ef6")
      page = src.get("page", "\u672a\u77e5\u9875\u7801")
      score = src.get("score", 0.0)
      snippet = src.get("snippet", "")[:220].replace("\n", " ").strip()

      st.markdown(f"**{idx}. {filename}\uff08\u7b2c {page} \u9875\uff09**")
      st.caption(f"\u76f8\u4f3c\u5ea6\u5206\u6570\uff1a{score:.4f}")
      if snippet:
        st.write(snippet)


def render_uploaded_files_expander():
  uploaded_files = st.session_state.get(
    f"uploaded_files_{st.session_state.uploader_key}",
    [],
  )
  if uploaded_files and not st.session_state.get("unsubmitted_files"):
    with st.expander("\u5df2\u63d0\u4ea4\u7684\u7b80\u5386\u6587\u4ef6"):
      for file_obj in uploaded_files:
        st.markdown(f"- {file_obj.name}")


def render_interview_history():
  for item in st.session_state.get("interview_transcript", []):
    with st.chat_message(item.get("role", "assistant")):
      st.markdown(item.get("text", ""))
      render_sources(item.get("sources", []))


def _build_feedback_text(result: dict) -> str:
  lines = [f"\u672c\u8f6e\u5f97\u5206\uff1a{result.get('score', 0)}/10"]

  summary = result.get("summary", "")
  if summary:
    lines.append(f"\u603b\u7ed3\uff1a{summary}")

  strengths = result.get("strengths", [])
  if strengths:
    lines.append("\u4f18\u70b9\uff1a")
    lines.extend([f"- {item}" for item in strengths])

  weaknesses = result.get("weaknesses", [])
  if weaknesses:
    lines.append("\u4e0d\u8db3\uff1a")
    lines.extend([f"- {item}" for item in weaknesses])

  suggestions = result.get("suggestions", [])
  if suggestions:
    lines.append("\u5efa\u8bae\uff1a")
    lines.extend([f"- {item}" for item in suggestions])

  followup = result.get("followup_question", "")
  if followup:
    lines.append(f"\u8ffd\u95ee\u5efa\u8bae\uff1a{followup}")

  return "\n".join(lines)


def _append_turn_result(question_id: str, question_text: str, user_answer: str, result: dict):
  turns = st.session_state.get("interview_turns", [])
  for turn in turns:
    if turn.get("question_id") == question_id:
      turn["question"] = question_text
      turn["answer"] = user_answer
      turn["feedback"] = result
      break

  next_question = result.get("next_question")
  if next_question:
    turns.append({
      "question_id": next_question.get("id", ""),
      "question": next_question.get("question", ""),
      "answer": "",
      "feedback": {},
    })
  st.session_state.interview_turns = turns


def _handle_interview_answer(model_provider, model, user_answer: str):
  current_question = st.session_state.get("interview_current_question")
  session_id = st.session_state.get("interview_session_id", "")

  if not session_id or not current_question:
    st.warning("\u8bf7\u5148\u70b9\u51fb\u5de6\u4fa7\u201c\u5f00\u59cb\u9762\u8bd5\u201d\u3002")
    return

  question_id = current_question.get("id", "")
  question_text = current_question.get("question", "")

  with st.chat_message("user"):
    st.markdown(user_answer)
  with st.chat_message("assistant"):
    with st.spinner("\u9762\u8bd5\u5b98\u6b63\u5728\u8bc4\u4f30\u4f60\u7684\u56de\u7b54..."):
      try:
        result = score_interview_answer(
          model_provider,
          model,
          session_id,
          question_id,
          user_answer,
        )
        feedback_text = _build_feedback_text(result)
        sources = result.get("sources", [])
        st.markdown(feedback_text)
        render_sources(sources)

        append_interview_message("user", user_answer)
        append_interview_message("assistant", feedback_text, sources)
        _append_turn_result(question_id, question_text, user_answer, result)

        next_question = result.get("next_question")
        st.session_state.interview_current_question = next_question
        st.session_state.interview_progress = result.get("progress", {})
        st.session_state.interview_status = result.get("status", "active")

        if next_question:
          next_message = f"\u9762\u8bd5\u5b98\uff1a{next_question.get('question', '')}"
          st.markdown(next_message)
          append_interview_message("assistant", next_message)
      except Exception as e:
        st.error(f"\u9519\u8bef\uff1a{str(e)}")


def render_unified_input(model_provider, model):
  user_input = st.chat_input(
    "\u8bf7\u8f93\u5165\u4f60\u7684\u9762\u8bd5\u56de\u7b54",
    disabled=not is_interview_active(),
  )

  if not user_input:
    return

  _handle_interview_answer(model_provider, model, user_input)
