import streamlit as st


def setup_session_state():
  default_state = {
    "resume_ready": False,
    "pdf_files": [],
    "model_provider": "Deepseek",
    "model": "deepseek-reasoner",
    "interview_status": "draft",
    "interview_session_id": "",
    "interview_job_description": "",
    "interview_opening_style": "",
    "interview_current_question": None,
    "interview_progress": {},
    "interview_transcript": [],
    "interview_turns": [],
    "interview_rubric": {},
    "interview_report": {},
    "interview_report_markdown": "",
    "pending_interview_start": False,
    "interview_start_error": "",
    "last_provider": None,
    "unsubmitted_files": False,
    "uploader_key": 0,
  }

  for key, default in default_state.items():
    if key not in st.session_state:
      st.session_state[key] = default


def is_resume_ready():
  return (
    st.session_state.get("resume_ready")
    and st.session_state.get(f"uploaded_files_{st.session_state.uploader_key}", [])
    and not st.session_state.get("unsubmitted_files")
  )


def is_interview_active():
  return st.session_state.get("interview_status") == "active"
