import json

import streamlit as st

from components.chat import seed_interview_session
from state.session import is_interview_active, is_resume_ready
from utils.helpers import (
  end_interview_session,
  load_interview_report,
  start_interview_session_stream,
)


def _render_bullet_section(title: str, items: list[str]):
  if not items:
    return

  st.markdown(f"**{title}**")
  for item in items:
    st.markdown(f"- {item}")


def request_interview_start():
  st.session_state.pending_interview_start = True
  st.session_state.interview_start_error = ""


def render_pending_interview_start(model_provider, model_name):
  if not st.session_state.get("pending_interview_start"):
    return

  streamed_text = ""
  with st.chat_message("assistant"):
    stream_placeholder = st.empty()

    try:
      for event in start_interview_session_stream(
        model_provider,
        model_name,
        st.session_state.get("interview_job_description", ""),
        st.session_state.get("interview_opening_style", ""),
      ):
        if event["event"] == "delta":
          streamed_text += event["data"].get("text", "")
          stream_placeholder.markdown(streamed_text)
        elif event["event"] == "done":
          st.session_state.pending_interview_start = False
          st.session_state.interview_start_error = ""
          stream_placeholder.empty()
          result = event["data"]["payload"]
          seed_interview_session(result)
          st.toast("\u6a21\u62df\u9762\u8bd5\u5df2\u5f00\u59cb\u3002")
          st.rerun()
        elif event["event"] == "error":
          raise Exception(event["data"].get("message", "\u542f\u52a8\u9762\u8bd5\u5931\u8d25"))
    except Exception as e:
      st.session_state.pending_interview_start = False
      st.session_state.interview_start_error = str(e)
      stream_placeholder.empty()
      st.error(f"\u9519\u8bef\uff1a{str(e)}")


def _end_interview():
  session_id = st.session_state.get("interview_session_id", "")
  if not session_id:
    return

  with st.spinner("\u6b63\u5728\u7ed3\u675f\u672c\u8f6e\u9762\u8bd5..."):
    try:
      result = end_interview_session(session_id)
      st.session_state.interview_status = result.get("status", "ended")
      st.session_state.interview_current_question = None
      st.session_state.interview_progress = {
        "current_round": len(st.session_state.get("interview_turns", [])),
        "status": "ended",
      }
      st.toast("\u672c\u8f6e\u9762\u8bd5\u5df2\u7ed3\u675f\u3002")
      st.rerun()
    except Exception as e:
      st.error(f"\u9519\u8bef\uff1a{str(e)}")


def _render_report(report: dict):
  st.markdown("### \u9762\u8bd5\u62a5\u544a")

  col1, col2, col3 = st.columns(3)
  col1.metric("\u5e73\u5747\u5206", f"{report.get('average_score', 0)}/10")
  col2.metric("\u5df2\u5b8c\u6210\u8f6e\u6b21", str(report.get("answered_count", 0)))
  col3.metric("\u603b\u8f6e\u6b21", str(report.get("question_count", 0)))

  summary = report.get("summary", "")
  if summary:
    st.markdown(f"**\u6574\u4f53\u7ed3\u8bba**\n\n{summary}")

  if report.get("job_description"):
    with st.expander("\u5c97\u4f4d JD \u6458\u8981"):
      st.markdown(report["job_description"])

  _render_bullet_section("\u4e3b\u8981\u4f18\u70b9", report.get("strengths", []))
  _render_bullet_section("\u4e3b\u8981\u4e0d\u8db3", report.get("weaknesses", []))
  _render_bullet_section("\u6539\u8fdb\u5efa\u8bae", report.get("suggestions", []))


def render_interview_sidebar_controls(model_provider, model_name):
  st.text_area(
    "\u5c97\u4f4d JD",
    key="interview_job_description",
    height=140,
    placeholder="\u7c98\u8d34\u76ee\u6807\u5c97\u4f4d JD\uff0c\u7cfb\u7edf\u4f1a\u4ee5\u6280\u672f\u9762\u8bd5\u5b98\u8eab\u4efd\u56f4\u7ed5\u8be5\u5c97\u4f4d\u8fde\u7eed\u63d0\u95ee\u3002",
  )
  st.text_input(
    "\u5f00\u573a\u98ce\u683c\uff08\u53ef\u9009\uff09",
    key="interview_opening_style",
    placeholder="\u4f8b\u5982\uff1a\u7b80\u6d01\u76f4\u63a5 / \u66f4\u50cf\u5927\u5382\u6280\u672f\u4e00\u9762 / \u504f\u538b\u529b\u9762",
  )

  start_disabled = (
    not is_resume_ready()
    or is_interview_active()
    or st.session_state.get("pending_interview_start", False)
  )
  end_disabled = not is_interview_active()
  report_disabled = (
    not st.session_state.get("interview_session_id")
    or st.session_state.get("interview_status") != "ended"
  )

  if st.button(
    "\u5f00\u59cb\u9762\u8bd5",
    use_container_width=True,
    key="start_interview_button",
    disabled=start_disabled,
  ):
    request_interview_start()

  if st.button(
    "\u7ed3\u675f\u9762\u8bd5",
    use_container_width=True,
    key="end_interview_button",
    disabled=end_disabled,
  ):
    _end_interview()

  session_id = st.session_state.get("interview_session_id", "")
  if session_id and st.button(
    "\u751f\u6210\u9762\u8bd5\u62a5\u544a",
    use_container_width=True,
    key="generate_interview_report",
    disabled=report_disabled,
  ):
    with st.spinner("\u6b63\u5728\u751f\u6210\u9762\u8bd5\u62a5\u544a..."):
      try:
        report = load_interview_report(session_id, "json")
        report_markdown = load_interview_report(session_id, "markdown")
        st.session_state.interview_report = report
        st.session_state.interview_report_markdown = report_markdown
      except Exception as e:
        st.error(f"\u9519\u8bef\uff1a{str(e)}")

  if session_id:
    st.caption(f"\u4f1a\u8bdd ID\uff1a{session_id}")
  st.caption(f"\u5f53\u524d\u72b6\u6001\uff1a{st.session_state.get('interview_status', 'draft')}")


def render_interview_report():
  report = st.session_state.get("interview_report", {})
  report_markdown = st.session_state.get("interview_report_markdown", "")
  session_id = st.session_state.get("interview_session_id", "")

  if not report:
    return

  _render_report(report)

  col1, col2 = st.columns(2)
  with col1:
    st.download_button(
      "\u4e0b\u8f7d\u62a5\u544a\uff08JSON\uff09",
      data=json.dumps(report, ensure_ascii=False, indent=2),
      file_name=f"interview-report-{session_id or 'session'}.json",
      mime="application/json",
      use_container_width=True,
    )

  if report_markdown:
    with col2:
      st.download_button(
        "\u4e0b\u8f7d\u62a5\u544a\uff08Markdown\uff09",
        data=report_markdown,
        file_name=f"interview-report-{session_id or 'session'}.md",
        mime="text/markdown",
        use_container_width=True,
      )
