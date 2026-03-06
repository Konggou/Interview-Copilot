import streamlit as st

from components.chat import (
  render_interview_history,
  render_unified_input,
  render_uploaded_files_expander,
)
from components.interview import render_interview_report
from components.sidebar import (
  render_interview_setup,
  render_model_selector,
  sidebar_file_upload,
  sidebar_provider_change_check,
  sidebar_utilities,
)
from state.session import (
  is_interview_active,
  is_resume_ready,
  setup_session_state,
)


BASE_THEME_CSS = """
<style>
:root {
  --bg-main: #f4f1e8;
  --bg-card: #fffaf0;
  --border: #d2c5a7;
  --text-main: #1f2937;
  --text-muted: #6b7280;
  --accent: #20543f;
}

.stApp {
  background:
    radial-gradient(circle at top right, rgba(184, 134, 11, 0.12), transparent 28%),
    radial-gradient(circle at top left, rgba(32, 84, 63, 0.08), transparent 22%),
    linear-gradient(180deg, #f7f3ea 0%, #f2ecdf 100%);
}

.hero-card,
.content-card,
.status-card {
  background: rgba(255, 250, 240, 0.92);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  box-shadow: 0 10px 30px rgba(31, 41, 55, 0.06);
}

.hero-title {
  margin: 0;
  font-size: 2rem;
  font-weight: 800;
  color: var(--text-main);
}

.hero-subtitle {
  margin-top: 0.45rem;
  color: var(--text-muted);
  line-height: 1.6;
}

.section-title {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent);
}

.section-copy {
  margin-top: 0.35rem;
  color: var(--text-muted);
  line-height: 1.5;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
}

.status-label {
  font-size: 0.78rem;
  color: var(--text-muted);
}

.status-value {
  margin-top: 0.2rem;
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-main);
}

[data-testid="stChatMessage"] {
  border-radius: 16px;
  border: 1px solid rgba(210, 197, 167, 0.9);
  background: rgba(255, 250, 240, 0.78);
  padding: 0.2rem 0.55rem;
}

[data-testid="stChatInput"] {
  background: rgba(255, 250, 240, 0.95);
  border-top: 1px solid rgba(210, 197, 167, 0.9);
}

.stButton > button {
  border-radius: 12px;
  border: 1px solid rgba(32, 84, 63, 0.18);
  background: linear-gradient(180deg, #20543f 0%, #173d2e 100%);
  color: #ffffff;
  font-weight: 700;
}

.stButton > button:hover {
  border-color: rgba(32, 84, 63, 0.32);
  background: linear-gradient(180deg, #26674d 0%, #1b4d39 100%);
  color: #ffffff;
}

.stDownloadButton > button {
  border-radius: 12px;
  border: 1px solid rgba(184, 134, 11, 0.28);
  background: linear-gradient(180deg, #f4e1a6 0%, #e7cd7d 100%);
  color: #4a3b14;
  font-weight: 700;
}

#MainMenu {
  visibility: hidden;
}
</style>
"""


SIDEBAR_THEME_CSS = """
<style>
[data-testid="stSidebar"] {
  background:
    radial-gradient(circle at top, rgba(244, 225, 166, 0.12), transparent 26%),
    linear-gradient(180deg, #123629 0%, #0f2d22 100%);
}

[data-testid="stSidebar"] * {
  color: #f8f3e7;
}

[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
  color: #1f2937 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stFileUploader > div {
  background: rgba(255, 250, 240, 0.96) !important;
  border-radius: 12px;
}

[data-testid="stSidebar"] .stFileUploader section,
[data-testid="stSidebar"] .stFileUploader section * ,
[data-testid="stSidebar"] .stFileUploader small,
[data-testid="stSidebar"] .stFileUploader button {
  color: #1f2937 !important;
}

[data-testid="stSidebar"] details {
  border: 1px solid rgba(248, 243, 231, 0.14) !important;
  border-radius: 14px !important;
  background: rgba(248, 243, 231, 0.06) !important;
  margin-bottom: 0.8rem !important;
}

[data-testid="stSidebar"] details summary {
  background: rgba(18, 54, 41, 0.88) !important;
  border-radius: 14px !important;
  padding: 0.35rem 0.6rem !important;
}

[data-testid="stSidebar"] details summary * {
  color: #fff7df !important;
}

[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .stSelectbox label {
  color: #f8f3e7 !important;
}

.sidebar-section {
  margin: 0.25rem 0 0.8rem 0;
  padding: 0.8rem 0.9rem;
  border: 1px solid rgba(248, 243, 231, 0.12);
  border-radius: 14px;
  background: rgba(248, 243, 231, 0.06);
}

.sidebar-section-title {
  font-size: 0.95rem;
  font-weight: 800;
  color: #fff7df;
}

.sidebar-section-copy {
  margin-top: 0.2rem;
  font-size: 0.82rem;
  color: rgba(248, 243, 231, 0.78);
  line-height: 1.45;
}

[data-testid="collapsedControl"] {
  background: rgba(18, 54, 41, 0.92) !important;
  border-radius: 10px !important;
  border: 1px solid rgba(248, 243, 231, 0.12) !important;
}

[data-testid="collapsedControl"] svg {
  fill: #fff7df !important;
}
</style>
"""


def _inject_theme():
  st.markdown(BASE_THEME_CSS, unsafe_allow_html=True)
  st.markdown(SIDEBAR_THEME_CSS, unsafe_allow_html=True)


def _render_hero():
  st.markdown(
    """
    <div class="hero-card">
      <h1 class="hero-title">Interview Copilot</h1>
      <div class="hero-subtitle">
        一个更像 GPT 的模拟面试界面。主区域只保留对话流和一个输入框，
        面试的开始与结束由你在左侧显式控制。
      </div>
    </div>
    """,
    unsafe_allow_html=True,
  )


def _render_status(model_provider: str, model: str):
  resume_count = len(st.session_state.get("pdf_files", []))
  interview_status = st.session_state.get("interview_status", "draft")

  st.markdown(
    f"""
    <div class="status-card">
      <div class="status-grid">
        <div>
          <div class="status-label">当前模型</div>
          <div class="status-value">{model_provider} / {model}</div>
        </div>
        <div>
          <div class="status-label">已提交简历</div>
          <div class="status-value">{resume_count} 份</div>
        </div>
        <div>
          <div class="status-label">面试状态</div>
          <div class="status-value">{interview_status}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
  )


def _render_section_header(title: str, copy: str):
  st.markdown(
    f"""
    <div class="content-card">
      <div class="section-title">{title}</div>
      <div class="section-copy">{copy}</div>
    </div>
    """,
    unsafe_allow_html=True,
  )


def main():
  st.set_page_config(
    page_title="Interview Copilot",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
  )
  setup_session_state()
  _inject_theme()

  with st.sidebar:
    with st.expander("\u914d\u7f6e", expanded=True):
      st.caption("\u9009\u62e9\u6a21\u578b\u5e76\u4e0a\u4f20\u7b80\u5386\u3002")
      model_provider, model = render_model_selector()
      sidebar_file_upload(model_provider)
      sidebar_provider_change_check(model_provider, model)

    with st.expander("\u9762\u8bd5\u914d\u7f6e", expanded=True):
      st.caption("\u586b\u5199\u5c97\u4f4d JD\uff0c\u70b9\u51fb\u5f00\u59cb\u9762\u8bd5\uff1b\u7ed3\u675f\u7531\u4f60\u624b\u52a8\u63a7\u5236\u3002")
      render_interview_setup(model_provider, model)

    with st.expander("\u5de5\u5177", expanded=True):
      st.caption("\u4f1a\u8bdd\u63a7\u5236\u4e0e\u5feb\u6377\u64cd\u4f5c")
      sidebar_utilities()

  _render_hero()
  st.write("")
  _render_status(model_provider or "Deepseek", model or "deepseek-reasoner")
  st.write("")
  render_uploaded_files_expander()

  if is_resume_ready():
    if is_interview_active():
      section_copy = "\u672c\u8f6e\u9762\u8bd5\u8fdb\u884c\u4e2d\u3002\u8bf7\u76f4\u63a5\u5728\u5e95\u90e8\u8f93\u5165\u6846\u56de\u7b54\u5f53\u524d\u95ee\u9898\u3002"
    elif st.session_state.get("interview_status") == "ended":
      section_copy = "\u672c\u8f6e\u9762\u8bd5\u5df2\u7ed3\u675f\u3002\u4f60\u53ef\u4ee5\u5728\u5de6\u4fa7\u751f\u6210\u62a5\u544a\uff0c\u6216\u518d\u6b21\u70b9\u51fb\u201c\u5f00\u59cb\u9762\u8bd5\u201d\u5f00\u542f\u4e0b\u4e00\u8f6e\u3002"
    else:
      section_copy = "\u8bf7\u5148\u5728\u5de6\u4fa7\u586b\u5199\u5c97\u4f4d JD \u5e76\u70b9\u51fb\u201c\u5f00\u59cb\u9762\u8bd5\u201d\uff0c\u7136\u540e\u5728\u5e95\u90e8\u8f93\u5165\u6846\u56de\u7b54\u3002"

    _render_section_header("\u9762\u8bd5\u5bf9\u8bdd", section_copy)
    render_interview_history()
    render_interview_report()
    render_unified_input(model_provider, model)
  else:
    _render_section_header(
      "\u5f00\u59cb\u4f7f\u7528",
      "\u8bf7\u5148\u5728\u5de6\u4fa7\u9009\u62e9\u6a21\u578b\u3001\u4e0a\u4f20\u5e76\u63d0\u4ea4\u7b80\u5386\u3002\u7136\u540e\u586b\u5199\u5c97\u4f4d JD\uff0c\u5e76\u7531\u4f60\u624b\u52a8\u5f00\u59cb\u672c\u8f6e\u9762\u8bd5\u3002",
    )


if __name__ == "__main__":
  main()
