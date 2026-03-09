from types import SimpleNamespace

import streamlit as st

from components.chat import reset_interview_view_state
from components.interview import render_interview_sidebar_controls
from utils.helpers import (
  get_model_providers,
  get_models,
  process_uploaded_pdfs,
)


LOCAL_FALLBACK_PROVIDERS = ["Deepseek"]
LOCAL_FALLBACK_MODELS = {
  "Deepseek": ["deepseek-reasoner", "deepseek-chat"],
}


def render_sidebar_section_title(title: str, subtitle: str = ""):
  subtitle_html = f'<div class="sidebar-section-copy">{subtitle}</div>' if subtitle else ""
  st.markdown(
    f"""
    <div class="sidebar-section">
      <div class="sidebar-section-title">{title}</div>
      {subtitle_html}
    </div>
    """,
    unsafe_allow_html=True,
  )


def _get_provider_options():
  try:
    return get_model_providers()
  except Exception:
    st.warning("\u540e\u7aef\u670d\u52a1\u4e0d\u53ef\u7528\uff0c\u5df2\u5207\u6362\u4e3a\u672c\u5730\u9ed8\u8ba4\u6a21\u578b\u9009\u9879\u3002")
    return LOCAL_FALLBACK_PROVIDERS


def _get_model_options(model_provider: str):
  try:
    return get_models(model_provider)
  except Exception:
    return LOCAL_FALLBACK_MODELS.get(model_provider, ["deepseek-reasoner"])


def render_model_selector():
  providers = _get_provider_options()
  if not providers:
    st.session_state["model_provider"] = ""
    st.session_state["model"] = ""
    return "", ""

  if st.session_state.get("model_provider") not in providers:
    st.session_state["model_provider"] = providers[0]

  model_provider = st.selectbox(
    "\u6a21\u578b\u63d0\u4f9b\u5546",
    options=providers,
    key="model_provider",
  )

  models = _get_model_options(model_provider)
  if not models:
    st.session_state["model"] = ""
    return model_provider or "", ""

  if st.session_state.get("model") not in models:
    st.session_state["model"] = models[0]

  model = st.selectbox(
    "\u6a21\u578b\u540d\u79f0",
    options=models,
    disabled=not model_provider,
    key="model",
  )

  return model_provider or "", model or ""


def render_upload_files_button():
  uploaded_files = st.file_uploader(
    "\u4e0a\u4f20\u7b80\u5386 PDF",
    type=["pdf"],
    accept_multiple_files=True,
    disabled=not st.session_state.get("model"),
    key=f"uploaded_files_{st.session_state.get('uploader_key')}",
  )

  uploaded_filenames = [f.name for f in uploaded_files] if uploaded_files else []
  session_filenames = [f.name for f in st.session_state.get("pdf_files", [])]
  if uploaded_files and uploaded_filenames != session_filenames:
    st.session_state.update(unsubmitted_files=True)

  submitted = st.button(
    "\u63d0\u4ea4\u7b80\u5386",
    disabled=not st.session_state.get("model"),
    use_container_width=True,
  )
  return uploaded_files, submitted


def sidebar_file_upload(model_provider):
  uploaded_files, submitted = render_upload_files_button()

  if submitted:
    if uploaded_files:
      file_objs = [
        SimpleNamespace(name=f.name, type=f.type, data=f.read())
        for f in uploaded_files
      ]

      with st.spinner("\u6b63\u5728\u5904\u7406\u7b80\u5386\u6587\u4ef6..."):
        try:
          process_uploaded_pdfs(model_provider, file_objs)
          reset_interview_view_state()
          st.session_state.update(
            resume_ready=True,
          )
        except Exception as e:
          st.error(f"\u9519\u8bef\uff1a{str(e)}")
          return uploaded_files, submitted

      st.session_state.update(
        pdf_files=file_objs,
        unsubmitted_files=False,
      )
      st.toast("\u7b80\u5386\u5904\u7406\u5b8c\u6210\u3002")
    else:
      st.warning("\u5c1a\u672a\u4e0a\u4f20\u6587\u4ef6\u3002")

  return uploaded_files, submitted


def sidebar_provider_change_check(model_provider, model):
  if model_provider != st.session_state.get("last_provider"):
    st.session_state.update(resume_ready=False)
    if model:
      st.session_state.update(last_provider=model_provider)
      if st.session_state.get("pdf_files"):
        with st.spinner(f"\u6b63\u5728\u4f7f\u7528 {model_provider} \u91cd\u65b0\u5904\u7406\u7b80\u5386..."):
          try:
            process_uploaded_pdfs(model_provider, st.session_state.get("pdf_files"))
            reset_interview_view_state()
            st.session_state.update(
              resume_ready=True,
              unsubmitted_files=False,
            )
          except Exception as e:
            st.error(f"\u9519\u8bef\uff1a{str(e)}")
            return

        st.toast("\u7b80\u5386\u91cd\u65b0\u5904\u7406\u5b8c\u6210\u3002")


def render_interview_setup(model_provider, model):
  render_interview_sidebar_controls(model_provider, model)


def sidebar_utilities():
  col1, col2 = st.columns(2)

  if col1.button("\u91cd\u7f6e", use_container_width=True):
    st.session_state.clear()
    st.session_state["model_provider"] = "Deepseek"
    st.session_state["model"] = "deepseek-reasoner"
    st.toast("\u4f1a\u8bdd\u5df2\u91cd\u7f6e\u3002")
    st.rerun()

  if col2.button("\u6e05\u7a7a\u7b80\u5386", use_container_width=True):
    st.session_state.update(
      pdf_files=[],
      resume_ready=False,
      unsubmitted_files=False,
    )
    reset_interview_view_state()
    st.session_state.uploader_key += 1
    st.toast("\u7b80\u5386\u4e0e\u9762\u8bd5\u72b6\u6001\u5df2\u6e05\u7a7a\u3002")
    st.rerun()
