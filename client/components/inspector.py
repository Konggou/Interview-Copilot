import streamlit as st

from state.session import is_chat_ready
from utils.helpers import (
  get_documents_count,
  get_similar_chunks,
)


def render_inspect_query(model_provider):
  st.caption("查看向量库检索结果")
  try:
    doc_count = get_documents_count(model_provider)
    st.success(f"当前向量库中共有 {doc_count} 条文档片段。")
  except Exception as e:
    st.error("无法获取文档数量。")
    st.code(str(e))

  query = st.chat_input(
    "输入一个测试检索的问题",
    disabled=not is_chat_ready(),
  )

  if not query:
    return

  with st.chat_message("user"):
    st.markdown(query)
  with st.chat_message("ai"):
    with st.spinner("正在检索..."):
      try:
        results = get_similar_chunks(model_provider, query)
        if results:
          st.markdown("### 最相关的文档片段")
          for i, item in enumerate(results):
            score = item.get("score")
            content = item.get("page_content", "")[:300]
            score_display = f"（分数：{score:.4f}）" if score is not None else ""
            st.markdown(f"**结果 {i + 1} {score_display}**\n\n{content}...")
        else:
          st.info("未找到匹配的文档片段。")
      except Exception as e:
        st.error("检索向量库时发生错误。")
        st.code(str(e))
