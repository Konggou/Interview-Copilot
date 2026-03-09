from langchain_core.prompts import ChatPromptTemplate

from config.settings import (
  DEEPSEEK_MODEL,
)


INTERVIEW_CONTEXT_CHAR_LIMIT = 1600
INTERVIEW_HISTORY_CHAR_LIMIT = 900


def get_prompt():
  return ChatPromptTemplate.from_messages([
    (
      "system",
      "Answer as detailed as possible using the context below. If unknown, say 'I don't know.'",
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
  ])


def get_initial_interview_prompt():
  return ChatPromptTemplate.from_messages([
    (
      "system",
      (
        "你是一名资深中文技术面试官，正在进行一场真实的技术面试。"
        "你必须始终以面试官视角发言。"
        "你只能使用给定的简历证据提问。"
        "如果提供了岗位 JD，请结合岗位要求调整问题重点和深度。"
        "禁止编造简历中不存在的技能、项目、经历或结果。"
        "请严格按下面格式返回，字段名必须保持一致：\n"
        "开场白：<一句自然的面试官开场白>\n"
        "问题：<第一道问题>\n"
        "考察点：<一句话>\n"
        "难度：easy|mid|hard\n"
        "评分标准：1-10"
      ),
    ),
    (
      "human",
      "简历证据：\n{context}\n\n岗位 JD：\n{job_description}\n\n开场风格：\n{opening_style}",
    ),
  ])


def get_next_interview_question_prompt():
  return ChatPromptTemplate.from_messages([
    (
      "system",
      (
        "你是一名资深中文技术面试官，正在继续一场真实的技术面试。"
        "你必须始终以面试官视角发言，一次只提出一道问题。"
        "你只能使用给定的简历证据提问。"
        "如果提供了岗位 JD，请保持问题与岗位要求一致。"
        "历史轮次只用于保持连贯性，不能覆盖简历证据。"
        "禁止编造简历中不存在的技能、项目、经历或结果。"
        "请严格按下面格式返回，字段名必须保持一致：\n"
        "问题：<下一道问题>\n"
        "考察点：<一句话>\n"
        "难度：easy|mid|hard"
      ),
    ),
    (
      "human",
      (
        "简历证据：\n{context}\n\n"
        "岗位 JD：\n{job_description}\n\n"
        "历史面试轮次摘要：\n{prior_conversation}\n\n"
        "上一轮反馈摘要：\n{latest_feedback_summary}"
      ),
    ),
  ])


def get_interview_feedback_prompt():
  return ChatPromptTemplate.from_messages([
    (
      "system",
      (
        "你是一名资深中文技术面试官。"
        "你必须从面试官视角评估候选人的回答。"
        "你只能依据给定的简历证据进行评分和反馈。"
        "历史轮次只用于判断表述前后一致性，不能覆盖简历证据。"
        "如证据不足，请明确指出证据不足，但不要编造简历事实。"
        "请严格按下面格式返回，字段名必须保持一致：\n"
        "评分：<0-10 的整数>\n"
        "总结：<一段总结>\n"
        "优点：\n- <第一条>\n- <第二条>\n"
        "不足：\n- <第一条>\n- <第二条>\n"
        "建议：\n- <第一条>\n- <第二条>\n"
        "追问：<一句话，没有就写 无>"
      ),
    ),
    (
      "human",
      (
        "当前问题：\n{question}\n\n"
        "历史面试轮次摘要：\n{prior_conversation}\n\n"
        "简历证据：\n{context}\n\n"
        "候选人回答：\n{user_answer}"
      ),
    ),
  ])


def _truncate_text(text: str, limit: int) -> str:
  if len(text) <= limit:
    return text
  return f"{text[: limit - 3].rstrip()}..."


def _prompt_to_messages(prompt, payload: dict):
  messages = prompt.format_messages(**payload)
  serialized = []
  for message in messages:
    role = message.type
    if role == "human":
      role = "user"
    elif role == "ai":
      role = "assistant"
    serialized.append({
      "role": role,
      "content": message.content,
    })
  return serialized


def build_chat_messages(context: str, user_input: str):
  return _prompt_to_messages(
    get_prompt(),
    {
      "context": context,
      "input": user_input,
    },
  )


def build_initial_interview_messages(
  context: str,
  job_description: str = "",
  opening_style: str = "",
):
  return _prompt_to_messages(
    get_initial_interview_prompt(),
    {
      "context": _truncate_text(context, INTERVIEW_CONTEXT_CHAR_LIMIT),
      "job_description": (job_description or "").strip() or "未提供。",
      "opening_style": (opening_style or "").strip() or "专业、自然、中文技术面试官风格。",
    },
  )


def build_next_interview_question_messages(
  context: str,
  job_description: str = "",
  prior_conversation: str = "",
  latest_feedback_summary: str = "",
):
  return _prompt_to_messages(
    get_next_interview_question_prompt(),
    {
      "context": _truncate_text(context, INTERVIEW_CONTEXT_CHAR_LIMIT),
      "job_description": (job_description or "").strip() or "未提供。",
      "prior_conversation": _truncate_text(
        (prior_conversation or "").strip() or "暂无历史轮次。",
        INTERVIEW_HISTORY_CHAR_LIMIT,
      ),
      "latest_feedback_summary": (
        latest_feedback_summary or ""
      ).strip() or "暂无上一轮反馈。",
    },
  )


def build_interview_feedback_messages(
  question: str,
  user_answer: str,
  context: str,
  prior_conversation: str = "",
):
  return _prompt_to_messages(
    get_interview_feedback_prompt(),
    {
      "question": question,
      "user_answer": user_answer,
      "context": _truncate_text(context, INTERVIEW_CONTEXT_CHAR_LIMIT),
      "prior_conversation": _truncate_text(
        (prior_conversation or "").strip() or "暂无历史轮次。",
        INTERVIEW_HISTORY_CHAR_LIMIT,
      ),
    },
  )


def _extract_field(lines: list[str], field_name: str) -> str:
  for line in lines:
    for prefix in (f"{field_name}：", f"{field_name}:"):
      if line.startswith(prefix):
        return line[len(prefix) :].strip()
  return ""


def _extract_list(lines: list[str], field_name: str) -> list[str]:
  items = []
  start_index = None
  for index, line in enumerate(lines):
    if line in (f"{field_name}：", f"{field_name}:"):
      start_index = index
      break
  if start_index is None:
    return items

  for line in lines[start_index + 1 :]:
    if not line:
      continue
    if (("：" in line) or (":" in line)) and not line.startswith("- "):
      break
    if line.startswith("- "):
      items.append(line[2:].strip())
  return items


def parse_initial_interview_response(raw_text: str) -> dict:
  lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
  question_text = _extract_field(lines, "问题")
  return {
    "opening_message": _extract_field(lines, "开场白"),
    "question": {
      "id": "q1",
      "question": question_text,
      "focus": _extract_field(lines, "考察点") or "简历匹配度",
      "difficulty": _extract_field(lines, "难度") or "mid",
    },
    "rubric": {
      "score_scale": _extract_field(lines, "评分标准") or "1-10",
    },
  }


def parse_next_interview_question_response(raw_text: str, fallback_id: str) -> dict:
  lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
  return {
    "question": {
      "id": fallback_id,
      "question": _extract_field(lines, "问题"),
      "focus": _extract_field(lines, "考察点") or "简历匹配度",
      "difficulty": _extract_field(lines, "难度") or "mid",
    },
  }


def parse_interview_feedback_response(raw_text: str) -> dict:
  lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
  score_text = _extract_field(lines, "评分") or "0"
  try:
    score = int(score_text)
  except ValueError:
    score = 0

  followup_question = _extract_field(lines, "追问")
  if followup_question == "无":
    followup_question = ""

  return {
    "score": max(0, min(score, 10)),
    "summary": _extract_field(lines, "总结"),
    "strengths": _extract_list(lines, "优点"),
    "weaknesses": _extract_list(lines, "不足"),
    "suggestions": _extract_list(lines, "建议"),
    "followup_question": followup_question,
  }


def get_default_model(model_provider: str) -> str:
  if model_provider != "deepseek":
    raise ValueError(f"Unsupported LLM Provider: {model_provider}")
  return DEEPSEEK_MODEL
