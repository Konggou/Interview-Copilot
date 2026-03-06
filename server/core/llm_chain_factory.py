import json
import time

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError

from config.settings import (
  DEEPSEEK_API_KEY,
  DEEPSEEK_BASE_URL,
  DEEPSEEK_MODEL,
)
from utils.logger import logger


INTERVIEW_CONTEXT_CHAR_LIMIT = 1600
INTERVIEW_HISTORY_CHAR_LIMIT = 900
LLM_RETRY_DELAY_SECONDS = 1
LLM_MAX_RETRIES = 3


def get_prompt():
  logger.debug("Creating chat prompt template.")
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
        "\u4f60\u662f\u4e00\u540d\u8d44\u6df1\u4e2d\u6587\u6280\u672f\u9762\u8bd5\u5b98\uff0c\u6b63\u5728\u8fdb\u884c\u4e00\u573a\u771f\u5b9e\u7684\u6280\u672f\u9762\u8bd5\u3002"
        "\u4f60\u5fc5\u987b\u59cb\u7ec8\u4ee5\u9762\u8bd5\u5b98\u89c6\u89d2\u53d1\u8a00\u3002"
        "\u4f60\u53ea\u80fd\u4f7f\u7528\u7ed9\u5b9a\u7684\u7b80\u5386\u8bc1\u636e\u63d0\u95ee\u3002"
        "\u5982\u679c\u63d0\u4f9b\u4e86\u5c97\u4f4d JD\uff0c\u8bf7\u7ed3\u5408\u5c97\u4f4d\u8981\u6c42\u8c03\u6574\u95ee\u9898\u91cd\u70b9\u548c\u6df1\u5ea6\u3002"
        "\u7981\u6b62\u7f16\u9020\u7b80\u5386\u4e2d\u4e0d\u5b58\u5728\u7684\u6280\u80fd\u3001\u9879\u76ee\u3001\u7ecf\u5386\u6216\u7ed3\u679c\u3002"
        "\u8bf7\u5148\u7ed9\u51fa\u4e00\u53e5\u7b80\u77ed\u81ea\u7136\u7684\u9762\u8bd5\u5b98\u5f00\u573a\u767d\uff0c\u7136\u540e\u53ea\u63d0\u51fa\u4e00\u9053\u9996\u8f6e\u95ee\u9898\u3002"
        "\u5fc5\u987b\u8fd4\u56de\u4e25\u683c JSON\uff0c\u683c\u5f0f\u5982\u4e0b\uff1a"
        "{{\"opening_message\":\"...\","
        "\"question\":{{\"id\":\"q1\",\"question\":\"...\",\"focus\":\"...\",\"difficulty\":\"easy|mid|hard\"}},"
        "\"rubric\":{{\"score_scale\":\"1-10\"}}}}"
      ),
    ),
    (
      "human",
      "\u7b80\u5386\u8bc1\u636e\uff1a\n{context}\n\n\u5c97\u4f4d JD\uff1a\n{job_description}\n\n\u5f00\u573a\u98ce\u683c\uff1a\n{opening_style}",
    ),
  ])


def get_next_interview_question_prompt():
  return ChatPromptTemplate.from_messages([
    (
      "system",
      (
        "\u4f60\u662f\u4e00\u540d\u8d44\u6df1\u4e2d\u6587\u6280\u672f\u9762\u8bd5\u5b98\uff0c\u6b63\u5728\u7ee7\u7eed\u4e00\u573a\u771f\u5b9e\u7684\u6280\u672f\u9762\u8bd5\u3002"
        "\u4f60\u5fc5\u987b\u59cb\u7ec8\u4ee5\u9762\u8bd5\u5b98\u89c6\u89d2\u53d1\u8a00\uff0c\u5e76\u4e14\u4e00\u6b21\u53ea\u63d0\u51fa\u4e00\u9053\u4e0b\u4e00\u8f6e\u95ee\u9898\u3002"
        "\u4f60\u53ea\u80fd\u4f7f\u7528\u7ed9\u5b9a\u7684\u7b80\u5386\u8bc1\u636e\u63d0\u95ee\u3002"
        "\u5982\u679c\u63d0\u4f9b\u4e86\u5c97\u4f4d JD\uff0c\u8bf7\u4fdd\u6301\u95ee\u9898\u4e0e\u5c97\u4f4d\u8981\u6c42\u4e00\u81f4\u3002"
        "\u5386\u53f2\u8f6e\u6b21\u53ea\u7528\u4e8e\u4fdd\u6301\u5bf9\u8bdd\u8fde\u8d2f\u6027\uff0c\u4e0d\u80fd\u8986\u76d6\u7b80\u5386\u8bc1\u636e\u3002"
        "\u7981\u6b62\u7f16\u9020\u7b80\u5386\u4e2d\u4e0d\u5b58\u5728\u7684\u6280\u80fd\u3001\u9879\u76ee\u3001\u7ecf\u5386\u6216\u7ed3\u679c\u3002"
        "\u5fc5\u987b\u8fd4\u56de\u4e25\u683c JSON\uff0c\u683c\u5f0f\u5982\u4e0b\uff1a"
        "{{\"question\":{{\"id\":\"q_next\",\"question\":\"...\",\"focus\":\"...\",\"difficulty\":\"easy|mid|hard\"}}}}"
      ),
    ),
    (
      "human",
      (
        "\u7b80\u5386\u8bc1\u636e\uff1a\n{context}\n\n"
        "\u5c97\u4f4d JD\uff1a\n{job_description}\n\n"
        "\u5386\u53f2\u9762\u8bd5\u8f6e\u6b21\u6458\u8981\uff1a\n{prior_conversation}\n\n"
        "\u4e0a\u4e00\u8f6e\u53cd\u9988\u6458\u8981\uff1a\n{latest_feedback_summary}"
      ),
    ),
  ])


def get_interview_feedback_prompt():
  return ChatPromptTemplate.from_messages([
    (
      "system",
      (
        "\u4f60\u662f\u4e00\u540d\u8d44\u6df1\u4e2d\u6587\u6280\u672f\u9762\u8bd5\u5b98\u3002"
        "\u4f60\u5fc5\u987b\u4ece\u9762\u8bd5\u5b98\u89c6\u89d2\u8bc4\u4f30\u5019\u9009\u4eba\u7684\u56de\u7b54\u3002"
        "\u4f60\u53ea\u80fd\u4f9d\u636e\u7ed9\u5b9a\u7684\u7b80\u5386\u8bc1\u636e\u8fdb\u884c\u8bc4\u5206\u548c\u53cd\u9988\u3002"
        "\u5386\u53f2\u8f6e\u6b21\u53ea\u7528\u4e8e\u5224\u65ad\u8868\u8ff0\u524d\u540e\u4e00\u81f4\u6027\uff0c\u4e0d\u80fd\u8986\u76d6\u7b80\u5386\u8bc1\u636e\u3002"
        "\u5982\u679c\u8bc1\u636e\u4e0d\u8db3\uff0c\u8bf7\u660e\u786e\u6307\u51fa\u8bc1\u636e\u4e0d\u8db3\uff0c\u4f46\u4e0d\u8981\u7f16\u9020\u7b80\u5386\u4e8b\u5b9e\u3002"
        "\u8bf7\u4f18\u5148\u5224\u65ad\u56de\u7b54\u662f\u5426\u4e0e\u73b0\u6709\u7b80\u5386\u8bc1\u636e\u76f8\u7b26\uff0c\u518d\u7ed9\u51fa\u6539\u8fdb\u5efa\u8bae\u3002"
        "\u5fc5\u987b\u8fd4\u56de\u4e25\u683c JSON\uff0c\u683c\u5f0f\u5982\u4e0b\uff1a"
        "{{\"score\":0-10,\"summary\":\"...\",\"strengths\":[\"...\"],"
        "\"weaknesses\":[\"...\"],\"suggestions\":[\"...\"],"
        "\"followup_question\":\"...\"}}"
      ),
    ),
    (
      "human",
      (
        "\u5f53\u524d\u95ee\u9898\uff1a\n{question}\n\n"
        "\u5386\u53f2\u9762\u8bd5\u8f6e\u6b21\u6458\u8981\uff1a\n{prior_conversation}\n\n"
        "\u7b80\u5386\u8bc1\u636e\uff1a\n{context}\n\n"
        "\u5019\u9009\u4eba\u56de\u7b54\uff1a\n{user_answer}"
      ),
    ),
  ])


def get_llm(model_provider: str, model: str):
  logger.debug(f"Initializing LLM for {model_provider} - {model}")
  if model_provider != "deepseek":
    logger.error(f"Unsupported LLM Provider: {model_provider}")
    raise ValueError(f"Unsupported LLM Provider: {model_provider}")

  return ChatOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model=model,
    temperature=0.2,
    timeout=45,
  )


def build_llm_chain(model_provider: str, model: str, vectorstore):
  logger.debug(f"Building LLM chain for provider: {model_provider}, model: {model}")
  prompt = get_prompt()
  llm = get_llm(model_provider, model)
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

  return create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(llm, prompt=prompt),
  )


def _extract_json_text(raw_content):
  if isinstance(raw_content, str):
    return raw_content.strip()
  if isinstance(raw_content, list):
    text_parts = []
    for item in raw_content:
      if isinstance(item, dict) and item.get("type") == "text":
        text_parts.append(item.get("text", ""))
      elif hasattr(item, "text"):
        text_parts.append(item.text)
      else:
        text_parts.append(str(item))
    return "".join(text_parts).strip()
  return str(raw_content).strip()


def _parse_json_response(raw_content):
  text = _extract_json_text(raw_content)
  if text.startswith("```"):
    text = text.strip("`")
    if text.lower().startswith("json"):
      text = text[4:].strip()
  start = text.find("{")
  end = text.rfind("}")
  if start != -1 and end != -1:
    text = text[start : end + 1]
  return json.loads(text)


def _truncate_text(text: str, limit: int) -> str:
  if len(text) <= limit:
    return text
  return f"{text[: limit - 3].rstrip()}..."


def _invoke_json_prompt(model_provider: str, model: str, prompt, payload: dict):
  llm = get_llm(model_provider, model)
  safe_payload = dict(payload)
  if "context" in safe_payload:
    safe_payload["context"] = _truncate_text(
      safe_payload["context"],
      INTERVIEW_CONTEXT_CHAR_LIMIT,
    )
  if "prior_conversation" in safe_payload:
    safe_payload["prior_conversation"] = _truncate_text(
      safe_payload["prior_conversation"],
      INTERVIEW_HISTORY_CHAR_LIMIT,
    )

  message = prompt.format_messages(**safe_payload)
  last_error = None

  for attempt in range(1, LLM_MAX_RETRIES + 1):
    try:
      response = llm.invoke(message)
      parsed = _parse_json_response(response.content)
      logger.debug("Structured JSON response parsed successfully.")
      return parsed
    except APIConnectionError as e:
      last_error = e
      logger.warning(f"LLM connection error on attempt {attempt}/{LLM_MAX_RETRIES}: {e}")
      if attempt < LLM_MAX_RETRIES:
        time.sleep(LLM_RETRY_DELAY_SECONDS)

  raise last_error


def generate_initial_interview_turn(
  model_provider: str,
  model: str,
  context: str,
  job_description: str = "",
  opening_style: str = "",
):
  return _invoke_json_prompt(
    model_provider,
    model,
    get_initial_interview_prompt(),
    {
      "context": context,
      "job_description": (job_description or "").strip() or "\u672a\u63d0\u4f9b\u3002",
      "opening_style": (opening_style or "").strip() or "\u4e13\u4e1a\u3001\u81ea\u7136\u3001\u4e2d\u6587\u6280\u672f\u9762\u8bd5\u5b98\u98ce\u683c\u3002",
    },
  )


def generate_next_interview_question(
  model_provider: str,
  model: str,
  context: str,
  job_description: str = "",
  prior_conversation: str = "",
  latest_feedback_summary: str = "",
):
  return _invoke_json_prompt(
    model_provider,
    model,
    get_next_interview_question_prompt(),
    {
      "context": context,
      "job_description": (job_description or "").strip() or "\u672a\u63d0\u4f9b\u3002",
      "prior_conversation": (prior_conversation or "").strip() or "\u6682\u65e0\u5386\u53f2\u8f6e\u6b21\u3002",
      "latest_feedback_summary": (
        latest_feedback_summary or ""
      ).strip() or "\u6682\u65e0\u4e0a\u4e00\u8f6e\u53cd\u9988\u3002",
    },
  )


def evaluate_interview_answer(
  model_provider: str,
  model: str,
  question: str,
  user_answer: str,
  context: str,
  prior_conversation: str = "",
):
  return _invoke_json_prompt(
    model_provider,
    model,
    get_interview_feedback_prompt(),
    {
      "question": question,
      "user_answer": user_answer,
      "context": context,
      "prior_conversation": (prior_conversation or "").strip() or "\u6682\u65e0\u5386\u53f2\u8f6e\u6b21\u3002",
    },
  )


def get_default_model(model_provider: str) -> str:
  if model_provider != "deepseek":
    raise ValueError(f"Unsupported LLM Provider: {model_provider}")
  return DEEPSEEK_MODEL
