import os
from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from api.schemas import (
  ChatRequest,
  InterviewAnswerRequest,
  InterviewEndRequest,
  InterviewStartRequest,
  SearchQueryRequest,
)
from config.settings import (
  INTERVIEW_CONTEXT_SNIPPET_LENGTH,
  MODEL_OPTIONS,
  SIMILARITY_THRESHOLD,
  VECTORSTORE_DIRECTORY,
)
from core.interview_session_store import (
  append_turn,
  get_completed_turns,
  get_current_turn,
  get_session,
  mark_session_status,
  save_session,
  set_current_question,
  update_session_answer,
  update_session_report,
)
from core.llm_chain_factory import (
  build_chat_messages,
  build_initial_interview_messages,
  build_interview_feedback_messages,
  build_next_interview_question_messages,
  get_default_model,
  parse_initial_interview_response,
  parse_interview_feedback_response,
  parse_next_interview_question_response,
)
from core.llm_service import (
  get_interview_semaphore,
  inspect_completion_cache,
  invoke_completion,
  stream_completion,
)
from core.metrics import render_metrics
from core.semantic_cache import ping as ping_redis
from core.sse import format_sse_event
from core.vector_database import (
  find_similar_chunks,
  get_collections_count,
  retrieve_scored_chunks,
  serialize_search_results,
  upsert_vectorstore_from_pdfs,
)
from utils.logger import logger


router = APIRouter()


def _success_response(data=None, message: str | None = None, headers: dict | None = None):
  return JSONResponse(
    content={
      "status": "success",
      "data": data,
      "message": message,
    },
    headers=headers or {},
  )


def _error_response(
  message: str,
  headers: dict | None = None,
  status_code: int = 400,
):
  return JSONResponse(
    content={
      "status": "error",
      "data": None,
      "message": message,
    },
    headers=headers or {},
    status_code=status_code,
  )


def _validate_model(model_provider: str, model_name: str | None):
  normalized_provider = model_provider.lower()
  if normalized_provider not in MODEL_OPTIONS:
    raise ValueError("Invalid model provider.")

  resolved_model = model_name or get_default_model(normalized_provider)
  if resolved_model not in MODEL_OPTIONS[normalized_provider]["models"]:
    raise ValueError("Invalid model name.")

  return normalized_provider, resolved_model


def _compact_sources(results_with_scores):
  return [
    {
      "source": item["source"],
      "page": item["page"],
      "score": item["score"],
      "snippet": item["snippet"],
    }
    for item in serialize_search_results(results_with_scores)
  ]


def _build_interview_context(results_with_scores) -> str:
  context_blocks = []
  for item in serialize_search_results(
    results_with_scores,
    snippet_length=INTERVIEW_CONTEXT_SNIPPET_LENGTH,
  ):
    context_blocks.append(
      (
        f"来源: {item['source']}\n"
        f"页码: {item['page']}\n"
        f"片段: {item['snippet']}"
      ),
    )
  return "\n\n".join(context_blocks)


def _build_context_from_compact_sources(compact_sources: list[dict]) -> str:
  context_blocks = []
  for item in compact_sources:
    context_blocks.append(
      (
        f"来源: {item.get('source', 'Unknown')}\n"
        f"页码: {item.get('page', 0)}\n"
        f"片段: {item.get('snippet', '')}"
      ),
    )
  return "\n\n".join(context_blocks)


def _normalize_question(raw_question: dict | None, fallback_id: str) -> dict | None:
  if not isinstance(raw_question, dict):
    return None

  question_text = str(raw_question.get("question", "")).strip()
  if not question_text:
    return None

  return {
    "id": str(raw_question.get("id") or fallback_id),
    "question": question_text,
    "focus": str(raw_question.get("focus", "简历匹配度")).strip() or "简历匹配度",
    "difficulty": str(raw_question.get("difficulty", "mid")).strip() or "mid",
  }


async def _build_recent_interview_memory(session_id: str, limit: int = 3) -> str:
  completed_turns = (await get_completed_turns(session_id))[-limit:]
  return _build_recent_interview_memory_from_turns(completed_turns)


def _build_recent_interview_memory_from_turns(turns: list[dict], limit: int = 3) -> str:
  completed_turns = turns[-limit:]
  if not completed_turns:
    return ""

  blocks = []
  for index, turn in enumerate(completed_turns, start=1):
    feedback = turn.get("feedback", {})
    blocks.append(
      (
        f"第 {index} 轮\n"
        f"问题: {turn.get('question', '')}\n"
        f"回答: {turn.get('user_answer', '')}\n"
        f"反馈摘要: {feedback.get('summary', '')}\n"
        f"分数: {feedback.get('score', '')}"
      ),
    )
  return "\n\n".join(blocks)


def _fallback_feedback():
  return {
    "score": 0,
    "summary": "无法从简历判断",
    "strengths": [],
    "weaknesses": ["简历中缺少与该问题直接相关的证据。"],
    "suggestions": ["补充更具体的项目经历、技术实现和量化结果。"],
    "followup_question": "",
    "sources": [],
  }


def _get_turn_count(session: dict) -> int:
  return len(session.get("turns", []))


def _build_turn_payload(question: dict, sources: list[dict], turn_index: int) -> dict:
  turn_id = f"t{turn_index}"
  return {
    "turn_id": turn_id,
    "question_id": question["id"],
    "question": question["question"],
    "focus": question.get("focus", ""),
    "difficulty": question.get("difficulty", "mid"),
    "sources": list(sources),
    "user_answer": "",
    "feedback": {},
    "created_at": datetime.now(timezone.utc).isoformat(),
    "answered_at": None,
  }


def _sse_headers(cache_status: str):
  return {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Cache": cache_status,
  }


def _build_report(session: dict):
  turns = session.get("turns", [])
  scores = []
  strengths = []
  weaknesses = []
  suggestions = []

  for turn in turns:
    feedback = turn.get("feedback", {})
    score = feedback.get("score")
    if isinstance(score, (int, float)):
      scores.append(score)
    strengths.extend(feedback.get("strengths", []))
    weaknesses.extend(feedback.get("weaknesses", []))
    suggestions.extend(feedback.get("suggestions", []))

  answered_turns = [turn for turn in turns if turn.get("user_answer")]
  average_score = round(sum(scores) / len(scores), 2) if scores else 0

  return {
    "session_id": session.get("session_id", ""),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "question_count": len(turns),
    "answered_count": len(answered_turns),
    "average_score": average_score,
    "rubric": session.get("rubric", {"score_scale": "1-10"}),
    "job_description": session.get("jd_text", ""),
    "summary": f"本轮面试共完成 {len(answered_turns)} 轮，平均分 {average_score}/10。",
    "strengths": list(dict.fromkeys(strengths))[:5],
    "weaknesses": list(dict.fromkeys(weaknesses))[:5],
    "suggestions": list(dict.fromkeys(suggestions))[:5],
    "details": [
      {
        "turn_id": turn.get("turn_id", ""),
        "question_id": turn.get("question_id", ""),
        "question": turn.get("question", ""),
        "focus": turn.get("focus", ""),
        "difficulty": turn.get("difficulty", ""),
        "answer": turn.get("user_answer", ""),
        "feedback": turn.get("feedback", {}),
      }
      for turn in turns
    ],
  }


def _build_report_markdown(report: dict):
  lines = [
    "# 面试报告",
    "",
    f"- 会话 ID: {report['session_id']}",
    f"- 生成时间: {report['generated_at']}",
    f"- 完成轮次: {report['answered_count']}",
    f"- 平均分: {report['average_score']}/10",
    "",
    "## 总结",
    report["summary"],
    "",
    "## 主要优点",
  ]

  if report["strengths"]:
    lines.extend([f"- {item}" for item in report["strengths"]])
  else:
    lines.append("- 暂无记录。")

  lines.extend(["", "## 主要不足"])
  if report["weaknesses"]:
    lines.extend([f"- {item}" for item in report["weaknesses"]])
  else:
    lines.append("- 暂无记录。")

  lines.extend(["", "## 改进建议"])
  if report["suggestions"]:
    lines.extend([f"- {item}" for item in report["suggestions"]])
  else:
    lines.append("- 暂无记录。")

  lines.extend(["", "## 分轮详情"])
  for index, detail in enumerate(report["details"], start=1):
    lines.extend([
      f"### 第 {index} 轮",
      f"- 问题: {detail['question']}",
      f"- 考察点: {detail['focus'] or '暂无'}",
      f"- 难度: {detail['difficulty'] or '暂无'}",
      f"- 回答: {detail['answer'] or '未作答'}",
      f"- 总结: {detail['feedback'].get('summary', '暂无')}",
      f"- 分数: {detail['feedback'].get('score', '暂无')}",
      "",
    ])

  return "\n".join(lines).strip()


async def _retrieve_question_specific_sources(
  model_provider: str,
  question_text: str,
  user_answer: str = "",
  fallback_sources: list[dict] | None = None,
):
  query = question_text.strip()
  if user_answer.strip():
    query = f"{question_text}\n候选人回答：{user_answer}"

  try:
    retrieval = await retrieve_scored_chunks(
      model_provider,
      query,
      k=3,
      threshold=SIMILARITY_THRESHOLD,
    )
    if retrieval["results"]:
      return _compact_sources(retrieval["results"])
  except Exception as exc:
    logger.warning(
      "Question-specific retrieval failed, falling back to stored sources",
      error=str(exc),
    )

  return list(fallback_sources or [])


def _feedback_as_text(result: dict) -> str:
  lines = [
    f"评分：{result.get('score', 0)}/10",
    f"总结：{result.get('summary', '')}",
  ]

  if result.get("strengths"):
    lines.append("优点：")
    lines.extend([f"- {item}" for item in result["strengths"]])

  if result.get("weaknesses"):
    lines.append("不足：")
    lines.extend([f"- {item}" for item in result["weaknesses"]])

  if result.get("suggestions"):
    lines.append("建议：")
    lines.extend([f"- {item}" for item in result["suggestions"]])

  if result.get("followup_question"):
    lines.append(f"追问：{result['followup_question']}")

  return "\n".join(lines).strip()


@router.get("/health")
async def health_check():
  redis_ok = await ping_redis()
  vectorstores = {
    provider: os.path.isdir(path)
    for provider, path in VECTORSTORE_DIRECTORY.items()
  }
  return _success_response(
    data={
      "app": "ok",
      "redis": "ok" if redis_ok else "down",
      "vectorstores": vectorstores,
      "overall_status": "ok" if redis_ok else "degraded",
    },
    message="Service health checked",
  )


@router.get("/metrics")
async def metrics():
  payload, content_type = render_metrics()
  return Response(content=payload, media_type=content_type)


@router.get("/llm")
async def get_llm_options():
  return _success_response(data=[provider.title() for provider in MODEL_OPTIONS.keys()])


@router.get("/llm/{model_provider}")
async def get_llm_models(model_provider: str):
  normalized_provider = model_provider.lower()
  if normalized_provider not in MODEL_OPTIONS:
    return _error_response("Invalid model provider.")
  return _success_response(data=MODEL_OPTIONS[normalized_provider]["models"])


@router.post("/upload_and_process_pdfs")
async def upload_and_process_pdfs(
  files: list[UploadFile] = File(...),
  model_provider: str = Form(...),
):
  try:
    normalized_provider, _model_name = _validate_model(model_provider, None)
    await upsert_vectorstore_from_pdfs(files, normalized_provider)
    return _success_response(message="Files processed and stored successfully.")
  except Exception as exc:
    logger.exception("Error while uploading and processing files")
    return _error_response(str(exc), status_code=500)


@router.get("/vector_store/count/{model_provider}")
async def get_vectorstore_count(model_provider: str):
  try:
    normalized_provider, _model_name = _validate_model(model_provider, None)
    count = await get_collections_count(normalized_provider)
    return _success_response(data=count)
  except Exception as exc:
    logger.exception("Error getting collection count")
    return _error_response(str(exc), status_code=500)


@router.post("/vector_store/search")
async def get_vectorstore_search(request: SearchQueryRequest):
  try:
    normalized_provider, _model_name = _validate_model(request.model_provider, None)
    results_with_scores = await find_similar_chunks(normalized_provider, request.query)
    return _success_response(data=serialize_search_results(results_with_scores))
  except Exception as exc:
    logger.exception("Error during similarity search")
    return _error_response(str(exc), status_code=500)


@router.post("/chat")
async def chat(request: ChatRequest):
  try:
    normalized_provider, model_name = _validate_model(
      request.model_provider,
      request.model_name,
    )
    retrieval = await retrieve_scored_chunks(
      normalized_provider,
      request.message,
      threshold=SIMILARITY_THRESHOLD,
    )
    if not retrieval["passes_threshold"]:
      return _error_response(
        "No relevant information found. Please upload more documents.",
        headers={"X-Cache": "MISS"},
        status_code=404,
      )

    context = _build_interview_context(retrieval["results"])
    messages = build_chat_messages(context, request.message)
    result = await invoke_completion(
      use_case="chat",
      model_provider=normalized_provider,
      model_name=model_name,
      messages=messages,
    )
    response = {
      "answer": result.raw_text,
      "sources": _compact_sources(retrieval["results"]),
    }
    return _success_response(
      data=response,
      headers={"X-Cache": "HIT" if result.cache_hit else "MISS"},
    )
  except Exception as exc:
    logger.exception("Chat endpoint encountered an error")
    return _error_response(str(exc), headers={"X-Cache": "MISS"}, status_code=500)


@router.post("/chat/stream")
async def stream_chat(request: ChatRequest):
  try:
    normalized_provider, model_name = _validate_model(
      request.model_provider,
      request.model_name,
    )
    retrieval = await retrieve_scored_chunks(
      normalized_provider,
      request.message,
      threshold=SIMILARITY_THRESHOLD,
    )
    if not retrieval["passes_threshold"]:
      async def error_stream():
        yield format_sse_event(
          "error",
          {"message": "No relevant information found. Please upload more documents."},
        )

      return StreamingResponse(
        error_stream(),
        media_type="text/event-stream",
        headers=_sse_headers("MISS"),
      )

    context = _build_interview_context(retrieval["results"])
    messages = build_chat_messages(context, request.message)
    cache_lookup = await inspect_completion_cache(
      use_case="chat",
      model_provider=normalized_provider,
      model_name=model_name,
      messages=messages,
    )
    cache_status = "HIT" if cache_lookup["cached"] else "MISS"

    async def event_stream():
      try:
        async for event in stream_completion(
          use_case="chat",
          phase="answer",
          model_provider=normalized_provider,
          model_name=model_name,
          messages=messages,
          cache_lookup=cache_lookup,
        ):
          if event["event"] != "done":
            yield format_sse_event(event["event"], event["data"])
            continue

          yield format_sse_event(
            "done",
            {
              "phase": "answer",
              "payload": {
                "answer": event["data"]["raw_text"],
                "sources": _compact_sources(retrieval["results"]),
              },
              "prompt_tokens": event["data"]["prompt_tokens"],
              "completion_tokens": event["data"]["completion_tokens"],
              "cache": event["data"]["cache"],
            },
          )
      except Exception as exc:
        logger.exception("Chat stream endpoint encountered an error")
        yield format_sse_event("error", {"message": str(exc)})

    return StreamingResponse(
      event_stream(),
      media_type="text/event-stream",
      headers=_sse_headers(cache_status),
    )
  except Exception as exc:
    logger.exception("Chat stream setup failed")
    error_message = str(exc)

    async def error_stream():
      yield format_sse_event("error", {"message": error_message})

    return StreamingResponse(
      error_stream(),
      media_type="text/event-stream",
      headers=_sse_headers("MISS"),
    )


async def _prepare_interview_context(model_provider: str):
  try:
    count = await get_collections_count(model_provider)
    if count <= 0:
      raise ValueError("No resume knowledge found. Please upload resume documents first.")
  except ValueError:
    raise
  except Exception as exc:
    logger.warning("Collection count check failed, continuing with retrieval", error=str(exc))

  retrieval = await retrieve_scored_chunks(
    model_provider,
    "请总结候选人的技术经历、核心项目、技术栈和可量化结果。",
    k=4,
    threshold=SIMILARITY_THRESHOLD,
  )
  if retrieval["results"]:
    return _compact_sources(retrieval["results"]), _build_interview_context(retrieval["results"])

  fallback_results = await find_similar_chunks(model_provider, "简历", k=4)
  if not fallback_results:
    raise ValueError("No resume knowledge found. Please upload resume documents first.")

  return _compact_sources(fallback_results), _build_interview_context(fallback_results)


async def _build_start_payload(
  request: InterviewStartRequest,
  normalized_provider: str,
  model_name: str,
  generated: dict,
  compact_sources: list[dict],
  session_id: str,
):
  first_question = _normalize_question(generated.get("question"), "q1")
  if not first_question:
    raise ValueError("Failed to generate the first interview question.")

  first_turn_sources = await _retrieve_question_specific_sources(
    normalized_provider,
    first_question["question"],
    fallback_sources=compact_sources,
  )

  session_payload = {
    "session_id": session_id,
    "status": "active",
    "model_provider": normalized_provider,
    "model_name": model_name,
    "jd_text": request.jd_text,
    "rubric": generated.get("rubric", {"score_scale": "1-10"}),
    "resume_sources": compact_sources,
    "turns": [],
    "current_question_id": None,
    "report": None,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
  }
  await save_session(session_id, session_payload)

  first_turn = _build_turn_payload(first_question, first_turn_sources, 1)
  await append_turn(session_id, first_turn)
  await set_current_question(session_id, first_question["id"])

  return {
    "session_id": session_id,
    "status": "active",
    "opening_message": generated.get("opening_message", "你好，我们开始这轮技术面试。"),
    "current_question": first_question,
    "progress": {
      "asked_count": 1,
      "answered_count": 0,
    },
    "rubric": session_payload["rubric"],
  }


@router.post("/interview/start")
async def start_interview(request: InterviewStartRequest):
  try:
    normalized_provider, model_name = _validate_model(
      request.model_provider,
      request.model_name,
    )
    compact_sources, context = await _prepare_interview_context(normalized_provider)
    messages = build_initial_interview_messages(
      context,
      job_description=request.jd_text or "",
      opening_style=request.opening_style or "",
    )

    async with get_interview_semaphore():
      result = await invoke_completion(
        use_case="interview_start",
        model_provider=normalized_provider,
        model_name=model_name,
        messages=messages,
        parser=parse_initial_interview_response,
      )

    payload = await _build_start_payload(
      request,
      normalized_provider,
      model_name,
      result.parsed_payload,
      compact_sources,
      str(uuid.uuid4()),
    )
    return _success_response(
      data=payload,
      headers={"X-Cache": "HIT" if result.cache_hit else "MISS"},
    )
  except Exception as exc:
    logger.exception("Interview start endpoint encountered an error")
    return _error_response(str(exc), headers={"X-Cache": "MISS"}, status_code=500)


@router.post("/interview/start/stream")
async def stream_start_interview(request: InterviewStartRequest):
  try:
    normalized_provider, model_name = _validate_model(
      request.model_provider,
      request.model_name,
    )
    compact_sources, context = await _prepare_interview_context(normalized_provider)
    messages = build_initial_interview_messages(
      context,
      job_description=request.jd_text or "",
      opening_style=request.opening_style or "",
    )
    cache_lookup = await inspect_completion_cache(
      use_case="interview_start",
      model_provider=normalized_provider,
      model_name=model_name,
      messages=messages,
    )
    cache_status = "HIT" if cache_lookup["cached"] else "MISS"
    session_id = str(uuid.uuid4())

    async def event_stream():
      try:
        async with get_interview_semaphore():
          async for event in stream_completion(
            use_case="interview_start",
            phase="opening",
            model_provider=normalized_provider,
            model_name=model_name,
            messages=messages,
            parser=parse_initial_interview_response,
            cache_lookup=cache_lookup,
          ):
            if event["event"] != "done":
              yield format_sse_event(event["event"], event["data"])
              continue

            payload = await _build_start_payload(
              request,
              normalized_provider,
              model_name,
              event["data"]["parsed_payload"],
              compact_sources,
              session_id,
            )
            yield format_sse_event(
              "done",
              {
                "phase": "opening",
                "payload": payload,
                "prompt_tokens": event["data"]["prompt_tokens"],
                "completion_tokens": event["data"]["completion_tokens"],
                "cache": event["data"]["cache"],
              },
            )
      except Exception as exc:
        logger.exception("Interview start stream endpoint encountered an error")
        yield format_sse_event("error", {"message": str(exc)})

    return StreamingResponse(
      event_stream(),
      media_type="text/event-stream",
      headers=_sse_headers(cache_status),
    )
  except Exception as exc:
    logger.exception("Interview start stream setup failed")
    error_message = str(exc)

    async def error_stream():
      yield format_sse_event("error", {"message": error_message})

    return StreamingResponse(
      error_stream(),
      media_type="text/event-stream",
      headers=_sse_headers("MISS"),
    )


async def _prepare_answer_dependencies(request: InterviewAnswerRequest):
  session = await get_session(request.session_id)
  if not session:
    raise ValueError("Invalid session_id.")
  if session.get("status") != "active":
    raise ValueError("Interview is not active.")

  current_turn = await get_current_turn(request.session_id)
  if not current_turn:
    raise ValueError("No active question found.")
  if current_turn.get("question_id") != request.question_id:
    raise ValueError("Invalid question_id.")

  normalized_provider, model_name = _validate_model(
    request.model_provider or session.get("model_provider", "deepseek"),
    request.model_name or session.get("model_name"),
  )
  enhanced_sources = await _retrieve_question_specific_sources(
    normalized_provider,
    current_turn["question"],
    user_answer=request.user_answer,
    fallback_sources=current_turn.get("sources") or session.get("resume_sources", []),
  )
  return session, current_turn, normalized_provider, model_name, enhanced_sources


async def _invoke_next_question(
  *,
  request: InterviewAnswerRequest,
  normalized_provider: str,
  model_name: str,
  context: str,
  refreshed_session: dict,
  summary: str,
):
  next_messages = build_next_interview_question_messages(
    context,
    job_description=refreshed_session.get("jd_text", ""),
    prior_conversation=await _build_recent_interview_memory(request.session_id),
    latest_feedback_summary=summary,
  )
  return await invoke_completion(
    use_case="interview_next_question",
    model_provider=normalized_provider,
    model_name=model_name,
    messages=next_messages,
    parser=lambda raw_text: parse_next_interview_question_response(
      raw_text,
      f"q{_get_turn_count(refreshed_session) + 1}",
    ),
  )


async def _build_answer_payload(
  request: InterviewAnswerRequest,
  normalized_provider: str,
  model_name: str,
  enhanced_sources: list[dict],
  feedback: dict,
):
  response_feedback = {
    "score": feedback.get("score", 0),
    "summary": feedback.get("summary", ""),
    "strengths": feedback.get("strengths", []),
    "weaknesses": feedback.get("weaknesses", []),
    "suggestions": feedback.get("suggestions", []),
    "followup_question": feedback.get("followup_question", ""),
    "sources": enhanced_sources,
  }

  await update_session_answer(
    request.session_id,
    request.question_id,
    request.user_answer,
    response_feedback,
  )
  refreshed_session = await get_session(request.session_id)
  if not refreshed_session:
    raise ValueError("Interview session expired.")

  context = _build_context_from_compact_sources(enhanced_sources)
  next_result = await _invoke_next_question(
    request=request,
    normalized_provider=normalized_provider,
    model_name=model_name,
    context=context,
    refreshed_session=refreshed_session,
    summary=response_feedback["summary"],
  )
  next_question = _normalize_question(
    next_result.parsed_payload.get("question"),
    f"q{_get_turn_count(refreshed_session) + 1}",
  )

  if not next_question:
    return {
      **response_feedback,
      "next_question": None,
      "progress": {
        "asked_count": _get_turn_count(refreshed_session),
        "answered_count": len(await get_completed_turns(request.session_id)),
      },
      "is_finished": False,
      "status": "active",
    }, response_feedback, next_result

  next_turn_sources = await _retrieve_question_specific_sources(
    normalized_provider,
    next_question["question"],
    fallback_sources=refreshed_session.get("resume_sources", []),
  )
  next_turn = _build_turn_payload(
    next_question,
    next_turn_sources,
    _get_turn_count(refreshed_session) + 1,
  )
  await append_turn(request.session_id, next_turn)
  await set_current_question(request.session_id, next_question["id"])
  latest_session = await get_session(request.session_id) or refreshed_session

  return {
    **response_feedback,
    "next_question": next_question,
    "progress": {
      "asked_count": _get_turn_count(latest_session),
      "answered_count": len(await get_completed_turns(request.session_id)),
    },
    "is_finished": False,
    "status": "active",
  }, response_feedback, next_result


@router.post("/interview/answer")
async def answer_interview(request: InterviewAnswerRequest):
  try:
    (
      _session,
      current_turn,
      normalized_provider,
      model_name,
      enhanced_sources,
    ) = await _prepare_answer_dependencies(request)

    if not enhanced_sources:
      return _success_response(data=_fallback_feedback(), headers={"X-Cache": "MISS"})

    context = _build_context_from_compact_sources(enhanced_sources)
    feedback_messages = build_interview_feedback_messages(
      current_turn["question"],
      request.user_answer,
      context,
      prior_conversation=await _build_recent_interview_memory(request.session_id),
    )

    async with get_interview_semaphore():
      feedback_result = await invoke_completion(
        use_case="interview_feedback",
        model_provider=normalized_provider,
        model_name=model_name,
        messages=feedback_messages,
        parser=parse_interview_feedback_response,
      )
      payload, _response_feedback, next_result = await _build_answer_payload(
        request,
        normalized_provider,
        model_name,
        enhanced_sources,
        feedback_result.parsed_payload,
      )

    header_cache = "HIT" if feedback_result.cache_hit and next_result.cache_hit else "MISS"
    return _success_response(data=payload, headers={"X-Cache": header_cache})
  except Exception as exc:
    logger.exception("Interview answer endpoint encountered an error")
    return _error_response(str(exc), headers={"X-Cache": "MISS"}, status_code=500)


@router.post("/interview/answer/stream")
async def stream_answer_interview(request: InterviewAnswerRequest):
  try:
    (
      session,
      current_turn,
      normalized_provider,
      model_name,
      enhanced_sources,
    ) = await _prepare_answer_dependencies(request)

    if not enhanced_sources:
      async def empty_stream():
        yield format_sse_event(
          "done",
          {
            "phase": "answer",
            "payload": _fallback_feedback(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache": "MISS",
          },
        )

      return StreamingResponse(
        empty_stream(),
        media_type="text/event-stream",
        headers=_sse_headers("MISS"),
      )

    context = _build_context_from_compact_sources(enhanced_sources)
    feedback_messages = build_interview_feedback_messages(
      current_turn["question"],
      request.user_answer,
      context,
      prior_conversation=await _build_recent_interview_memory(request.session_id),
    )
    feedback_cache_lookup = await inspect_completion_cache(
      use_case="interview_feedback",
      model_provider=normalized_provider,
      model_name=model_name,
      messages=feedback_messages,
    )
    cache_status = "HIT" if feedback_cache_lookup["cached"] else "MISS"

    async def event_stream():
      try:
        async with get_interview_semaphore():
          feedback_payload = None
          response_feedback = None
          async for event in stream_completion(
            use_case="interview_feedback",
            phase="feedback",
            model_provider=normalized_provider,
            model_name=model_name,
            messages=feedback_messages,
            parser=parse_interview_feedback_response,
            cache_lookup=feedback_cache_lookup,
          ):
            if event["event"] != "done":
              yield format_sse_event(event["event"], event["data"])
              continue

            feedback_payload = event["data"]["parsed_payload"]
            response_feedback = {
              "score": feedback_payload.get("score", 0),
              "summary": feedback_payload.get("summary", ""),
              "strengths": feedback_payload.get("strengths", []),
              "weaknesses": feedback_payload.get("weaknesses", []),
              "suggestions": feedback_payload.get("suggestions", []),
              "followup_question": feedback_payload.get("followup_question", ""),
              "sources": enhanced_sources,
            }
            yield format_sse_event(
              "meta",
              {
                "phase": "feedback",
                "status": "complete",
                "payload": response_feedback,
                "rendered_text": _feedback_as_text(response_feedback),
              },
            )

          await update_session_answer(
            request.session_id,
            request.question_id,
            request.user_answer,
            response_feedback,
          )
          refreshed_session = await get_session(request.session_id)
          if not refreshed_session:
            raise ValueError("Interview session expired.")

          next_messages = build_next_interview_question_messages(
            context,
            job_description=refreshed_session.get("jd_text", ""),
            prior_conversation=await _build_recent_interview_memory(request.session_id),
            latest_feedback_summary=response_feedback["summary"],
          )
          next_cache_lookup = await inspect_completion_cache(
            use_case="interview_next_question",
            model_provider=normalized_provider,
            model_name=model_name,
            messages=next_messages,
          )
          next_question_fallback_id = f"q{_get_turn_count(refreshed_session) + 1}"

          async for event in stream_completion(
            use_case="interview_next_question",
            phase="next_question",
            model_provider=normalized_provider,
            model_name=model_name,
            messages=next_messages,
            parser=lambda raw_text: parse_next_interview_question_response(
              raw_text,
              next_question_fallback_id,
            ),
            cache_lookup=next_cache_lookup,
          ):
            if event["event"] != "done":
              yield format_sse_event(event["event"], event["data"])
              continue

            next_question = _normalize_question(
              event["data"]["parsed_payload"].get("question"),
              next_question_fallback_id,
            )
            if next_question:
              next_turn_sources = await _retrieve_question_specific_sources(
                normalized_provider,
                next_question["question"],
                fallback_sources=refreshed_session.get("resume_sources", []),
              )
              next_turn = _build_turn_payload(
                next_question,
                next_turn_sources,
                _get_turn_count(refreshed_session) + 1,
              )
              await append_turn(request.session_id, next_turn)
              await set_current_question(request.session_id, next_question["id"])

            latest_session = await get_session(request.session_id) or refreshed_session
            final_payload = {
              **response_feedback,
              "next_question": next_question,
              "progress": {
                "asked_count": _get_turn_count(latest_session),
                "answered_count": len(await get_completed_turns(request.session_id)),
              },
              "is_finished": False,
              "status": "active",
            }
            yield format_sse_event(
              "done",
              {
                "phase": "answer",
                "payload": final_payload,
                "prompt_tokens": (
                  feedback_cache_lookup["prompt_tokens"] + next_cache_lookup["prompt_tokens"]
                ),
                "completion_tokens": event["data"]["completion_tokens"],
                "cache": "HIT" if cache_status == "HIT" and event["data"]["cache"] == "HIT" else "MISS",
              },
            )
      except Exception as exc:
        logger.exception("Interview answer stream endpoint encountered an error")
        yield format_sse_event("error", {"message": str(exc)})

    return StreamingResponse(
      event_stream(),
      media_type="text/event-stream",
      headers=_sse_headers(cache_status),
    )
  except Exception as exc:
    logger.exception("Interview answer stream setup failed")
    error_message = str(exc)

    async def error_stream():
      yield format_sse_event("error", {"message": error_message})

    return StreamingResponse(
      error_stream(),
      media_type="text/event-stream",
      headers=_sse_headers("MISS"),
    )


@router.post("/interview/end")
async def end_interview(request: InterviewEndRequest):
  try:
    session = await get_session(request.session_id)
    if not session:
      return _error_response("Invalid session_id.", status_code=404)

    if session.get("status") == "ended":
      return _success_response(
        data={
          "session_id": request.session_id,
          "status": "ended",
        },
        message="Interview ended by user.",
      )

    await mark_session_status(request.session_id, "ended")
    await set_current_question(request.session_id, None)
    return _success_response(
      data={
        "session_id": request.session_id,
        "status": "ended",
      },
      message="Interview ended by user.",
    )
  except Exception as exc:
    logger.exception("Interview end endpoint encountered an error")
    return _error_response(str(exc), status_code=500)


@router.get("/interview/report/{session_id}")
async def get_interview_report(
  session_id: str,
  report_format: str = Query("json"),
):
  try:
    session = await get_session(session_id)
    if not session:
      return _error_response("Invalid session_id.", status_code=404)
    if session.get("status") != "ended":
      return _error_response("请先结束本轮面试，再生成报告。", status_code=409)

    report = session.get("report") or _build_report(session)
    await update_session_report(session_id, report)

    if report_format == "markdown":
      return _success_response(
        data={
          "session_id": session_id,
          "markdown": _build_report_markdown(report),
        },
      )

    return _success_response(data=report)
  except Exception as exc:
    logger.exception("Interview report endpoint encountered an error")
    return _error_response(str(exc), status_code=500)
