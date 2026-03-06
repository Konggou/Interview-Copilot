from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, File, Form, Query, UploadFile

from api.schemas import (
  ChatRequest,
  InterviewAnswerRequest,
  InterviewEndRequest,
  InterviewStartRequest,
  SearchQueryRequest,
  StandardAPIResponse,
)
from config.settings import (
  INTERVIEW_CONTEXT_SNIPPET_LENGTH,
  MODEL_OPTIONS,
  SIMILARITY_THRESHOLD,
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
  build_llm_chain,
  evaluate_interview_answer,
  generate_initial_interview_turn,
  generate_next_interview_question,
  get_default_model,
)
from core.vector_database import (
  find_similar_chunks,
  get_collections_count,
  load_vectorstore,
  retrieve_scored_chunks,
  serialize_search_results,
  upsert_vectorstore_from_pdfs,
)
from utils.logger import logger


router = APIRouter()


def _validate_model(model_provider: str, model_name: str | None):
  normalized_provider = model_provider.lower()
  if normalized_provider not in MODEL_OPTIONS:
    logger.warning(f"Invalid model provider: {normalized_provider}")
    raise ValueError("Invalid model provider.")

  resolved_model = model_name or get_default_model(normalized_provider)
  if resolved_model not in MODEL_OPTIONS[normalized_provider]["models"]:
    logger.warning(f"Invalid model name: {resolved_model}")
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
        f"\u6765\u6e90: {item['source']}\n"
        f"\u9875\u7801: {item['page']}\n"
        f"\u7247\u6bb5: {item['snippet']}"
      )
    )
  return "\n\n".join(context_blocks)


def _build_context_from_compact_sources(compact_sources: list[dict]) -> str:
  context_blocks = []
  for item in compact_sources:
    context_blocks.append(
      (
        f"\u6765\u6e90: {item.get('source', 'Unknown')}\n"
        f"\u9875\u7801: {item.get('page', 0)}\n"
        f"\u7247\u6bb5: {item.get('snippet', '')}"
      )
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
    "focus": str(raw_question.get("focus", "\u7b80\u5386\u5339\u914d\u5ea6")).strip() or "\u7b80\u5386\u5339\u914d\u5ea6",
    "difficulty": str(raw_question.get("difficulty", "mid")).strip() or "mid",
  }


def _build_recent_interview_memory(session_id: str, limit: int = 3) -> str:
  completed_turns = get_completed_turns(session_id)[-limit:]
  if not completed_turns:
    return ""

  blocks = []
  for index, turn in enumerate(completed_turns, start=1):
    feedback = turn.get("feedback", {})
    blocks.append(
      (
        f"\u7b2c {index} \u8f6e\n"
        f"\u95ee\u9898: {turn.get('question', '')}\n"
        f"\u56de\u7b54: {turn.get('user_answer', '')}\n"
        f"\u53cd\u9988\u6458\u8981: {feedback.get('summary', '')}\n"
        f"\u5206\u6570: {feedback.get('score', '')}"
      )
    )
  return "\n\n".join(blocks)


def _fallback_feedback():
  return {
    "score": 0,
    "summary": "\u65e0\u6cd5\u4ece\u7b80\u5386\u5224\u65ad",
    "strengths": [],
    "weaknesses": ["\u7b80\u5386\u4e2d\u7f3a\u5c11\u4e0e\u8be5\u95ee\u9898\u76f4\u63a5\u76f8\u5173\u7684\u8bc1\u636e\u3002"],
    "suggestions": ["\u8865\u5145\u66f4\u5177\u4f53\u7684\u9879\u76ee\u7ecf\u5386\u3001\u6280\u672f\u5b9e\u73b0\u548c\u91cf\u5316\u7ed3\u679c\u3002"],
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
    "summary": (
      f"\u672c\u8f6e\u9762\u8bd5\u5171\u5b8c\u6210 {len(answered_turns)} \u8f6e\uff0c\u5e73\u5747\u5206 {average_score}/10\u3002"
    ),
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
    "# \u9762\u8bd5\u62a5\u544a",
    "",
    f"- \u4f1a\u8bdd ID: {report['session_id']}",
    f"- \u751f\u6210\u65f6\u95f4: {report['generated_at']}",
    f"- \u5b8c\u6210\u8f6e\u6b21: {report['answered_count']}",
    f"- \u5e73\u5747\u5206: {report['average_score']}/10",
    "",
    "## \u603b\u7ed3",
    report["summary"],
    "",
    "## \u4e3b\u8981\u4f18\u70b9",
  ]

  if report["strengths"]:
    lines.extend([f"- {item}" for item in report["strengths"]])
  else:
    lines.append("- \u6682\u65e0\u8bb0\u5f55\u3002")

  lines.extend(["", "## \u4e3b\u8981\u4e0d\u8db3"])
  if report["weaknesses"]:
    lines.extend([f"- {item}" for item in report["weaknesses"]])
  else:
    lines.append("- \u6682\u65e0\u8bb0\u5f55\u3002")

  lines.extend(["", "## \u6539\u8fdb\u5efa\u8bae"])
  if report["suggestions"]:
    lines.extend([f"- {item}" for item in report["suggestions"]])
  else:
    lines.append("- \u6682\u65e0\u8bb0\u5f55\u3002")

  lines.extend(["", "## \u5206\u8f6e\u8be6\u60c5"])
  for index, detail in enumerate(report["details"], start=1):
    focus_text = detail["focus"] or "\u6682\u65e0"
    difficulty_text = detail["difficulty"] or "\u6682\u65e0"
    answer_text = detail["answer"] or "\u672a\u4f5c\u7b54"
    summary_text = detail["feedback"].get("summary", "\u6682\u65e0")
    score_text = detail["feedback"].get("score", "\u6682\u65e0")
    lines.extend([
      f"### \u7b2c {index} \u8f6e",
      f"- \u95ee\u9898: {detail['question']}",
      f"- \u8003\u5bdf\u70b9: {focus_text}",
      f"- \u96be\u5ea6: {difficulty_text}",
      f"- \u56de\u7b54: {answer_text}",
      f"- \u603b\u7ed3: {summary_text}",
      f"- \u5206\u6570: {score_text}",
      "",
    ])

  return "\n".join(lines).strip()


def _retrieve_question_specific_sources(
  model_provider: str,
  question_text: str,
  user_answer: str = "",
  fallback_sources: list[dict] | None = None,
):
  query = question_text.strip()
  if user_answer.strip():
    query = f"{question_text}\n\u5019\u9009\u4eba\u56de\u7b54\uff1a{user_answer}"

  try:
    retrieval = retrieve_scored_chunks(
      model_provider,
      query,
      k=3,
      threshold=SIMILARITY_THRESHOLD,
    )
    if retrieval["results"]:
      return _compact_sources(retrieval["results"])
  except Exception as e:
    logger.warning(f"Question-specific retrieval failed, falling back to stored sources: {e}")

  return list(fallback_sources or [])


@router.get("/health", response_model=StandardAPIResponse)
async def health_check():
  return StandardAPIResponse(
    status="success",
    data="ok",
    message="Service is healthy",
  )


@router.get("/llm", response_model=StandardAPIResponse)
async def get_llm_options():
  logger.debug("Fetching LLM providers.")
  return StandardAPIResponse(
    status="success",
    data=[provider.title() for provider in MODEL_OPTIONS.keys()],
  )


@router.get("/llm/{model_provider}", response_model=StandardAPIResponse)
async def get_llm_models(model_provider: str):
  normalized_provider = model_provider.lower()
  if normalized_provider not in MODEL_OPTIONS:
    return StandardAPIResponse(status="error", message="Invalid model provider.")

  logger.debug(f"Fetching models for provider: {normalized_provider}")
  return StandardAPIResponse(
    status="success",
    data=MODEL_OPTIONS[normalized_provider]["models"],
  )


@router.post("/upload_and_process_pdfs", response_model=StandardAPIResponse)
async def upload_and_process_pdfs(
  files: list[UploadFile] = File(...),
  model_provider: str = Form(...),
):
  try:
    normalized_provider, _model_name = _validate_model(model_provider, None)
    await upsert_vectorstore_from_pdfs(files, normalized_provider)
    return StandardAPIResponse(
      status="success",
      message="Files processed and stored successfully.",
    )
  except Exception as e:
    logger.exception("Error while uploading and processing files")
    return StandardAPIResponse(status="error", message=str(e))


@router.get("/vector_store/count/{model_provider}", response_model=StandardAPIResponse)
async def get_vectorstore_count(model_provider: str):
  try:
    normalized_provider, _model_name = _validate_model(model_provider, None)
    count = get_collections_count(normalized_provider)
    return StandardAPIResponse(status="success", data=count)
  except Exception as e:
    logger.exception("Error getting collection count")
    return StandardAPIResponse(status="error", message=str(e))


@router.post("/vector_store/search", response_model=StandardAPIResponse)
async def get_vectorstore_search(request: SearchQueryRequest):
  try:
    normalized_provider, _model_name = _validate_model(request.model_provider, None)
    logger.info(f"Search requested with query: {request.query} for provider: {normalized_provider}")
    results_with_scores = find_similar_chunks(normalized_provider, request.query)
    return StandardAPIResponse(
      status="success",
      data=serialize_search_results(results_with_scores),
    )
  except Exception as e:
    logger.exception("Error during similarity search")
    return StandardAPIResponse(status="error", message=str(e))


@router.post("/chat", response_model=StandardAPIResponse)
async def chat(request: ChatRequest):
  try:
    normalized_provider, model_name = _validate_model(
      request.model_provider,
      request.model_name,
    )
    logger.debug(f"Chat request for model: {model_name} (provider: {normalized_provider})")

    retrieval = retrieve_scored_chunks(
      normalized_provider,
      request.message,
      threshold=SIMILARITY_THRESHOLD,
    )
    if not retrieval["passes_threshold"]:
      return StandardAPIResponse(
        status="error",
        message="No relevant information found. Please upload more documents.",
      )

    vectorstore = load_vectorstore(normalized_provider)
    chain = build_llm_chain(normalized_provider, model_name, vectorstore)
    if not chain:
      return StandardAPIResponse(status="error", message="Failed to create LLM chain.")

    answer = chain.invoke({"input": request.message})["answer"]
    response = {
      "answer": answer,
      "sources": _compact_sources(retrieval["results"]),
    }
    return StandardAPIResponse(status="success", data=response)
  except Exception as e:
    logger.exception("Chat endpoint encountered an error")
    return StandardAPIResponse(status="error", message=str(e))


@router.post("/interview/start", response_model=StandardAPIResponse)
async def start_interview(request: InterviewStartRequest):
  try:
    normalized_provider, model_name = _validate_model(
      request.model_provider,
      request.model_name,
    )

    try:
      count = get_collections_count(normalized_provider)
      if count <= 0:
        return StandardAPIResponse(
          status="error",
          message="No resume knowledge found. Please upload resume documents first.",
        )
    except Exception as e:
      logger.warning(f"Collection count check failed, continuing with retrieval: {e}")

    retrieval = retrieve_scored_chunks(
      normalized_provider,
      "\u8bf7\u603b\u7ed3\u5019\u9009\u4eba\u7684\u6280\u672f\u7ecf\u5386\u3001\u6838\u5fc3\u9879\u76ee\u3001\u6280\u672f\u6808\u548c\u53ef\u91cf\u5316\u7ed3\u679c\u3002",
      k=4,
      threshold=SIMILARITY_THRESHOLD,
    )
    if retrieval["results"]:
      compact_sources = _compact_sources(retrieval["results"])
      context = _build_interview_context(retrieval["results"])
    else:
      fallback_results = find_similar_chunks(normalized_provider, "\u7b80\u5386", k=4)
      if not fallback_results:
        return StandardAPIResponse(
          status="error",
          message="No resume knowledge found. Please upload resume documents first.",
        )
      compact_sources = _compact_sources(fallback_results)
      context = _build_interview_context(fallback_results)

    generated = generate_initial_interview_turn(
      normalized_provider,
      model_name,
      context,
      job_description=request.jd_text,
      opening_style=request.opening_style or "",
    )

    first_question = _normalize_question(generated.get("question"), "q1")
    if not first_question:
      return StandardAPIResponse(
        status="error",
        message="Failed to generate the first interview question.",
      )

    first_turn_sources = _retrieve_question_specific_sources(
      normalized_provider,
      first_question["question"],
      fallback_sources=compact_sources,
    )

    session_id = str(uuid.uuid4())
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
    save_session(session_id, session_payload)

    first_turn = _build_turn_payload(first_question, first_turn_sources, 1)
    append_turn(session_id, first_turn)
    set_current_question(session_id, first_question["id"])

    return StandardAPIResponse(
      status="success",
      data={
        "session_id": session_id,
        "status": "active",
        "opening_message": generated.get(
          "opening_message",
          "\u4f60\u597d\uff0c\u6211\u4eec\u5f00\u59cb\u8fd9\u8f6e\u6280\u672f\u9762\u8bd5\u3002",
        ),
        "current_question": first_question,
        "progress": {
          "asked_count": 1,
          "answered_count": 0,
        },
        "rubric": session_payload["rubric"],
      },
    )
  except Exception as e:
    logger.exception("Interview start endpoint encountered an error")
    return StandardAPIResponse(status="error", message=str(e))


@router.post("/interview/answer", response_model=StandardAPIResponse)
async def answer_interview(request: InterviewAnswerRequest):
  try:
    session = get_session(request.session_id)
    if not session:
      return StandardAPIResponse(status="error", message="Invalid session_id.")
    if session.get("status") != "active":
      return StandardAPIResponse(status="error", message="Interview is not active.")

    current_turn = get_current_turn(request.session_id)
    if not current_turn:
      return StandardAPIResponse(status="error", message="No active question found.")
    if current_turn.get("question_id") != request.question_id:
      return StandardAPIResponse(status="error", message="Invalid question_id.")

    normalized_provider, model_name = _validate_model(
      request.model_provider or session.get("model_provider", "deepseek"),
      request.model_name or session.get("model_name"),
    )

    enhanced_sources = _retrieve_question_specific_sources(
      normalized_provider,
      current_turn["question"],
      user_answer=request.user_answer,
      fallback_sources=current_turn.get("sources") or session.get("resume_sources", []),
    )
    if not enhanced_sources:
      return StandardAPIResponse(status="success", data=_fallback_feedback())

    context = _build_context_from_compact_sources(enhanced_sources)
    prior_conversation = _build_recent_interview_memory(request.session_id)
    feedback = evaluate_interview_answer(
      normalized_provider,
      model_name,
      current_turn["question"],
      request.user_answer,
      context,
      prior_conversation=prior_conversation,
    )

    response_feedback = {
      "score": feedback.get("score", 0),
      "summary": feedback.get("summary", ""),
      "strengths": feedback.get("strengths", []),
      "weaknesses": feedback.get("weaknesses", []),
      "suggestions": feedback.get("suggestions", []),
      "followup_question": feedback.get("followup_question", ""),
      "sources": enhanced_sources,
    }
    update_session_answer(
      request.session_id,
      request.question_id,
      request.user_answer,
      response_feedback,
    )

    refreshed_session = get_session(request.session_id)
    if not refreshed_session:
      return StandardAPIResponse(status="error", message="Interview session expired.")

    next_question_payload = generate_next_interview_question(
      normalized_provider,
      model_name,
      context,
      job_description=refreshed_session.get("jd_text", ""),
      prior_conversation=_build_recent_interview_memory(request.session_id),
      latest_feedback_summary=response_feedback["summary"],
    )
    next_question = _normalize_question(
      next_question_payload.get("question"),
      f"q{_get_turn_count(refreshed_session) + 1}",
    )

    if not next_question:
      return StandardAPIResponse(
        status="success",
        data={
          **response_feedback,
          "next_question": None,
          "progress": {
            "asked_count": _get_turn_count(refreshed_session),
            "answered_count": len(get_completed_turns(request.session_id)),
          },
          "is_finished": False,
          "status": "active",
        },
        message="\u5f53\u524d\u56de\u7b54\u5df2\u8bb0\u5f55\uff0c\u4f46\u65e0\u6cd5\u751f\u6210\u4e0b\u4e00\u9898\uff0c\u8bf7\u624b\u52a8\u7ed3\u675f\u672c\u8f6e\u9762\u8bd5\u540e\u67e5\u770b\u62a5\u544a\u3002",
      )

    next_turn_sources = _retrieve_question_specific_sources(
      normalized_provider,
      next_question["question"],
      fallback_sources=refreshed_session.get("resume_sources", []),
    )
    next_turn = _build_turn_payload(
      next_question,
      next_turn_sources,
      _get_turn_count(refreshed_session) + 1,
    )
    append_turn(request.session_id, next_turn)
    set_current_question(request.session_id, next_question["id"])

    latest_session = get_session(request.session_id) or refreshed_session
    return StandardAPIResponse(
      status="success",
      data={
        **response_feedback,
        "next_question": next_question,
        "progress": {
          "asked_count": _get_turn_count(latest_session),
          "answered_count": len(get_completed_turns(request.session_id)),
        },
        "is_finished": False,
        "status": "active",
      },
    )
  except Exception as e:
    logger.exception("Interview answer endpoint encountered an error")
    return StandardAPIResponse(status="error", message=str(e))


@router.post("/interview/end", response_model=StandardAPIResponse)
async def end_interview(request: InterviewEndRequest):
  try:
    session = get_session(request.session_id)
    if not session:
      return StandardAPIResponse(status="error", message="Invalid session_id.")

    if session.get("status") == "ended":
      return StandardAPIResponse(
        status="success",
        data={
          "session_id": request.session_id,
          "status": "ended",
        },
        message="Interview ended by user.",
      )

    mark_session_status(request.session_id, "ended")
    set_current_question(request.session_id, None)
    return StandardAPIResponse(
      status="success",
      data={
        "session_id": request.session_id,
        "status": "ended",
      },
      message="Interview ended by user.",
    )
  except Exception as e:
    logger.exception("Interview end endpoint encountered an error")
    return StandardAPIResponse(status="error", message=str(e))


@router.get("/interview/report/{session_id}", response_model=StandardAPIResponse)
async def get_interview_report(
  session_id: str,
  report_format: str = Query("json"),
):
  try:
    session = get_session(session_id)
    if not session:
      return StandardAPIResponse(status="error", message="Invalid session_id.")
    if session.get("status") != "ended":
      return StandardAPIResponse(
        status="error",
        message="\u8bf7\u5148\u7ed3\u675f\u672c\u8f6e\u9762\u8bd5\uff0c\u518d\u751f\u6210\u62a5\u544a\u3002",
      )

    report = session.get("report") or _build_report(session)
    update_session_report(session_id, report)

    if report_format == "markdown":
      return StandardAPIResponse(
        status="success",
        data={
          "session_id": session_id,
          "markdown": _build_report_markdown(report),
        },
      )

    return StandardAPIResponse(status="success", data=report)
  except Exception as e:
    logger.exception("Interview report endpoint encountered an error")
    return StandardAPIResponse(status="error", message=str(e))
