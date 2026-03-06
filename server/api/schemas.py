from typing import Any, Literal, Optional

from pydantic import BaseModel


class SearchQueryRequest(BaseModel):
  model_provider: str
  query: str


class ChatRequest(BaseModel):
  model_provider: str
  model_name: Optional[str] = None
  message: str


class InterviewStartRequest(BaseModel):
  model_provider: str
  model_name: Optional[str] = None
  question_count: Optional[int] = None
  jd_text: Optional[str] = None
  opening_style: Optional[str] = None


class InterviewAnswerRequest(BaseModel):
  model_provider: str
  model_name: Optional[str] = None
  session_id: str
  question_id: str
  user_answer: str


class InterviewEndRequest(BaseModel):
  session_id: str


class StandardAPIResponse(BaseModel):
  status: Literal["success", "error"]
  data: Optional[Any] = None
  message: Optional[str] = None
