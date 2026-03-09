import asyncio
import json
import time
from dataclasses import dataclass

import httpx
import tiktoken

from config.settings import (
  DEEPSEEK_API_KEY,
  DEEPSEEK_BASE_URL,
  DEEPSEEK_TIMEOUT_SECONDS,
  MAX_CONCURRENT_INTERVIEWS,
)
from core.metrics import observe_llm_call
from core.semantic_cache import cache_response, find_cached_response
from utils.logger import logger


LLM_RETRY_DELAY_SECONDS = 1
LLM_MAX_RETRIES = 3

_http_client: httpx.AsyncClient | None = None
_token_encoder = tiktoken.get_encoding("cl100k_base")
_interview_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INTERVIEWS)


@dataclass
class LLMCallResult:
  raw_text: str
  parsed_payload: object
  prompt_tokens: int
  completion_tokens: int
  cache_hit: bool
  latency_seconds: float


async def get_http_client() -> httpx.AsyncClient:
  global _http_client
  if _http_client is None:
    _http_client = httpx.AsyncClient(
      base_url=DEEPSEEK_BASE_URL,
      timeout=httpx.Timeout(DEEPSEEK_TIMEOUT_SECONDS),
    )
  return _http_client


async def close_http_client():
  global _http_client
  if _http_client is not None:
    await _http_client.aclose()
    _http_client = None


def get_interview_semaphore() -> asyncio.Semaphore:
  return _interview_semaphore


def _estimate_tokens_from_messages(messages: list[dict]) -> int:
  token_total = 0
  for message in messages:
    token_total += len(_token_encoder.encode(message.get("content", ""))) + 4
  return token_total + 2


def _estimate_tokens_from_text(text: str) -> int:
  return len(_token_encoder.encode(text or ""))


def _normalize_prompt_text(messages: list[dict]) -> str:
  blocks = []
  for message in messages:
    blocks.append(f"{message.get('role', 'user')}:\n{message.get('content', '')}")
  return "\n\n".join(blocks)


def _replay_chunks(text: str, chunk_size: int = 12):
  for index in range(0, len(text), chunk_size):
    yield text[index : index + chunk_size]


async def _post_completion(payload: dict):
  if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is not configured.")

  client = await get_http_client()
  headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
  }

  last_error = None
  for attempt in range(1, LLM_MAX_RETRIES + 1):
    try:
      response = await client.post("/chat/completions", headers=headers, json=payload)
      response.raise_for_status()
      return response
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
      last_error = exc
      logger.warning(
        "DeepSeek request failed",
        attempt=attempt,
        max_retries=LLM_MAX_RETRIES,
        error=str(exc),
      )
      if attempt < LLM_MAX_RETRIES:
        await asyncio.sleep(LLM_RETRY_DELAY_SECONDS)

  raise last_error


async def invoke_completion(
  *,
  use_case: str,
  model_provider: str,
  model_name: str,
  messages: list[dict],
  parser=None,
  temperature: float = 0.2,
  cache_lookup: dict | None = None,
):
  cache_lookup = cache_lookup or await inspect_completion_cache(
    use_case=use_case,
    model_provider=model_provider,
    model_name=model_name,
    messages=messages,
  )
  prompt_text = cache_lookup["prompt_text"]
  prompt_tokens = cache_lookup["prompt_tokens"]
  cached = cache_lookup["cached"]
  if cached:
    record = cached["record"]
    latency_seconds = 0.0
    observe_llm_call(
      use_case=use_case,
      cache_hit=True,
      prompt_tokens=record.get("prompt_tokens", prompt_tokens),
      completion_tokens=record.get("completion_tokens", 0),
      latency_seconds=latency_seconds,
    )
    logger.info(
      "LLM cache hit",
      use_case=use_case,
      similarity=round(cached["similarity"], 6),
      prompt_tokens=record.get("prompt_tokens", prompt_tokens),
      completion_tokens=record.get("completion_tokens", 0),
      latency_seconds=latency_seconds,
      cache_hit=True,
    )
    return LLMCallResult(
      raw_text=record.get("raw_text", ""),
      parsed_payload=record.get("parsed_payload"),
      prompt_tokens=record.get("prompt_tokens", prompt_tokens),
      completion_tokens=record.get("completion_tokens", 0),
      cache_hit=True,
      latency_seconds=latency_seconds,
    )

  started_at = time.perf_counter()
  response = await _post_completion({
    "model": model_name,
    "messages": messages,
    "temperature": temperature,
  })
  payload = response.json()
  raw_text = payload["choices"][0]["message"]["content"].strip()
  parsed_payload = parser(raw_text) if parser else raw_text
  completion_tokens = _estimate_tokens_from_text(raw_text)
  latency_seconds = time.perf_counter() - started_at

  await cache_response(
    use_case=use_case,
    model_provider=model_provider,
    model_name=model_name,
    prompt_text=prompt_text,
    raw_text=raw_text,
    parsed_payload=parsed_payload,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
  )
  observe_llm_call(
    use_case=use_case,
    cache_hit=False,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
    latency_seconds=latency_seconds,
  )
  logger.info(
    "LLM call complete",
    use_case=use_case,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
    latency_seconds=round(latency_seconds, 4),
    cache_hit=False,
  )
  return LLMCallResult(
    raw_text=raw_text,
    parsed_payload=parsed_payload,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
    cache_hit=False,
    latency_seconds=latency_seconds,
  )


async def stream_completion(
  *,
  use_case: str,
  phase: str,
  model_provider: str,
  model_name: str,
  messages: list[dict],
  parser=None,
  temperature: float = 0.2,
  cache_lookup: dict | None = None,
):
  cache_lookup = cache_lookup or await inspect_completion_cache(
    use_case=use_case,
    model_provider=model_provider,
    model_name=model_name,
    messages=messages,
  )
  prompt_text = cache_lookup["prompt_text"]
  prompt_tokens = cache_lookup["prompt_tokens"]
  cached = cache_lookup["cached"]
  if cached:
    record = cached["record"]
    latency_seconds = 0.0
    observe_llm_call(
      use_case=use_case,
      cache_hit=True,
      prompt_tokens=record.get("prompt_tokens", prompt_tokens),
      completion_tokens=record.get("completion_tokens", 0),
      latency_seconds=latency_seconds,
    )
    logger.info(
      "LLM cache hit",
      use_case=use_case,
      similarity=round(cached["similarity"], 6),
      prompt_tokens=record.get("prompt_tokens", prompt_tokens),
      completion_tokens=record.get("completion_tokens", 0),
      latency_seconds=latency_seconds,
      cache_hit=True,
    )
    yield {
      "event": "meta",
      "data": {
        "phase": phase,
        "cache": "HIT",
      },
    }
    raw_text = record.get("raw_text", "")
    for chunk in _replay_chunks(raw_text):
      yield {
        "event": "delta",
        "data": {
          "phase": phase,
          "text": chunk,
        },
      }
    yield {
      "event": "done",
      "data": {
        "phase": phase,
        "raw_text": raw_text,
        "parsed_payload": record.get("parsed_payload"),
        "prompt_tokens": record.get("prompt_tokens", prompt_tokens),
        "completion_tokens": record.get("completion_tokens", 0),
        "cache": "HIT",
      },
    }
    return

  yield {
    "event": "meta",
    "data": {
      "phase": phase,
      "cache": "MISS",
    },
  }

  started_at = time.perf_counter()
  if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is not configured.")

  client = await get_http_client()
  headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
  }
  payload = {
    "model": model_name,
    "messages": messages,
    "temperature": temperature,
    "stream": True,
  }

  raw_text = ""
  last_error = None
  for attempt in range(1, LLM_MAX_RETRIES + 1):
    try:
      async with client.stream(
        "POST",
        "/chat/completions",
        headers=headers,
        json=payload,
      ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
          if not line or not line.startswith("data: "):
            continue
          data = line[6:]
          if data == "[DONE]":
            break
          chunk_payload = json.loads(data)
          delta = chunk_payload["choices"][0].get("delta", {}).get("content", "")
          if not delta:
            continue
          raw_text += delta
          yield {
            "event": "delta",
            "data": {
              "phase": phase,
              "text": delta,
            },
          }
        break
    except (httpx.TimeoutException, httpx.HTTPError, json.JSONDecodeError) as exc:
      last_error = exc
      logger.warning(
        "DeepSeek streaming request failed",
        attempt=attempt,
        max_retries=LLM_MAX_RETRIES,
        error=str(exc),
      )
      if raw_text:
        raise last_error
      if attempt < LLM_MAX_RETRIES:
        await asyncio.sleep(LLM_RETRY_DELAY_SECONDS)
      else:
        raise last_error

  parsed_payload = parser(raw_text) if parser else raw_text
  completion_tokens = _estimate_tokens_from_text(raw_text)
  latency_seconds = time.perf_counter() - started_at
  await cache_response(
    use_case=use_case,
    model_provider=model_provider,
    model_name=model_name,
    prompt_text=prompt_text,
    raw_text=raw_text,
    parsed_payload=parsed_payload,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
  )
  observe_llm_call(
    use_case=use_case,
    cache_hit=False,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
    latency_seconds=latency_seconds,
  )
  logger.info(
    "LLM stream complete",
    use_case=use_case,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
    latency_seconds=round(latency_seconds, 4),
    cache_hit=False,
  )
  yield {
    "event": "done",
    "data": {
      "phase": phase,
      "raw_text": raw_text,
      "parsed_payload": parsed_payload,
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
      "cache": "MISS",
    },
  }


async def inspect_completion_cache(
  *,
  use_case: str,
  model_provider: str,
  model_name: str,
  messages: list[dict],
):
  prompt_text = _normalize_prompt_text(messages)
  prompt_tokens = _estimate_tokens_from_messages(messages)
  cached = await find_cached_response(use_case, model_provider, model_name, prompt_text)
  return {
    "prompt_text": prompt_text,
    "prompt_tokens": prompt_tokens,
    "cached": cached,
  }
