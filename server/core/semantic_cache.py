import json
import math
import time
import uuid

from redis.asyncio import Redis

from config.settings import (
  REDIS_URL,
  SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
  SEMANTIC_CACHE_TTL_SECONDS,
)
from core.vector_database import embed_query_text
from utils.logger import logger


_redis_client: Redis | None = None


def _namespace_key(use_case: str, provider: str, model: str) -> str:
  return f"semantic-cache:{use_case}:{provider}:{model}:index"


def _entry_key(entry_id: str) -> str:
  return f"semantic-cache:entry:{entry_id}"


async def get_redis_client() -> Redis:
  global _redis_client
  if _redis_client is None:
    _redis_client = Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
  return _redis_client


async def close_redis_client():
  global _redis_client
  if _redis_client is not None:
    await _redis_client.aclose()
    _redis_client = None


async def ping() -> bool:
  try:
    client = await get_redis_client()
    return bool(await client.ping())
  except Exception:
    logger.exception("Redis ping failed")
    return False


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
  if not vector_a or not vector_b:
    return 0.0

  dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
  norm_a = math.sqrt(sum(a * a for a in vector_a))
  norm_b = math.sqrt(sum(b * b for b in vector_b))
  if norm_a == 0 or norm_b == 0:
    return 0.0
  return dot_product / (norm_a * norm_b)


async def find_cached_response(
  use_case: str,
  model_provider: str,
  model_name: str,
  prompt_text: str,
):
  try:
    client = await get_redis_client()
    namespace_key = _namespace_key(use_case, model_provider, model_name)
    candidate_ids = list(await client.smembers(namespace_key))
  except Exception:
    logger.warning(
      "Semantic cache unavailable during lookup",
      use_case=use_case,
      model_provider=model_provider,
      model_name=model_name,
    )
    return None

  if not candidate_ids:
    return None

  prompt_embedding = await embed_query_text(model_provider, prompt_text)
  try:
    payloads = await client.mget([_entry_key(entry_id) for entry_id in candidate_ids])
  except Exception:
    logger.warning(
      "Semantic cache unavailable while loading candidates",
      use_case=use_case,
      model_provider=model_provider,
      model_name=model_name,
    )
    return None

  best_match = None
  stale_ids = []
  for entry_id, payload in zip(candidate_ids, payloads):
    if not payload:
      stale_ids.append(entry_id)
      continue

    try:
      record = json.loads(payload)
    except json.JSONDecodeError:
      stale_ids.append(entry_id)
      continue

    similarity = _cosine_similarity(prompt_embedding, record.get("embedding", []))
    if similarity < SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
      continue
    if not best_match or similarity > best_match["similarity"]:
      best_match = {
        "entry_id": entry_id,
        "similarity": similarity,
        "record": record,
      }

  if stale_ids:
    try:
      await client.srem(namespace_key, *stale_ids)
    except Exception:
      logger.warning(
        "Semantic cache cleanup skipped",
        use_case=use_case,
        model_provider=model_provider,
        model_name=model_name,
      )

  return best_match


async def cache_response(
  use_case: str,
  model_provider: str,
  model_name: str,
  prompt_text: str,
  raw_text: str,
  parsed_payload,
  prompt_tokens: int,
  completion_tokens: int,
):
  try:
    client = await get_redis_client()
    entry_id = str(uuid.uuid4())
    namespace_key = _namespace_key(use_case, model_provider, model_name)
    record = {
      "use_case": use_case,
      "model_provider": model_provider,
      "model_name": model_name,
      "prompt_text": prompt_text,
      "embedding": await embed_query_text(model_provider, prompt_text),
      "raw_text": raw_text,
      "parsed_payload": parsed_payload,
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
      "created_at": int(time.time()),
    }
    await client.set(
      _entry_key(entry_id),
      json.dumps(record, ensure_ascii=False),
      ex=SEMANTIC_CACHE_TTL_SECONDS,
    )
    await client.sadd(namespace_key, entry_id)
    await client.expire(namespace_key, SEMANTIC_CACHE_TTL_SECONDS)
  except Exception:
    logger.warning(
      "Semantic cache unavailable during write",
      use_case=use_case,
      model_provider=model_provider,
      model_name=model_name,
    )
