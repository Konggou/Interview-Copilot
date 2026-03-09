from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


LLM_REQUESTS_TOTAL = Counter(
  "ragbot_llm_requests_total",
  "Total number of LLM calls grouped by use case and cache outcome.",
  ["use_case", "cache_status"],
)

LLM_PROMPT_TOKENS_TOTAL = Counter(
  "ragbot_llm_prompt_tokens_total",
  "Estimated prompt tokens sent to the LLM.",
  ["use_case"],
)

LLM_COMPLETION_TOKENS_TOTAL = Counter(
  "ragbot_llm_completion_tokens_total",
  "Estimated completion tokens returned by the LLM.",
  ["use_case"],
)

LLM_LATENCY_SECONDS = Histogram(
  "ragbot_llm_latency_seconds",
  "Observed LLM latency in seconds.",
  ["use_case", "cache_status"],
)


def observe_llm_call(
  use_case: str,
  cache_hit: bool,
  prompt_tokens: int,
  completion_tokens: int,
  latency_seconds: float,
):
  cache_status = "HIT" if cache_hit else "MISS"
  LLM_REQUESTS_TOTAL.labels(use_case=use_case, cache_status=cache_status).inc()
  LLM_PROMPT_TOKENS_TOTAL.labels(use_case=use_case).inc(prompt_tokens)
  LLM_COMPLETION_TOKENS_TOTAL.labels(use_case=use_case).inc(completion_tokens)
  LLM_LATENCY_SECONDS.labels(
    use_case=use_case,
    cache_status=cache_status,
  ).observe(latency_seconds)


def render_metrics():
  return generate_latest(), CONTENT_TYPE_LATEST
