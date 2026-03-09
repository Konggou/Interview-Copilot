import logging
import sys

import structlog

from config.settings import LOG_LEVEL


def setup_logger():
  timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

  logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=getattr(logging, LOG_LEVEL, logging.INFO),
  )

  structlog.configure(
    processors=[
      structlog.contextvars.merge_contextvars,
      structlog.processors.add_log_level,
      timestamper,
      structlog.processors.StackInfoRenderer(),
      structlog.processors.format_exc_info,
      structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
      getattr(logging, LOG_LEVEL, logging.INFO),
    ),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
  )

  return structlog.get_logger("ragbot")


logger = setup_logger()
