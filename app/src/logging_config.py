import logging
import sys

import seqlog
import structlog
from structlog.contextvars import merge_contextvars
from structlog.processors import JSONRenderer, StackInfoRenderer, TimeStamper, UnicodeDecoder
from structlog.stdlib import ProcessorFormatter, add_log_level, add_logger_name

from app.src.config import ENVIRONMENT, LOG_LEVEL, SEQ_API_KEY, SEQ_SERVER_URL


def configure_logging() -> None:
    shared_processors = [
        merge_contextvars,
        add_log_level,
        add_logger_name,
        TimeStamper(fmt="iso", utc=True),
        StackInfoRenderer(),
        UnicodeDecoder(),
    ]

    structlog.configure(
        processors=shared_processors + [ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = ProcessorFormatter(
        processors=[ProcessorFormatter.remove_processors_meta, JSONRenderer()],
        foreign_pre_chain=shared_processors,
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    if SEQ_SERVER_URL:
        try:
            seq_handler = seqlog.SeqLogHandler(
                server_url=SEQ_SERVER_URL,
                api_key=SEQ_API_KEY or None,
                batch_size=10,
                auto_flush_timeout=2,
            )
            root_logger.addHandler(seq_handler)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to attach Seq log handler (non-fatal): %s", exc
            )

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True

    structlog.contextvars.bind_contextvars(environment=ENVIRONMENT)
