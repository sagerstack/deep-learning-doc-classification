import json
import logging
import sys
import urllib.request

import structlog
from structlog.contextvars import merge_contextvars
from structlog.processors import JSONRenderer, StackInfoRenderer, TimeStamper, UnicodeDecoder
from structlog.stdlib import ProcessorFormatter, add_log_level, add_logger_name

from app.src.config import ENVIRONMENT, LOG_LEVEL, SEQ_API_KEY, SEQ_SERVER_URL

# Fields consumed by structlog internals — not useful as Seq properties
_CLEF_RESERVED = frozenset({"timestamp", "level", "logger", "_record", "exc_info", "stack_info"})


class _SeqCLEFProcessor:
    """Structlog processor that POSTs events directly to Seq as CLEF.

    Runs before JSONRenderer so the event dict still has all structured fields.
    Every field becomes a top-level CLEF property — no double-encoding.
    """

    def __init__(self, server_url: str, api_key: str = "") -> None:
        self._url = server_url.rstrip("/") + "/api/events/raw?clef"
        self._headers: dict[str, str] = {"Content-Type": "application/vnd.serilog.clef"}
        if api_key:
            self._headers["X-Seq-ApiKey"] = api_key

    def __call__(self, logger, method: str, event_dict: dict) -> dict:
        try:
            # Map structlog fields to CLEF envelope
            clef: dict = {
                "@t": event_dict.get("timestamp", ""),
                "@mt": event_dict.get("event", ""),
                "@l": (event_dict.get("level") or "info").upper(),
            }
            # All other fields become top-level CLEF properties (structured)
            for k, v in event_dict.items():
                if k not in _CLEF_RESERVED and k != "event":
                    clef[k] = v

            data = json.dumps(clef, default=str).encode()
            req = urllib.request.Request(
                self._url, data=data, headers=self._headers, method="POST"
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass  # non-fatal — stdout pipeline is the primary sink
        return event_dict  # always pass through to next processor


def configure_logging() -> None:
    shared_processors = [
        merge_contextvars,
        add_log_level,
        add_logger_name,
        TimeStamper(fmt="iso", utc=True),
        StackInfoRenderer(),
        UnicodeDecoder(),
    ]

    # Direct CLEF processor — injected before JSONRenderer if Seq is configured
    seq_processors = []
    if SEQ_SERVER_URL:
        try:
            seq_processors = [_SeqCLEFProcessor(SEQ_SERVER_URL, SEQ_API_KEY or "")]
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to set up Seq CLEF processor (non-fatal): %s", exc
            )

    structlog.configure(
        processors=shared_processors + seq_processors + [ProcessorFormatter.wrap_for_formatter],
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

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True

    structlog.contextvars.bind_contextvars(environment=ENVIRONMENT)
