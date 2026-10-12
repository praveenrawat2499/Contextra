"""
Structured logging and request tracing for Contextra.

Provides:
  - Rotating file logging to logs/rag.log
  - Mirroring to stdout
  - Request / trace correlation ID generator
  - Stage-level latency tracing context manager
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Generator

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "rag.log"
_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure rotating file handler and console handler."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("Contextra")
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if reloaded
    if not root_logger.handlers:
        # Rotating File Handler (10MB per file, 5 backups)
        file_handler = RotatingFileHandler(
            filename=str(LOG_FILE),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str = "Contextra") -> logging.Logger:
    """Return a configured logger with standard prefix."""
    setup_logging()
    if not name.startswith("Contextra"):
        name = f"Contextra.{name}"
    return logging.getLogger(name)


def generate_trace_id() -> str:
    """Generate a short unique trace ID for correlating pipeline stages."""
    return uuid.uuid4().hex[:8]


@contextmanager
def trace_span(stage_name: str, trace_id: str | None = None, logger: logging.Logger | None = None) -> Generator[dict, None, None]:
    """
    Context manager to trace and log stage latency in milliseconds.
    
    Usage:
        with trace_span("dense_retrieval", trace_id) as span:
            results = store.query(...)
            span["result_count"] = len(results)
    """
    log = logger or get_logger("trace")
    tid = trace_id or generate_trace_id()
    span_data: dict = {"trace_id": tid, "stage": stage_name}

    start = time.perf_counter()
    log.info(f"[{tid}] [START] {stage_name}")
    try:
        yield span_data
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000.0
        span_data["duration_ms"] = duration_ms
        span_data["error"] = str(exc)
        log.error(f"[{tid}] [FAILED] {stage_name} after {duration_ms:.2f}ms — {type(exc).__name__}: {exc}")
        raise
    else:
        duration_ms = (time.perf_counter() - start) * 1000.0
        span_data["duration_ms"] = duration_ms
        extra_info = ""
        if "count" in span_data:
            extra_info = f" (count={span_data['count']})"
        elif "result_count" in span_data:
            extra_info = f" (results={span_data['result_count']})"
        log.info(f"[{tid}] [DONE] {stage_name} in {duration_ms:.2f}ms{extra_info}")
