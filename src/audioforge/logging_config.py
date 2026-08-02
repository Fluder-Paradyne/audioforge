"""Structured logging setup for CLI, API, and pipeline stages.

Console format is controlled by :class:`~audioforge.settings.AppSettings`
(``log_level`` / ``log_format``). Per-job file logs append to
``work/<job-id>/job.log`` via :func:`attach_job_file_handler`.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal

LogFormatName = Literal["text", "json"]

# Extra fields pipeline code may attach via ``logger.info(..., extra={...})``.
_STRUCTURED_KEYS: tuple[str, ...] = (
    "job_id",
    "stage",
    "chapter_index",
    "chapter_slug",
    "chars",
    "event",
    "voice",
    "chapter_total",
)

_ROOT_LOGGER_NAME = "audioforge"
_CONFIGURED_FLAG = "_audioforge_configured"


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record (JSON Lines)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _STRUCTURED_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable line with optional structured key=value suffixes."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        base = (
            f"{timestamp} {record.levelname:5s} [{record.name}] {record.getMessage()}"
        )
        extras: list[str] = []
        for key in _STRUCTURED_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                extras.append(f"{key}={value}")
        if extras:
            base = f"{base} | {' '.join(extras)}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def make_formatter(fmt: LogFormatName | str) -> logging.Formatter:
    """Return a formatter for ``text`` or ``json`` (unknown → text)."""
    if str(fmt).lower() == "json":
        return JsonFormatter()
    return TextFormatter()


def parse_log_level(level: str | int) -> int:
    """Resolve a level name or int; invalid names fall back to INFO."""
    if isinstance(level, int):
        return level
    mapping = logging.getLevelNamesMapping()
    return mapping.get(level.strip().upper(), logging.INFO)


def configure_logging(
    *,
    level: str | int = "INFO",
    fmt: LogFormatName | str = "text",
    stream: IO[str] | None = None,
    force: bool = False,
) -> None:
    """Configure the ``audioforge`` logger hierarchy for console output.

    Idempotent unless *force* is true. Does not touch the root logger so
    third-party libraries (uvicorn, httpx) keep their own handlers.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if getattr(logger, _CONFIGURED_FLAG, False) and not force:
        logger.setLevel(parse_log_level(level))
        return

    logger.handlers.clear()
    logger.setLevel(parse_log_level(level))
    logger.propagate = False

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(make_formatter(fmt))
    logger.addHandler(handler)
    setattr(logger, _CONFIGURED_FLAG, True)


def attach_job_file_handler(
    job_log: Path,
    *,
    fmt: LogFormatName | str = "text",
    level: str | int = "DEBUG",
) -> logging.Handler:
    """Append pipeline logs for one job to *job_log*; return the handler.

    Call :func:`detach_handler` when the job finishes so handlers do not
    accumulate across runs in long-lived processes (API server).

    Ensures the ``audioforge`` logger level is low enough that INFO/DEBUG
    records are not filtered by the root logger's default WARNING threshold.
    """
    job_log.parent.mkdir(parents=True, exist_ok=True)
    resolved = parse_log_level(level)
    package_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if package_logger.level == logging.NOTSET or resolved < package_logger.level:
        package_logger.setLevel(resolved)
    # Keep records on the package logger (and its file/stream handlers).
    package_logger.propagate = False

    handler = logging.FileHandler(job_log, mode="a", encoding="utf-8")
    handler.setLevel(resolved)
    handler.setFormatter(make_formatter(fmt))
    package_logger.addHandler(handler)
    return handler


def detach_handler(handler: logging.Handler) -> None:
    """Remove *handler* from the audioforge logger and close it."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.removeHandler(handler)
    handler.close()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return an ``audioforge`` child logger (or the package root)."""
    if name is None:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
