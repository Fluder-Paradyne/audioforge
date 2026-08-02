"""Structured logging setup for CLI, API, and pipeline stages.

Console format is controlled by :class:`~audioforge.settings.AppSettings`
(``log_level`` / ``log_format``). Per-job file logs append to
``work/<job-id>/job.log`` via :func:`attach_job_file_handler`.

Job isolation
-------------
File handlers are attached to the shared ``audioforge`` package logger but
each carries a :class:`JobIdFilter` so only records for that ``job_id`` are
written. :class:`InjectJobIdFilter` plus a :mod:`contextvars` token fill
``job_id`` on records emitted inside :func:`job_logging_context` so stage
code does not have to pass it on every call.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
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

# Operational levels accepted by settings / configure (not NOTSET/WARN aliases).
ALLOWED_LOG_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_current_job_id: ContextVar[str | None] = ContextVar(
    "audioforge_job_id",
    default=None,
)


class InjectJobIdFilter(logging.Filter):
    """Copy the active job id from context onto records that lack one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "job_id", None) is None:
            ctx = _current_job_id.get()
            if ctx is not None:
                record.job_id = ctx
        return True


class JobIdFilter(logging.Filter):
    """Accept only records whose ``job_id`` matches a single job."""

    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "job_id", None) == self.job_id


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


def normalize_log_level_name(level: str) -> str:
    """Return a canonical allowlisted level name (``DEBUG``…``CRITICAL``).

    Accepts ``WARN`` as an alias for ``WARNING``.

    Raises:
        ValueError: If *level* is not an operational log level name.
    """
    key = level.strip().upper()
    if key == "WARN":
        key = "WARNING"
    if key not in ALLOWED_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level {level!r}; use DEBUG, INFO, WARNING, ERROR, or CRITICAL"
        )
    return key


def parse_log_level(level: str | int) -> int:
    """Resolve a level name or int to a numeric logging level.

    Raises:
        ValueError: If *level* is a string that is not an allowlisted name.
    """
    if isinstance(level, int):
        return level
    return logging.getLevelNamesMapping()[normalize_log_level_name(level)]


def _is_console_handler(handler: logging.Handler) -> bool:
    """True for stream handlers that are not :class:`~logging.FileHandler`s.

    ``FileHandler`` is a ``StreamHandler`` subclass, so we must exclude it.
    """
    return isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, logging.FileHandler
    )


def _ensure_package_logger_ready() -> logging.Logger:
    """Return package logger with ``propagate=False`` and DEBUG gate.

    Handlers apply their own levels (console may be INFO while a job file
    is DEBUG). The package logger must stay at DEBUG so handler thresholds
    are not overridden by a coarser logger level.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger


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

    The package logger is always set to DEBUG; *level* controls the
    **console** handler only. Job file handlers may use a more verbose
    level (e.g. DEBUG job.log with INFO console).

    Job :class:`~logging.FileHandler` instances are preserved so a force
    reconfigure mid-run does not orphan open ``job.log`` handlers.

    Inject filters live on **handlers** (not the package logger) so child
    loggers that propagate still get ``job_id`` stamped from context.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    resolved = parse_log_level(level)
    if getattr(logger, _CONFIGURED_FLAG, False) and not force:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            if _is_console_handler(handler):
                handler.setLevel(resolved)
        return

    # Drop console handlers only; keep job FileHandlers.
    for handler in list(logger.handlers):
        if _is_console_handler(handler):
            logger.removeHandler(handler)
            handler.close()

    # Logger gate is DEBUG; console handler enforces *level*.
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console.setLevel(resolved)
    console.setFormatter(make_formatter(fmt))
    # Handler-level inject works for records from any child logger.
    console.addFilter(InjectJobIdFilter())
    logger.addHandler(console)
    setattr(logger, _CONFIGURED_FLAG, True)


def attach_job_file_handler(
    job_log: Path,
    *,
    job_id: str,
    fmt: LogFormatName | str = "text",
    level: str | int = "DEBUG",
) -> logging.Handler:
    """Append logs for *job_id* only to *job_log*; return the handler.

    Call :func:`detach_handler` when the job finishes so handlers do not
    accumulate across runs in long-lived processes (API server).

    The package logger stays at DEBUG so *level* on this handler is
    effective even when the console is configured at INFO.

    Records without matching ``job_id`` are filtered out (see
    :class:`JobIdFilter`). Prefer emitting inside :func:`job_logging_context`
    so :class:`InjectJobIdFilter` stamps the active job id automatically.
    """
    if not job_id.strip():
        raise ValueError("job_id must be non-empty")

    job_log.parent.mkdir(parents=True, exist_ok=True)
    resolved = parse_log_level(level)
    package_logger = _ensure_package_logger_ready()

    handler = logging.FileHandler(job_log, mode="a", encoding="utf-8")
    handler.setLevel(resolved)
    handler.setFormatter(make_formatter(fmt))
    # Order: inject context job_id first, then isolate by job_id.
    handler.addFilter(InjectJobIdFilter())
    handler.addFilter(JobIdFilter(job_id))
    package_logger.addHandler(handler)
    return handler


def detach_handler(handler: logging.Handler) -> None:
    """Remove *handler* from the audioforge logger and close it."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.removeHandler(handler)
    handler.close()


@contextmanager
def job_logging_context(job_id: str) -> Iterator[None]:
    """Bind *job_id* for the current context (thread/async task).

    Records logged inside the block receive ``job_id`` via
    :class:`InjectJobIdFilter` when not set explicitly in ``extra``.
    """
    if not job_id.strip():
        raise ValueError("job_id must be non-empty")
    token: Token[str | None] = _current_job_id.set(job_id)
    try:
        yield
    finally:
        _current_job_id.reset(token)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return an ``audioforge`` child logger (or the package root)."""
    if name is None:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
