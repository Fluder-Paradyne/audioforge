"""Tests for structured logging helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from audioforge.logging_config import (
    JsonFormatter,
    TextFormatter,
    attach_job_file_handler,
    configure_logging,
    detach_handler,
    get_logger,
    job_logging_context,
    make_formatter,
    parse_log_level,
)
from audioforge.settings import AppSettings


@pytest.fixture(autouse=True)
def _reset_audioforge_logger() -> Iterator[None]:
    """Isolate logging configuration across tests."""
    logger = logging.getLogger("audioforge")
    logger.handlers.clear()
    logger.filters.clear()
    if hasattr(logger, "_audioforge_configured"):
        delattr(logger, "_audioforge_configured")
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    yield
    logger.handlers.clear()
    logger.filters.clear()
    if hasattr(logger, "_audioforge_configured"):
        delattr(logger, "_audioforge_configured")


def test_parse_log_level_name_and_int() -> None:
    assert parse_log_level("DEBUG") == logging.DEBUG
    assert parse_log_level("info") == logging.INFO
    assert parse_log_level(logging.WARNING) == logging.WARNING
    with pytest.raises(ValueError, match="Invalid log level"):
        parse_log_level("not-a-level")


def test_make_formatter_json_and_text() -> None:
    assert isinstance(make_formatter("json"), JsonFormatter)
    assert isinstance(make_formatter("JSON"), JsonFormatter)
    assert isinstance(make_formatter("text"), TextFormatter)
    assert isinstance(make_formatter("other"), TextFormatter)


def test_json_formatter_includes_structured_fields() -> None:
    record = logging.LogRecord(
        name="audioforge.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.job_id = "job-1"
    record.stage = "tts"
    record.event = "chapter_start"
    record.chapter_index = 3
    record.chapter_slug = "three"
    record.chars = 42
    record.voice = "af_heart"
    record.chapter_total = 10
    line = JsonFormatter().format(record)
    payload = json.loads(line)
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "audioforge.test"
    assert payload["job_id"] == "job-1"
    assert payload["stage"] == "tts"
    assert payload["event"] == "chapter_start"
    assert payload["chapter_index"] == 3
    assert payload["chapter_slug"] == "three"
    assert payload["chars"] == 42
    assert payload["voice"] == "af_heart"
    assert payload["chapter_total"] == 10
    assert "timestamp" in payload


def test_json_formatter_includes_exc_info() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            name="audioforge.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    line = JsonFormatter().format(record)
    payload = json.loads(line)
    assert "RuntimeError: boom" in payload["exc_info"]


def test_text_formatter_appends_extras_and_exc() -> None:
    plain = logging.LogRecord(
        name="audioforge.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="plain message",
        args=(),
        exc_info=None,
    )
    plain_text = TextFormatter().format(plain)
    assert "plain message" in plain_text
    assert " |" not in plain_text

    record = logging.LogRecord(
        name="audioforge.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="stage start",
        args=(),
        exc_info=None,
    )
    record.job_id = "abc"
    record.event = "stage_start"
    text = TextFormatter().format(record)
    assert "stage start" in text
    assert "job_id=abc" in text
    assert "event=stage_start" in text

    try:
        raise ValueError("x")
    except ValueError:
        import sys

        record.exc_info = sys.exc_info()
    text_exc = TextFormatter().format(record)
    assert "ValueError: x" in text_exc


def test_configure_logging_writes_to_stream() -> None:
    stream = StringIO()
    configure_logging(level="DEBUG", fmt="json", stream=stream, force=True)
    log = get_logger("unit")
    log.info("ping", extra={"event": "test", "job_id": "j1"})
    stream.seek(0)
    lines = [ln for ln in stream.read().splitlines() if ln.strip()]
    assert lines
    payload = json.loads(lines[-1])
    assert payload["message"] == "ping"
    assert payload["event"] == "test"
    assert payload["job_id"] == "j1"


def test_configure_logging_idempotent_updates_level() -> None:
    stream = StringIO()
    configure_logging(level="WARNING", fmt="text", stream=stream, force=True)
    configure_logging(level="DEBUG", fmt="text", stream=stream, force=False)
    root = logging.getLogger("audioforge")
    assert root.level == logging.DEBUG
    # Still a single stream handler from first configure when not forced.
    assert len(root.handlers) == 1


def test_configure_logging_force_replaces_console_keeps_job_file(
    tmp_path: Path,
) -> None:
    stream_a = StringIO()
    stream_b = StringIO()
    configure_logging(level="INFO", fmt="text", stream=stream_a, force=True)
    job_log = tmp_path / "job.log"
    file_handler = attach_job_file_handler(
        job_log, job_id="keep-me", fmt="text", level="INFO"
    )
    configure_logging(level="INFO", fmt="text", stream=stream_b, force=True)
    assert file_handler in logging.getLogger("audioforge").handlers
    get_logger().info("only-b", extra={"job_id": "keep-me"})
    assert "only-b" not in stream_a.getvalue()
    assert "only-b" in stream_b.getvalue()
    detach_handler(file_handler)


def test_attach_and_detach_job_file_handler(tmp_path: Path) -> None:
    configure_logging(level="INFO", fmt="text", stream=StringIO(), force=True)
    job_log = tmp_path / "work" / "job-1" / "job.log"
    handler = attach_job_file_handler(
        job_log, job_id="job-1", fmt="json", level="DEBUG"
    )
    get_logger("pipeline").info(
        "chapter ok",
        extra={"job_id": "job-1", "event": "chapter_end", "chapter_index": 1},
    )
    detach_handler(handler)
    assert job_log.is_file()
    lines = job_log.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload["job_id"] == "job-1"
    assert payload["event"] == "chapter_end"


def test_overlapping_job_logs_are_isolated(tmp_path: Path) -> None:
    """Two concurrent attaches must not cross-contaminate job.log files."""
    configure_logging(level="INFO", fmt="json", stream=StringIO(), force=True)
    log_a = tmp_path / "a" / "job.log"
    log_b = tmp_path / "b" / "job.log"
    ha = attach_job_file_handler(log_a, job_id="job-a", fmt="json", level="DEBUG")
    hb = attach_job_file_handler(log_b, job_id="job-b", fmt="json", level="DEBUG")
    log = get_logger("pipeline")

    with job_logging_context("job-a"):
        log.info("from-a-context", extra={"event": "chapter_start"})
    with job_logging_context("job-b"):
        log.info("from-b-context", extra={"event": "chapter_start"})
    # Explicit extra without context must still route correctly.
    log.info("explicit-a", extra={"job_id": "job-a", "event": "chapter_end"})
    log.info("explicit-b", extra={"job_id": "job-b", "event": "chapter_end"})
    # Unrelated job_id must not land in either file.
    log.info("noise", extra={"job_id": "other", "event": "noise"})

    detach_handler(ha)
    detach_handler(hb)

    text_a = log_a.read_text(encoding="utf-8")
    text_b = log_b.read_text(encoding="utf-8")
    assert "from-a-context" in text_a
    assert "explicit-a" in text_a
    assert "from-b-context" not in text_a
    assert "explicit-b" not in text_a
    assert "noise" not in text_a

    assert "from-b-context" in text_b
    assert "explicit-b" in text_b
    assert "from-a-context" not in text_b
    assert "explicit-a" not in text_b
    assert "noise" not in text_b

    # Context-injected lines still carry job_id in JSON payload.
    payloads_a = [json.loads(ln) for ln in text_a.strip().splitlines()]
    assert all(p["job_id"] == "job-a" for p in payloads_a)


def test_attach_rejects_empty_job_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="job_id"):
        attach_job_file_handler(tmp_path / "j.log", job_id="  ")


def test_job_logging_context_rejects_empty() -> None:
    with pytest.raises(ValueError, match="job_id"), job_logging_context(""):
        pass


def test_get_logger_names() -> None:
    assert get_logger(None).name == "audioforge"
    assert get_logger("audioforge").name == "audioforge"
    assert get_logger("audioforge.pipeline.tts").name == "audioforge.pipeline.tts"
    assert get_logger("pipeline.tts").name == "audioforge.pipeline.tts"


def test_settings_rejects_invalid_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_LOG_LEVEL", "DEGUG")
    with pytest.raises(ValidationError):
        AppSettings()


def test_settings_normalizes_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_LOG_LEVEL", "debug")
    assert AppSettings().log_level == "DEBUG"
