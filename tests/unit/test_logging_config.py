"""Tests for structured logging helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest

from audioforge.logging_config import (
    JsonFormatter,
    TextFormatter,
    attach_job_file_handler,
    configure_logging,
    detach_handler,
    get_logger,
    make_formatter,
    parse_log_level,
)


@pytest.fixture(autouse=True)
def _reset_audioforge_logger() -> Iterator[None]:
    """Isolate logging configuration across tests."""
    logger = logging.getLogger("audioforge")
    logger.handlers.clear()
    if hasattr(logger, "_audioforge_configured"):
        delattr(logger, "_audioforge_configured")
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    yield
    logger.handlers.clear()
    if hasattr(logger, "_audioforge_configured"):
        delattr(logger, "_audioforge_configured")


def test_parse_log_level_name_and_int() -> None:
    assert parse_log_level("DEBUG") == logging.DEBUG
    assert parse_log_level("info") == logging.INFO
    assert parse_log_level(logging.WARNING) == logging.WARNING
    assert parse_log_level("not-a-level") == logging.INFO


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


def test_configure_logging_force_replaces_handlers() -> None:
    stream_a = StringIO()
    stream_b = StringIO()
    configure_logging(level="INFO", fmt="text", stream=stream_a, force=True)
    configure_logging(level="INFO", fmt="text", stream=stream_b, force=True)
    get_logger().info("only-b")
    assert "only-b" not in stream_a.getvalue()
    assert "only-b" in stream_b.getvalue()


def test_attach_and_detach_job_file_handler(tmp_path: Path) -> None:
    configure_logging(level="INFO", fmt="text", stream=StringIO(), force=True)
    job_log = tmp_path / "work" / "job-1" / "job.log"
    handler = attach_job_file_handler(job_log, fmt="json", level="DEBUG")
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


def test_get_logger_names() -> None:
    assert get_logger(None).name == "audioforge"
    assert get_logger("audioforge").name == "audioforge"
    assert get_logger("audioforge.pipeline.tts").name == "audioforge.pipeline.tts"
    assert get_logger("pipeline.tts").name == "audioforge.pipeline.tts"
