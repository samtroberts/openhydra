"""Tests for openhydra_logging — JSON formatter and configure_logging()."""
import json
import logging
import io

from openhydra_logging import _JsonFormatter, _TextFormatter, configure_logging


# ---------------------------------------------------------------------------
# _JsonFormatter
# ---------------------------------------------------------------------------

def _emit(level: int, msg: str, extra: dict | None = None, exc: Exception | None = None) -> dict:
    """Emit a single log record through _JsonFormatter and return the parsed JSON."""
    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname="test_logging_config.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=(type(exc), exc, exc.__traceback__) if exc else None,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    line = formatter.format(record)
    return json.loads(line)


def test_json_formatter_includes_required_fields():
    doc = _emit(logging.INFO, "hello world")
    assert doc["level"] == "INFO"
    assert doc["logger"] == "test.logger"
    assert doc["msg"] == "hello world"
    assert "ts" in doc
    # Timestamp format: 2026-03-06T12:00:00.123
    assert "T" in doc["ts"]
    assert "." in doc["ts"]


def test_json_formatter_promotes_extra_fields():
    doc = _emit(logging.WARNING, "req done", extra={"req_id": "abc-123", "status": 200})
    assert doc["req_id"] == "abc-123"
    assert doc["status"] == 200
    assert doc["level"] == "WARNING"


def test_json_formatter_does_not_leak_stdlib_attrs():
    doc = _emit(logging.INFO, "msg")
    # These internal attrs should NOT appear at the top level
    for key in ("args", "levelno", "lineno", "pathname", "process", "thread"):
        assert key not in doc, f"stdlib attr {key!r} leaked into JSON"


def test_json_formatter_includes_exc_text_on_exception():
    try:
        raise ValueError("boom")
    except ValueError as exc:
        doc = _emit(logging.ERROR, "caught", exc=exc)
    assert "exc" in doc
    assert "ValueError" in doc["exc"]
    assert "boom" in doc["exc"]


def test_json_formatter_emits_valid_json_line():
    formatter = _JsonFormatter()
    record = logging.makeLogRecord({"name": "x", "levelname": "INFO", "msg": "ok", "levelno": logging.INFO})
    line = formatter.format(record)
    assert "\n" not in line
    parsed = json.loads(line)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# _TextFormatter
# ---------------------------------------------------------------------------

def test_text_formatter_emits_human_readable_line():
    formatter = _TextFormatter()
    record = logging.makeLogRecord({
        "name": "coordinator.api_server",
        "levelname": "INFO",
        "levelno": logging.INFO,
        "msg": "server_start port=8080",
    })
    line = formatter.format(record)
    assert "INFO" in line
    assert "server_start" in line
    assert "coordinator.api_server" in line


# ---------------------------------------------------------------------------
# configure_logging()
# ---------------------------------------------------------------------------

def _capture_log(json_logs: bool, msg: str) -> str:
    """Call configure_logging, emit one record, and capture the output."""
    stream = io.StringIO()
    configure_logging(json_logs=json_logs)
    root = logging.getLogger()
    # Replace the StreamHandler's stream with our StringIO for capture.
    root.handlers[0].stream = stream  # type: ignore[attr-defined]

    log = logging.getLogger("test.capture")
    log.info(msg)
    return stream.getvalue().strip()


def test_configure_logging_text_mode_produces_human_line():
    line = _capture_log(json_logs=False, msg="hello text")
    assert "hello text" in line
    # Should NOT be a JSON object
    try:
        json.loads(line)
        is_json = True
    except json.JSONDecodeError:
        is_json = False
    assert not is_json


def test_configure_logging_json_mode_produces_json_line():
    line = _capture_log(json_logs=True, msg="hello json")
    doc = json.loads(line)  # must not raise
    assert doc["msg"] == "hello json"
    assert doc["level"] == "INFO"


def test_configure_logging_replaces_existing_handlers():
    # Add a stray handler first.
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())
    before = len(root.handlers)

    configure_logging(json_logs=False)
    # After configure_logging there should be exactly 1 handler.
    assert len(root.handlers) == 1


def test_configure_logging_respects_level():
    configure_logging(level="WARNING", json_logs=True)
    root = logging.getLogger()
    assert root.level == logging.WARNING
