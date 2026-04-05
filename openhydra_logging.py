# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
openhydra_logging — centralised logging configuration for all OpenHydra binaries.

Usage (in each binary's main()):
    from openhydra_logging import configure_logging
    configure_logging(json_logs=True)   # prod: machine-readable JSON lines
    configure_logging(json_logs=False)  # dev:  human-readable text (default)

The JSON formatter emits one JSON object per line, suitable for log
aggregators (Loki, ELK, CloudWatch, etc.).  Every record includes at minimum:

    {"ts": "2026-03-06T12:00:00.123", "level": "INFO",
     "logger": "coordinator.api_server", "msg": "request_done req_id=… …"}

Any keys added via `extra={…}` on a log call are promoted to top-level JSON
fields automatically, e.g.::

    logger.info("request_done", extra={"req_id": rid, "status": 200})
    # → {"ts":"…","level":"INFO","logger":"…","msg":"request_done",
    #    "req_id":"…","status":200}
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

# Standard LogRecord attributes that should NOT be lifted into extra fields.
_STDLIB_ATTRS: frozenset[str] = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName", "thread", "threadName",
})


class _JsonFormatter(logging.Formatter):
    """Format each :class:`logging.LogRecord` as a single-line JSON object."""

    # Some third-party libraries (e.g. absl-py via gRPC) remap the stdlib
    # level name WARNING → WARN.  Normalise back to the standard names so
    # downstream log aggregation pipelines see consistent values.
    _LEVEL_NORMALIZE: dict[str, str] = {"WARN": "WARNING"}

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        # Ensure exc_text / stack_info are populated on the record.
        super().format(record)

        ms = int(record.msecs)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
        level = self._LEVEL_NORMALIZE.get(record.levelname, record.levelname)
        doc: dict[str, Any] = {
            "ts": f"{ts}.{ms:03d}",
            "level": level,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Promote any extra= fields to top-level JSON.
        for key, value in record.__dict__.items():
            if key not in _STDLIB_ATTRS and not key.startswith("_"):
                doc[key] = value

        if record.exc_text:
            doc["exc"] = record.exc_text

        return json.dumps(doc, default=str)


class _TextFormatter(logging.Formatter):
    """Human-readable formatter for development / interactive use."""

    _FMT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
    _DATEFMT = "%Y-%m-%dT%H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool = False,
) -> None:
    """Configure the root logger for an OpenHydra binary.

    Call this **once**, early in ``main()``, after argument parsing so the
    deployment profile is known.  It replaces any prior ``basicConfig()``
    configuration.

    Args:
        level:     Log level string — ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
                   or ``"ERROR"``.  Case-insensitive.
        json_logs: When *True* emit JSON lines (one object per record).
                   When *False* (default) emit human-readable text.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter() if json_logs else _TextFormatter())

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Replace any handlers added by an earlier basicConfig call so this
    # function is idempotent when called multiple times (e.g. in tests).
    root.handlers = [handler]
