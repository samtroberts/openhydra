from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import urllib.request
import json


@dataclass
class OpenHydraClient:
    base_url: str

    def models(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
