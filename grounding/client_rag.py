from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import ssl
import time
from typing import Any
from urllib import parse, request


def inject_grounding(prompt: str, snippets: list[str]) -> str:
    if not snippets:
        return prompt
    context = "\n".join(f"- {item}" for item in snippets)
    return f"{prompt}\n\n[Grounding]\n{context}"


def pseudo_search(prompt: str, max_snippets: int = 3) -> list[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9_-]+", prompt) if len(w) > 4]
    unique = list(dict.fromkeys(words))
    return [f"Context about {term}" for term in unique[:max_snippets]]


@dataclass(frozen=True)
class GroundingConfig:
    cache_path: str = ".openhydra/grounding_cache.json"
    cache_ttl_seconds: int = 900
    timeout_s: float = 3.0
    use_network: bool = True
    fallback_enabled: bool = True


@dataclass(frozen=True)
class GroundingResult:
    snippets: list[str]
    provider: str
    cached: bool
    fallback_used: bool
    error: str | None


class GroundingCache:
    def __init__(self, path: str, ttl_seconds: int = 900):
        self.path = Path(path)
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text())
        if isinstance(payload, dict):
            self.data = payload

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def get(self, query: str) -> list[str] | None:
        item = self.data.get(query)
        if not item:
            return None
        ts = float(item.get("ts", 0))
        age = time.time() - ts
        if age > self.ttl_seconds:
            self.data.pop(query, None)
            self._save()
            return None
        snippets = item.get("snippets", [])
        return [str(s) for s in snippets if str(s).strip()]

    def put(self, query: str, snippets: list[str]) -> None:
        self.data[query] = {
            "ts": time.time(),
            "snippets": snippets,
        }
        self._save()


class GroundingClient:
    def __init__(self, config: GroundingConfig | None = None):
        self.config = config or GroundingConfig()
        self.cache = GroundingCache(self.config.cache_path, ttl_seconds=self.config.cache_ttl_seconds)

    def _fetch_json(self, url: str) -> dict[str, Any]:
        req = request.Request(
            url,
            headers={"User-Agent": "OpenHydra/0.1 (grounding)"},
            method="GET",
        )
        ctx = ssl.create_default_context()
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except (ImportError, OSError):
            pass
        with request.urlopen(req, timeout=self.config.timeout_s, context=ctx) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _extract_related_topics(topics: list[Any], out: list[str]) -> None:
        for item in topics:
            if isinstance(item, dict):
                if isinstance(item.get("Text"), str) and item["Text"].strip():
                    out.append(item["Text"].strip())
                nested = item.get("Topics")
                if isinstance(nested, list):
                    GroundingClient._extract_related_topics(nested, out)

    def _duckduckgo_instant_answer(self, query: str, max_snippets: int) -> list[str]:
        params = parse.urlencode(
            {
                "q": query,
                "format": "json",
                "no_html": "1",
                "no_redirect": "1",
                "skip_disambig": "1",
            }
        )
        url = f"https://api.duckduckgo.com/?{params}"
        payload = self._fetch_json(url)

        snippets: list[str] = []
        for field in ("AbstractText", "Answer"):
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                snippets.append(value.strip())

        related = payload.get("RelatedTopics")
        if isinstance(related, list):
            self._extract_related_topics(related, snippets)

        normalized: list[str] = []
        seen: set[str] = set()
        for snippet in snippets:
            compact = " ".join(snippet.split())
            if not compact:
                continue
            if compact in seen:
                continue
            seen.add(compact)
            normalized.append(compact)
            if len(normalized) >= max_snippets:
                break

        return normalized

    def search(self, prompt: str, max_snippets: int = 3) -> GroundingResult:
        query = " ".join(prompt.split()).strip()
        if not query:
            return GroundingResult(snippets=[], provider="none", cached=False, fallback_used=False, error=None)

        cached = self.cache.get(query)
        if cached is not None:
            return GroundingResult(snippets=cached[:max_snippets], provider="cache", cached=True, fallback_used=False, error=None)

        if self.config.use_network:
            try:
                snippets = self._duckduckgo_instant_answer(query, max_snippets=max_snippets)
                if snippets:
                    self.cache.put(query, snippets)
                    return GroundingResult(
                        snippets=snippets,
                        provider="duckduckgo",
                        cached=False,
                        fallback_used=False,
                        error=None,
                    )
            except Exception as exc:  # pragma: no cover
                if self.config.fallback_enabled:
                    fallback = pseudo_search(prompt, max_snippets=max_snippets)
                    return GroundingResult(
                        snippets=fallback,
                        provider="fallback",
                        cached=False,
                        fallback_used=True,
                        error=str(exc),
                    )
                return GroundingResult(
                    snippets=[],
                    provider="duckduckgo",
                    cached=False,
                    fallback_used=False,
                    error=str(exc),
                )

        fallback = pseudo_search(prompt, max_snippets=max_snippets) if self.config.fallback_enabled else []
        provider = "fallback" if self.config.fallback_enabled else "none"
        return GroundingResult(
            snippets=fallback,
            provider=provider,
            cached=False,
            fallback_used=self.config.fallback_enabled,
            error=None,
        )
