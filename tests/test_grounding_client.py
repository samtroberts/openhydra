import json

from grounding.client_rag import GroundingClient, GroundingConfig, inject_grounding, pseudo_search


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_inject_grounding_and_pseudo_search():
    snippets = pseudo_search("openhydra decentralized inference network", max_snippets=2)
    assert len(snippets) == 2

    prompt = inject_grounding("hello", snippets)
    assert "[Grounding]" in prompt
    assert "Context about" in prompt


def test_grounding_client_parses_duckduckgo_payload(monkeypatch, tmp_path):
    config = GroundingConfig(
        cache_path=str(tmp_path / "grounding_cache.json"),
        use_network=True,
        fallback_enabled=False,
    )
    client = GroundingClient(config)

    payload = {
        "AbstractText": "OpenHydra is a decentralized inference network.",
        "RelatedTopics": [
            {"Text": "Peer-to-peer systems distribute load."},
            {"Topics": [{"Text": "gRPC supports efficient RPC streaming."}]},
        ],
    }

    monkeypatch.setattr("grounding.client_rag.request.urlopen", lambda req, timeout=0: _FakeResponse(payload))

    result = client.search("What is OpenHydra?", max_snippets=3)
    assert result.provider == "duckduckgo"
    assert result.cached is False
    assert len(result.snippets) == 3
    assert any("OpenHydra" in s for s in result.snippets)


def test_grounding_client_uses_cache(monkeypatch, tmp_path):
    config = GroundingConfig(
        cache_path=str(tmp_path / "grounding_cache.json"),
        use_network=True,
        fallback_enabled=False,
    )
    client = GroundingClient(config)

    payload = {"AbstractText": "Cached snippet."}
    monkeypatch.setattr("grounding.client_rag.request.urlopen", lambda req, timeout=0: _FakeResponse(payload))

    first = client.search("cache query", max_snippets=2)
    assert first.provider == "duckduckgo"
    assert first.cached is False

    def fail_urlopen(*args, **kwargs):
        raise AssertionError("network should not be called after cache hit")

    monkeypatch.setattr("grounding.client_rag.request.urlopen", fail_urlopen)
    second = client.search("cache query", max_snippets=2)
    assert second.provider == "cache"
    assert second.cached is True
    assert second.snippets == first.snippets


def test_grounding_client_fallback_on_error(monkeypatch, tmp_path):
    config = GroundingConfig(
        cache_path=str(tmp_path / "grounding_cache.json"),
        use_network=True,
        fallback_enabled=True,
    )
    client = GroundingClient(config)

    def fail_urlopen(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("grounding.client_rag.request.urlopen", fail_urlopen)

    result = client.search("fallback behavior for grounding", max_snippets=2)
    assert result.provider == "fallback"
    assert result.fallback_used is True
    assert result.error is not None
    assert len(result.snippets) == 2
