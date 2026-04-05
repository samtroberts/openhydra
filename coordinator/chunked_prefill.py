# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Chunked Prefill — split long prompts into chunks for pipeline interleaving.

When a long prompt (>chunk_size tokens) arrives, it's split into smaller
chunks.  Each chunk is sent through the pipeline as a separate pass with
KV cache storage.  Between chunks, decode requests from other clients can
execute — preventing one long prompt from monopolizing the pipeline.

Reference: arxiv.org/abs/2403.02310 (Sarathi-Serve, OSDI 2024)

Architecture:
    1. Split prompt into chunks of chunk_size words (or tokens if tokenizer available)
    2. Process chunk 0: full pipeline with kv_store_activation=True
    3. Process chunk 1..N: pipeline with kv_use_cached_activation=True
    4. Return final activation from the last chunk

The chain_fn callable is the pipeline execution function — typically
chain.run() or the SpecPipe scheduler. Between chunks, the coordinator
can schedule other requests.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkedPrefillConfig:
    """Configuration for chunked prefill."""
    enabled: bool = True
    chunk_size: int = 256  # Words per chunk (tokens if tokenizer available)
    min_prompt_length: int = 32  # Don't chunk prompts shorter than this


class ChunkedPrefill:
    """Splits long prompts into chunks for stall-free pipeline processing.

    Args:
        config: Chunked prefill configuration.
    """

    def __init__(self, config: ChunkedPrefillConfig) -> None:
        self._config = config
        self._stats = {
            "total_chunks": 0,
            "total_tokens_est": 0,
            "total_requests": 0,
            "chunked_requests": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def split_prompt(
        self,
        prompt: str,
        tokenizer: Any | None = None,
    ) -> list[str]:
        """Split a prompt into chunks.

        Uses the tokenizer for precise token-level chunking when available.
        Falls back to word-level splitting otherwise.

        Args:
            prompt: The full prompt text.
            tokenizer: Optional HuggingFace tokenizer for precise splitting.

        Returns:
            List of prompt chunks (first chunk contains the beginning).
        """
        if not prompt:
            return [""]

        chunk_size = max(1, self._config.chunk_size)

        # Try tokenizer-based splitting
        if tokenizer is not None:
            try:
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(token_ids) <= chunk_size:
                    return [prompt]
                chunks = []
                for i in range(0, len(token_ids), chunk_size):
                    chunk_ids = token_ids[i:i + chunk_size]
                    chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
                    chunks.append(chunk_text)
                return chunks if chunks else [prompt]
            except Exception:
                pass  # Fall through to word-level

        # Word-level splitting
        words = prompt.split()
        if len(words) <= chunk_size:
            return [prompt]

        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
        return chunks if chunks else [prompt]

    def process(
        self,
        prompt: str,
        chain_fn: Callable[..., list[float]],
        tokenizer: Any | None = None,
        session_id: str | None = None,
        **chain_kwargs: Any,
    ) -> list[float]:
        """Process a prompt through the pipeline, chunking if needed.

        Args:
            prompt: The full prompt text.
            chain_fn: Callable(prompt, **kwargs) -> list[float] that runs
                one pipeline pass and returns the activation.
            tokenizer: Optional tokenizer for precise chunking.
            session_id: KV cache session ID for cross-chunk persistence.
            **chain_kwargs: Additional kwargs passed to chain_fn.

        Returns:
            Final activation from the last chunk.
        """
        chunks = self.split_prompt(prompt, tokenizer)
        self._stats["total_requests"] += 1
        self._stats["total_tokens_est"] += len(prompt.split())

        if len(chunks) <= 1:
            # Short prompt — single pass, no chunking needed
            self._stats["total_chunks"] += 1
            return chain_fn(
                prompt,
                kv_session_id=session_id,
                kv_store_activation=bool(session_id),
                kv_use_cached_activation=False,
                **chain_kwargs,
            )

        # Long prompt — process in chunks
        self._stats["chunked_requests"] += 1
        kv_sid = session_id or f"chunked-{uuid.uuid4().hex[:8]}"
        activation: list[float] = []

        for i, chunk in enumerate(chunks):
            is_first = i == 0
            is_last = i == len(chunks) - 1
            self._stats["total_chunks"] += 1

            logger.info(
                "chunked_prefill: chunk %d/%d (%d words) session=%s",
                i + 1, len(chunks), len(chunk.split()), kv_sid,
            )

            activation = chain_fn(
                chunk,
                kv_session_id=kv_sid,
                kv_store_activation=True,
                kv_use_cached_activation=not is_first,
                **chain_kwargs,
            )

        logger.info(
            "chunked_prefill_complete: %d chunks, %d est tokens, session=%s",
            len(chunks), len(prompt.split()), kv_sid,
        )

        return activation
