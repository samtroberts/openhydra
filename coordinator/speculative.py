from __future__ import annotations

from dataclasses import dataclass
import hashlib

from peer.model_shard import ModelShard


def _default_trust_remote_code(model_id: str) -> bool:
    return "qwen" in str(model_id or "").strip().lower()


@dataclass(frozen=True)
class SpeculativeSelection:
    accepted_tokens: list[str]
    matched_prefix: int
    mismatch: bool


class DraftTokenModel:
    """Deterministic toy draft model used for speculative decode experiments."""

    def __init__(self, seed: int = 13):
        self.seed = int(seed)

    def propose(self, prompt: str, max_tokens: int) -> list[str]:
        count = max(1, min(int(max_tokens), 48))
        digest = hashlib.sha256(f"{self.seed}:{prompt}".encode("utf-8")).digest()
        activation = [((b - 127) / 127.0) for b in digest]
        return ModelShard.decode_tokens(activation, max_tokens=count)


class PyTorchDraftModel:
    """Local lightweight HuggingFace draft model for distributed speculative decode."""

    def __init__(
        self,
        model_id: str = "sshleifer/tiny-gpt2",
        tokenizer_model_id: str = "gpt2",
        target: str = "cpu",
    ):
        self.model_id = str(model_id or "sshleifer/tiny-gpt2").strip() or "sshleifer/tiny-gpt2"
        self.tokenizer_model_id = str(tokenizer_model_id or "gpt2").strip() or "gpt2"
        self.target = str(target or "cpu").strip().lower()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "pytorch_draft_model_unavailable: install optional deps 'torch' and 'transformers'"
            ) from exc

        self._torch = torch
        requested_target = self.target
        if requested_target not in {"cpu", "cuda", "auto"}:
            requested_target = "cpu"
        if requested_target == "auto":
            requested_target = "cuda" if bool(torch.cuda.is_available()) else "cpu"
        if requested_target == "cuda" and not bool(torch.cuda.is_available()):
            requested_target = "cpu"
        self.target = requested_target
        self._device = torch.device("cuda" if self.target == "cuda" else "cpu")
        self._dtype = torch.float16 if self.target == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_model_id,
            trust_remote_code=_default_trust_remote_code(self.tokenizer_model_id),
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=self._dtype,
            trust_remote_code=_default_trust_remote_code(self.model_id),
        ).to(self._device)
        self._model.eval()

    def propose_token_ids(self, context_token_ids: list[int], max_tokens: int) -> list[int]:
        count = max(1, min(int(max_tokens), 16))
        sequence = [max(0, int(token)) for token in list(context_token_ids)]
        if not sequence:
            fallback = int(self._tokenizer.eos_token_id or 0)
            sequence = [fallback]

        out: list[int] = []
        with self._torch.no_grad():
            for _ in range(count):
                input_ids = self._torch.tensor([sequence], dtype=self._torch.long, device=self._device)
                logits = self._model(input_ids=input_ids, use_cache=False).logits
                next_token = int(self._torch.argmax(logits[:, -1, :], dim=-1).item())
                out.append(next_token)
                sequence.append(next_token)
        return out


def select_verified_tokens(verified_tokens: list[str], draft_tokens: list[str]) -> SpeculativeSelection:
    if not verified_tokens:
        return SpeculativeSelection(accepted_tokens=[], matched_prefix=0, mismatch=False)

    matched = 0
    for idx, token in enumerate(verified_tokens):
        if idx < len(draft_tokens) and token == draft_tokens[idx]:
            matched += 1
            continue
        return SpeculativeSelection(
            accepted_tokens=verified_tokens[: idx + 1],
            matched_prefix=matched,
            mismatch=(idx < len(draft_tokens)),
        )

    return SpeculativeSelection(
        accepted_tokens=list(verified_tokens),
        matched_prefix=matched,
        mismatch=False,
    )


@dataclass(frozen=True)
class SpeculativeTokenIdSelection:
    accepted_token_ids: list[int]
    matched_prefix: int
    mismatch: bool


def select_verified_token_ids(verified_token_ids: list[int], draft_token_ids: list[int]) -> SpeculativeTokenIdSelection:
    verified = [int(token) for token in list(verified_token_ids)]
    draft = [int(token) for token in list(draft_token_ids)]
    if not verified:
        return SpeculativeTokenIdSelection(accepted_token_ids=[], matched_prefix=0, mismatch=False)

    matched = 0
    for idx, token in enumerate(verified):
        if idx < len(draft) and token == draft[idx]:
            matched += 1
            continue
        accepted = draft[:idx]
        accepted.append(token)
        return SpeculativeTokenIdSelection(
            accepted_token_ids=accepted,
            matched_prefix=matched,
            mismatch=(idx < len(draft)),
        )

    return SpeculativeTokenIdSelection(
        accepted_token_ids=verified,
        matched_prefix=matched,
        mismatch=False,
    )
