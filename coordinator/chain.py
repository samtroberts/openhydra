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

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
import uuid

import grpc

from compression.autoencoder import CompressionProfile, TensorAutoencoder
from coordinator.path_finder import PeerEndpoint
from coordinator.transport import TransportConfig, create_channel
from peer.crypto import (
    build_activation_envelope,
    build_activation_envelope_with_pubkey,
    build_onion_route_envelope,
    build_onion_route_envelope_with_pubkeys,
    required_layers_for_level,
    verify_privacy_audit_tag,
)
from peer import peer_pb2
from peer import peer_pb2_grpc
from peer.model_shard import ModelShard


@dataclass(frozen=True)
class StageTrace:
    peer_id: str
    latency_ms: float
    stage_index: int
    attempt: int = 1
    failed_peer_id: str | None = None


@dataclass(frozen=True)
class _StageResult:
    activation: list[float]
    latency_ms: float
    latent_dim: int = 0
    activation_hash: bytes = b""


@dataclass(frozen=True)
class ChainResult:
    request_id: str
    text: str
    activation: list[float]
    traces: list[StageTrace]
    latency_ms: float
    compression: dict[str, float | int | bool | str] | None = None
    encryption: dict[str, float | int | bool | str] | None = None
    kv: dict[str, float | int | bool | str | None] | None = None
    activation_hash: bytes = b""  # TOPLOC hash from last pipeline stage


class InferenceChain:
    """Sequential activation-forwarding chain with optional stage failover."""

    def __init__(
        self,
        pipeline: list[PeerEndpoint],
        timeout_ms: int = 500,
        transport_config: TransportConfig | None = None,
        tensor_autoencoder_enabled: bool = False,
        tensor_autoencoder_latent_dim: int = 1024,
        advanced_encryption_enabled: bool = False,
        advanced_encryption_seed: str = "openhydra-tier3-dev-seed",
        advanced_encryption_level: str = "standard",
        activation_quantization_enabled: bool = False,
        stream_pool: Any | None = None,
        session: Any | None = None,
        peer_dead_callback: Any | None = None,
    ):
        if not pipeline:
            raise ValueError("pipeline must contain at least one peer")
        self.pipeline = pipeline
        self.timeout_s = timeout_ms / 1000.0
        self.transport_config = transport_config or TransportConfig()
        self.tensor_autoencoder_enabled = bool(tensor_autoencoder_enabled)
        self.activation_quantization_enabled = bool(activation_quantization_enabled)
        self.advanced_encryption_enabled = bool(advanced_encryption_enabled)
        self.advanced_encryption_seed = str(advanced_encryption_seed)
        self.advanced_encryption_level = str(advanced_encryption_level)
        # PR-3 (B1+B2): fast-fail hook. When set, the chain calls
        # ``peer_dead_callback(libp2p_peer_id, reason)`` synchronously from
        # the gRPC error handler so the coordinator can publish a
        # ``PEER_DEAD`` gossip event and wake the NegotiationLoop before
        # re-routing. Callback failures are swallowed — the retry loop
        # must never be derailed by a side-channel failure. ``None``
        # disables the hook (default, backwards-compat).
        self._peer_dead_callback = peer_dead_callback
        self._last_stage_kv_cache_hit = False
        self._autoencoder: TensorAutoencoder | None = None
        if self.tensor_autoencoder_enabled:
            self._autoencoder = TensorAutoencoder(
                CompressionProfile(latent_dim=max(1, int(tensor_autoencoder_latent_dim)))
            )
        self._last_onion_route_state: dict[str, object] | None = None
        self._last_onion_next_peer_id: str | None = None
        self._last_privacy_audit: dict[str, float | int | bool | str] | None = None
        # Phase B: Persistent streaming + session history
        self._stream_pool = stream_pool
        self._session = session

    def _request_stage(
        self,
        peer: PeerEndpoint,
        request_id: str,
        prompt: str,
        activation: list[float],
        stage_index: int,
        total_stages: int,
        max_tokens: int,
        kv_session_id: str | None = None,
        kv_store_activation: bool = False,
        kv_use_cached_activation: bool = False,
        onion_route_state: dict[str, object] | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        deadline: float | None = None,
        prompt_token_ids: list[int] | tuple[int, ...] | None = None,
    ) -> _StageResult:
        plain_activation = activation
        wire_activation = activation
        encrypted_activation = b""
        encryption_nonces: list[bytes] = []
        encryption_ephemeral_public_keys: list[bytes] = []
        encryption_suite = ""
        encryption_layers = 0
        compression_codec = ""
        compression_original_dim = 0
        compression_latent_dim = 0
        kv_session = str(kv_session_id or "").strip()
        kv_store = bool(kv_store_activation and kv_session)
        kv_use_cached = bool(kv_use_cached_activation and kv_session)
        onion_route_ciphertext = b""
        onion_route_nonces: list[bytes] = []
        onion_route_ephemeral_public_keys: list[bytes] = []
        onion_route_suite = ""
        onion_route_layers = 0
        if onion_route_state is not None:
            onion_route_ciphertext = bytes(onion_route_state.get("ciphertext", b""))
            onion_route_nonces = [bytes(item) for item in list(onion_route_state.get("nonces", []))]
            onion_route_ephemeral_public_keys = [
                bytes(item) for item in list(onion_route_state.get("ephemeral_public_keys", []))
            ]
            onion_route_suite = str(onion_route_state.get("suite", "") or "")
            onion_route_layers = max(0, int(onion_route_state.get("layers", 0) or 0))
        peer_runtime_backend = str(getattr(peer, "runtime_backend", "")).strip().lower()
        use_placeholder_autoencoder = (
            self._autoencoder is not None
            and stage_index > 0
            and activation
            and not peer_runtime_backend.startswith("pytorch")
        )

        if use_placeholder_autoencoder:
            latent = self._autoencoder.encode(activation)
            wire_activation = latent
            compression_codec = "tensor_autoencoder_mean_pool"
            compression_original_dim = len(activation)
            compression_latent_dim = len(latent)

        # INT8 activation compression (P0-B): quantize before wire transfer.
        # The activation payload has a [seq_len, hidden_size] header prefix
        # (2 floats) followed by the hidden state data. We preserve the
        # header exactly (as raw float32 bytes) and only INT8-quantize the
        # data values. This avoids header corruption from absmax scaling.
        _quantized_activation = b""
        _quantized_scales: list[float] = []
        _activation_quantization = ""
        if (
            self.activation_quantization_enabled
            and stage_index > 0
            and wire_activation
            and len(wire_activation) > 2
            and not self.advanced_encryption_enabled
        ):
            import struct as _struct
            from peer.activation_codec import quantize_int8
            _header = wire_activation[:2]
            _payload = wire_activation[2:]
            _quantized_payload, _quantized_scales = quantize_int8(_payload)
            # Pack: 2 header floats (8 bytes, exact) + quantized payload bytes
            _header_bytes = _struct.pack('<2f', *_header)
            _quantized_activation = _header_bytes + _quantized_payload
            _activation_quantization = "int8"
            plain_activation = []

        if self.advanced_encryption_enabled and wire_activation:
            peer_pubkey_hex = str(getattr(peer, "public_key_hex", "") or "")
            if peer_pubkey_hex:
                try:
                    raw_pub = bytes.fromhex(peer_pubkey_hex)
                except ValueError:
                    raw_pub = b""
            else:
                raw_pub = b""

            if raw_pub:
                envelope = build_activation_envelope_with_pubkey(
                    wire_activation,
                    raw_public_key_bytes=raw_pub,
                    peer_id=peer.peer_id,
                    request_id=request_id,
                    stage_index=stage_index,
                    level=self.advanced_encryption_level,
                )
            else:
                envelope = build_activation_envelope(
                    wire_activation,
                    peer_id=peer.peer_id,
                    request_id=request_id,
                    stage_index=stage_index,
                    shared_secret_seed=self.advanced_encryption_seed,
                    level=self.advanced_encryption_level,
                )
            plain_activation = []
            encrypted_activation = envelope.ciphertext
            encryption_nonces = list(envelope.nonces)
            encryption_ephemeral_public_keys = list(envelope.ephemeral_public_keys)
            encryption_suite = envelope.suite
            encryption_layers = envelope.layers
        else:
            plain_activation = wire_activation

        # Binary-pack float32 activation for faster serialization.
        # struct.pack is a single C call vs Python iterating each float.
        # Binary packing supersedes INT8 quantization — INT8 corrupts the
        # [seq_len, hidden_size] header embedded in the activation payload.
        _activation_packed = b""
        if plain_activation and not _quantized_activation:
            # PR-1: route through vectorised numpy packer (~10× faster than
            # ``struct.pack(*list)`` for 10⁵+ element activations).
            from peer.activation_codec import pack_fp32 as _pack_fp32
            logging.debug(
                "chain_pack: stage=%d n_floats=%d first2=[%s]",
                stage_index, len(plain_activation),
                ",".join(f"{v:.1f}" for v in plain_activation[:2]) if len(plain_activation) >= 2 else "?",
            )
            _activation_packed = _pack_fp32(plain_activation)
            plain_activation = []  # clear repeated float — use packed bytes

        _t_serial_start = time.perf_counter()
        req = peer_pb2.ForwardRequest(
            request_id=request_id,
            prompt=prompt if stage_index == 0 else "",
            activation=plain_activation,
            stage_index=stage_index,
            total_stages=total_stages,
            max_tokens=max_tokens,
            encrypted_activation=encrypted_activation,
            encryption_nonces=encryption_nonces,
            encryption_ephemeral_public_keys=encryption_ephemeral_public_keys,
            encryption_suite=encryption_suite,
            encryption_layers=encryption_layers,
            compression_codec=compression_codec,
            compression_original_dim=compression_original_dim,
            compression_latent_dim=compression_latent_dim,
            kv_session_id=kv_session,
            kv_store_activation=kv_store,
            kv_use_cached_activation=kv_use_cached,
            onion_route_ciphertext=onion_route_ciphertext,
            onion_route_nonces=onion_route_nonces,
            onion_route_ephemeral_public_keys=onion_route_ephemeral_public_keys,
            onion_route_suite=onion_route_suite,
            onion_route_layers=onion_route_layers,
            decode_do_sample=bool(decode_do_sample),
            decode_temperature=float(decode_temperature or 0.0),
            decode_top_p=float(decode_top_p or 0.0),
            decode_top_k=max(0, int(decode_top_k or 0)),
            decode_seed=int(decode_seed or 0),
            # Phase 3: pass shard layer range so the peer can validate it matches
            # its startup configuration and surface the info in its own logs.
            shard_layer_start=max(0, int(getattr(peer, "layer_start", 0) or 0)),
            shard_layer_end=max(0, int(getattr(peer, "layer_end", 0) or 0)),
            shard_total_layers=max(0, int(getattr(peer, "total_layers", 0) or 0)),
            quantized_activation=_quantized_activation,
            quantized_scales=_quantized_scales,
            activation_quantization=_activation_quantization,
            # Phase 4: Gemma 4 sharded adapter — ship the original prompt
            # token IDs to every stage so downstream peers can recompute the
            # per-layer input tensor locally. Unused by non-Gemma-4 families;
            # empty list when the caller didn't supply ids.
            prompt_token_ids=list(int(t) for t in (prompt_token_ids or [])),
            activation_packed=_activation_packed,
        )

        # --- Deadline-aware per-stage timeout ---
        # Use remaining wall-clock time when a request deadline was propagated;
        # never exceed the configured per-hop ceiling (self.timeout_s).
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError(
                    f"deadline_exceeded: no time remaining before gRPC Forward "
                    f"to peer {peer.peer_id} stage {stage_index}"
                )
            effective_timeout = min(self.timeout_s, remaining)
        else:
            effective_timeout = self.timeout_s

        _t_serial_ms = (time.perf_counter() - _t_serial_start) * 1000
        _t_grpc_start = time.perf_counter()
        t0 = time.perf_counter()

        # Phase B: Try persistent streaming connection first (avoids
        # per-request channel creation overhead ~5-15ms per hop).
        _kv_sid = str(kv_session_id or "").strip()
        _used_stream = False
        if self._stream_pool and _kv_sid:
            try:
                handle = self._stream_pool.get_or_create(
                    peer.peer_id, _kv_sid, peer.host, peer.port,
                )
                if handle is not None:
                    response = self._stream_pool.send_and_receive(
                        handle, req, timeout_s=effective_timeout,
                    )
                    _used_stream = True
            except Exception as exc:
                logging.debug("stream_fallback: peer=%s err=%s", peer.peer_id, exc)
                _used_stream = False

        if not _used_stream:
            # Cross-ISP relay: if the peer requires relay and we have a
            # P2P node, tunnel the gRPC request through libp2p instead
            # of connecting directly (the remote IP is unreachable).
            _p2p_node = getattr(self, '_p2p_node', None)
            _peer_libp2p_id = str(getattr(peer, 'libp2p_peer_id', '') or '').strip()
            if _p2p_node is not None and getattr(peer, 'requires_relay', False) and _peer_libp2p_id:
                req_bytes = b'\x01' + req.SerializeToString()  # 0x01 = ForwardRequest
                resp_bytes = _p2p_node.proxy_forward(
                    target_peer_id=_peer_libp2p_id,
                    data=req_bytes,
                )
                # Strip method prefix from response.
                raw_resp = bytes(resp_bytes)
                if raw_resp and raw_resp[0:1] in (b'\x01', b'\x02'):
                    raw_resp = raw_resp[1:]
                response = peer_pb2.ForwardResponse()
                response.ParseFromString(raw_resp)
            else:
                with create_channel(peer.address, self.transport_config) as channel:
                    stub = peer_pb2_grpc.PeerStub(channel)
                    response = stub.Forward(req, timeout=effective_timeout)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        _t_grpc_ms = (time.perf_counter() - _t_grpc_start) * 1000

        # Phase B: Record in session history for failover replay
        if self._session is not None:
            self._session.record(req, response)
        _t_deser_start = time.perf_counter()
        self._last_stage_kv_cache_hit = bool(getattr(response, "kv_cache_hit", False))
        if response.error:
            raise RuntimeError(f"peer {peer.peer_id} failed at stage {stage_index}: {response.error}")
        self._last_onion_next_peer_id = str(getattr(response, "onion_next_peer_id", "") or "")
        remaining_ciphertext = bytes(getattr(response, "onion_route_ciphertext", b""))
        if remaining_ciphertext:
            self._last_onion_route_state = {
                "ciphertext": remaining_ciphertext,
                "nonces": [bytes(item) for item in list(getattr(response, "onion_route_nonces", []))],
                "ephemeral_public_keys": [
                    bytes(item) for item in list(getattr(response, "onion_route_ephemeral_public_keys", []))
                ],
                "suite": str(getattr(response, "onion_route_suite", "") or ""),
                "layers": max(0, int(getattr(response, "onion_route_layers", 0) or 0)),
            }
        else:
            self._last_onion_route_state = None

        dp_applied = bool(getattr(response, "dp_noise_applied", False))
        dp_configured_variance = max(0.0, float(getattr(response, "dp_noise_configured_variance", 0.0) or 0.0))
        dp_observed_variance = max(0.0, float(getattr(response, "dp_noise_observed_variance", 0.0) or 0.0))
        dp_observed_std = max(0.0, float(getattr(response, "dp_noise_observed_std", 0.0) or 0.0))
        dp_payload_index = max(0, int(getattr(response, "dp_noise_payload_index", 0) or 0))
        dp_audit_tag = str(getattr(response, "dp_noise_audit_tag", "") or "")
        dp_audit_tag_valid = False
        if dp_applied and dp_payload_index > 0 and dp_audit_tag:
            dp_audit_tag_valid = verify_privacy_audit_tag(
                peer_id=peer.peer_id,
                request_id=request_id,
                stage_index=int(stage_index),
                payload_index=dp_payload_index,
                configured_variance=dp_configured_variance,
                observed_variance=dp_observed_variance,
                observed_std=dp_observed_std,
                audit_tag=dp_audit_tag,
                shared_secret_seed=self.advanced_encryption_seed,
            )
        self._last_privacy_audit = {
            "applied": dp_applied,
            "configured_variance": dp_configured_variance,
            "observed_variance": dp_observed_variance,
            "observed_std": dp_observed_std,
            "payload_index": dp_payload_index,
            "audit_tag": dp_audit_tag,
            "audit_tag_valid": dp_audit_tag_valid,
        }

        response_latent_dim = max(0, int(getattr(response, "compression_latent_dim", 0) or 0))
        if response_latent_dim <= 0:
            response_latent_dim = len(activation)

        _t_deser_ms = (time.perf_counter() - _t_deser_start) * 1000
        logging.info(
            "PROFILE _request_stage: serial=%.1fms grpc=%.1fms deser=%.1fms total=%.1fms",
            _t_serial_ms, _t_grpc_ms, _t_deser_ms,
            _t_serial_ms + _t_grpc_ms + _t_deser_ms,
        )

        # Prefer activation_packed (binary) over repeated float activation.
        _resp_packed = bytes(getattr(response, "activation_packed", b"") or b"")
        if _resp_packed and len(_resp_packed) >= 8:
            import struct as _struct_unpack
            _n_floats = len(_resp_packed) // 4
            resp_activation = list(_struct_unpack.unpack(f'<{_n_floats}f', _resp_packed))
        else:
            resp_activation = list(response.activation)

        # INT8 dequantization on response (header-preserving format)
        resp_quant = str(getattr(response, "activation_quantization", "") or "")
        if resp_quant == "int8" and getattr(response, "quantized_activation", b""):
            import struct as _struct_resp
            from peer.activation_codec import dequantize_int8
            _raw_resp = bytes(response.quantized_activation)
            if len(_raw_resp) > 8:
                _resp_header = list(_struct_resp.unpack('<2f', _raw_resp[:8]))
                _resp_payload = dequantize_int8(_raw_resp[8:], list(response.quantized_scales))
                resp_activation = _resp_header + _resp_payload
            else:
                resp_activation = dequantize_int8(_raw_resp, list(response.quantized_scales))

        _resp_hash = bytes(getattr(response, "activation_hash", b"") or b"")

        return _StageResult(
            activation=resp_activation,
            latency_ms=latency_ms,
            latent_dim=response_latent_dim,
            activation_hash=_resp_hash,
        )

    def verify_tokens(
        self,
        prompt: str,
        draft_token_ids: list[int],
        request_id: str | None = None,
        kv_session_id: str | None = None,
        deadline: float | None = None,
        **decode_controls: object,
    ) -> list[int]:
        """Send K draft tokens through the pipeline for batch verification.

        Each peer verifies the draft tokens against its model and returns
        the model's actual predictions.  Used by DSD (Decentralized
        Speculative Decoding) to verify locally-generated draft tokens.

        Returns:
            List of verified token IDs (the model's actual next-token
            predictions for each draft position).
        """
        rid = request_id or str(uuid.uuid4())
        k = len(draft_token_ids)
        if k == 0:
            return []

        # Use the last peer in the pipeline (full-model or last shard)
        peer = self.pipeline[-1]

        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                return []
            effective_timeout = min(self.timeout_s, remaining)
        else:
            effective_timeout = self.timeout_s

        req = peer_pb2.ForwardRequest(
            request_id=rid,
            prompt=prompt,
            activation=[],
            stage_index=0,
            total_stages=1,
            max_tokens=1,
            draft_token_ids=draft_token_ids,
            verify_batch_size=k,
        )

        with create_channel(peer.address, self.transport_config) as channel:
            stub = peer_pb2_grpc.PeerStub(channel)
            response = stub.Forward(req, timeout=effective_timeout)

        if response.error:
            raise RuntimeError(f"verify failed: {response.error}")

        return [int(t) for t in response.verified_token_ids]

    def _stage_candidates(
        self,
        stage_peer: PeerEndpoint,
        failover_pool: list[PeerEndpoint],
        max_failovers_per_stage: int,
    ) -> list[PeerEndpoint]:
        candidates = [stage_peer]
        for peer in failover_pool:
            if peer.peer_id == stage_peer.peer_id:
                continue
            if any(p.peer_id == peer.peer_id for p in candidates):
                continue
            candidates.append(peer)
            if len(candidates) >= 1 + max_failovers_per_stage:
                break
        return candidates

    def run(
        self,
        prompt: str,
        max_tokens: int = 24,
        request_id: str | None = None,
        failover_pool: list[PeerEndpoint] | None = None,
        max_failovers_per_stage: int = 0,
        initial_activation: list[float] | None = None,
        kv_session_id: str | None = None,
        kv_use_cached_activation: bool = False,
        kv_store_activation: bool = False,
        kv_cache_stage_index: int = 0,
        kv_cache_all_stages: bool = False,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
        deadline: float | None = None,
        prompt_token_ids: list[int] | tuple[int, ...] | None = None,
    ) -> ChainResult:
        rid = request_id or str(uuid.uuid4())
        activation: list[float] = list(initial_activation or [])

        # Phase 4 (Gemma 4 sharded adapter): derive the prompt-token sidecar
        # that ships with every stage so downstream peers can recompute
        # per-layer inputs locally. When the caller passes
        # ``prompt_token_ids`` explicitly we use it as-is; otherwise we
        # auto-detect the common case where ``initial_activation`` is a
        # list of integer-valued floats (the Phase 1 non-streaming loop
        # always passes token IDs this way). The sidecar is empty for
        # non-PyTorch / non-Gemma-4 paths — peers just ignore it.
        _chain_prompt_token_ids: list[int] = []
        if prompt_token_ids:
            try:
                _chain_prompt_token_ids = [int(t) for t in prompt_token_ids]
            except (TypeError, ValueError):
                _chain_prompt_token_ids = []
        elif initial_activation and all(
            abs(float(v) - round(float(v))) < 1e-6 and float(v) >= 0
            for v in initial_activation[:64]
        ):
            try:
                _chain_prompt_token_ids = [
                    int(round(float(v))) for v in initial_activation
                ]
            except (TypeError, ValueError):
                _chain_prompt_token_ids = []
        traces: list[StageTrace] = []
        compression_input = 0
        compression_latent = 0
        compression_hops = 0
        encryption_hops = 0
        onion_layers_peeled = 0
        privacy_audit_required = bool(
            self.advanced_encryption_enabled
            and str(self.advanced_encryption_level).strip().lower() == "maximum"
            and len(self.pipeline) > 1
        )
        privacy_audit_records: list[dict[str, float | int | bool | str]] = []
        privacy_audit_violations: list[str] = []
        kv_session = str(kv_session_id or "").strip()
        kv_cache_hit = False
        kv_cache_peer_id: str | None = None
        onion_enabled = bool(
            self.advanced_encryption_enabled
            and str(self.advanced_encryption_level).strip().lower() in {"enhanced", "maximum"}
            and len(self.pipeline) > 1
        )
        onion_route_state: dict[str, object] | None = None
        if onion_enabled:
            route_peer_ids = [peer.peer_id for peer in self.pipeline]
            peer_pubkeys: dict[str, bytes] = {}
            for peer in self.pipeline:
                hex_key = str(getattr(peer, "public_key_hex", "") or "")
                if not hex_key:
                    continue
                try:
                    peer_pubkeys[peer.peer_id] = bytes.fromhex(hex_key)
                except ValueError:
                    continue

            if len(peer_pubkeys) == len(self.pipeline):
                onion_envelope = build_onion_route_envelope_with_pubkeys(
                    route_peer_ids,
                    peer_public_keys=peer_pubkeys,
                    request_id=rid,
                )
            else:
                onion_envelope = build_onion_route_envelope(
                    route_peer_ids,
                    request_id=rid,
                    shared_secret_seed=self.advanced_encryption_seed,
                )
            onion_route_state = {
                "ciphertext": onion_envelope.ciphertext,
                "nonces": list(onion_envelope.nonces),
                "ephemeral_public_keys": list(onion_envelope.ephemeral_public_keys),
                "suite": onion_envelope.suite,
                "layers": onion_envelope.layers,
            }

        started = time.perf_counter()
        pool = failover_pool or []
        _last_activation_hash = b""
        for stage_index, stage_peer in enumerate(self.pipeline):
            candidates = self._stage_candidates(stage_peer, pool, max_failovers_per_stage)
            errors: list[str] = []
            failed_peer_id: str | None = None
            use_kv_stage = bool(
                kv_session and (
                    bool(kv_cache_all_stages)
                    or stage_index == int(kv_cache_stage_index)
                )
            )
            stage_use_cached = bool(use_kv_stage and kv_use_cached_activation)
            stage_store_activation = bool(use_kv_stage and kv_store_activation)

            for attempt, candidate in enumerate(candidates, start=1):
                try:
                    # B1 rendezvous: if the candidate is relay-bound, ask it
                    # (via gossipsub) to simultaneously dial us. The
                    # gossip client's per-pair 5 s debounce means it's
                    # safe to call this unconditionally on every attempt —
                    # duplicate publishes collapse transparently.
                    _gossip = getattr(self, "_gossip_client", None)
                    _self_libp2p = getattr(self, "_self_libp2p_peer_id", "")
                    if _gossip is not None and _self_libp2p:
                        try:
                            from coordinator.path_finder import maybe_request_hole_punch as _maybe_rhp
                            _maybe_rhp(
                                _gossip,
                                self_libp2p_peer_id=str(_self_libp2p),
                                peer=candidate,
                            )
                        except Exception:  # pragma: no cover — never derail
                            logging.debug(
                                "b1_rendezvous_publish_failed",
                                exc_info=True,
                            )
                    stage_input = list(activation)
                    candidate_backend = str(getattr(candidate, "runtime_backend", "")).strip().lower()
                    compressed_transfer = (
                        self._autoencoder is not None
                        and stage_index > 0
                        and bool(stage_input)
                        and not candidate_backend.startswith("pytorch")
                    )
                    encrypted_transfer = self.advanced_encryption_enabled and stage_index > 0 and bool(stage_input)
                    self._last_stage_kv_cache_hit = False
                    stage_kwargs: dict[str, object] = {}
                    if use_kv_stage:
                        stage_kwargs.update(
                            {
                                "kv_session_id": kv_session,
                                "kv_store_activation": stage_store_activation,
                                "kv_use_cached_activation": stage_use_cached,
                            }
                        )
                    if onion_enabled:
                        stage_kwargs["onion_route_state"] = onion_route_state
                    if (
                        decode_do_sample is not None
                        or decode_temperature is not None
                        or decode_top_p is not None
                        or decode_top_k is not None
                        or decode_seed is not None
                    ):
                        stage_kwargs.update(
                            {
                                "decode_do_sample": decode_do_sample,
                                "decode_temperature": decode_temperature,
                                "decode_top_p": decode_top_p,
                                "decode_top_k": decode_top_k,
                                "decode_seed": decode_seed,
                            }
                        )

                    stage_result = self._request_stage(
                        peer=candidate,
                        request_id=rid,
                        prompt=prompt,
                        activation=stage_input,
                        stage_index=stage_index,
                        total_stages=len(self.pipeline),
                        max_tokens=max_tokens,
                        deadline=deadline,
                        prompt_token_ids=_chain_prompt_token_ids,
                        **stage_kwargs,
                    )
                    if isinstance(stage_result, tuple):
                        activation = list(stage_result[0])
                        latency_ms = float(stage_result[1])
                        stage_latent_dim = len(stage_input)
                    else:
                        activation = list(stage_result.activation)
                        latency_ms = float(stage_result.latency_ms)
                        stage_latent_dim = max(0, int(stage_result.latent_dim))
                    # Capture TOPLOC hash from last pipeline stage
                    _last_activation_hash = getattr(stage_result, "activation_hash", b"") or b""
                    if stage_use_cached and self._last_stage_kv_cache_hit:
                        kv_cache_hit = True
                        kv_cache_peer_id = candidate.peer_id
                    if onion_enabled:
                        expected_next_peer_id = (
                            self.pipeline[stage_index + 1].peer_id
                            if (stage_index + 1) < len(self.pipeline)
                            else ""
                        )
                        observed_next_peer_id = str(self._last_onion_next_peer_id or "")
                        if observed_next_peer_id != expected_next_peer_id:
                            raise RuntimeError(
                                "onion_route_mismatch:"
                                f"stage={stage_index};expected={expected_next_peer_id};observed={observed_next_peer_id}"
                            )
                        onion_layers_peeled += 1
                        onion_route_state = self._last_onion_route_state
                    stage_privacy = dict(self._last_privacy_audit or {})
                    if stage_privacy:
                        stage_privacy["stage_index"] = int(stage_index)
                        stage_privacy["peer_id"] = candidate.peer_id
                        stage_privacy["required"] = bool(
                            privacy_audit_required and stage_index < (len(self.pipeline) - 1)
                        )
                        expected_variance = max(0.0, float(getattr(candidate, "privacy_noise_variance", 0.0)))
                        stage_privacy["expected_variance"] = expected_variance
                        verified = True
                        if bool(stage_privacy["required"]):
                            configured = float(stage_privacy.get("configured_variance", 0.0))
                            applied = bool(stage_privacy.get("applied", False))
                            tag_valid = bool(stage_privacy.get("audit_tag_valid", False))
                            expected_floor = expected_variance if expected_variance > 0.0 else 1e-12
                            verified = applied and tag_valid and configured >= expected_floor
                            if not verified:
                                privacy_audit_violations.append(
                                    f"stage={stage_index};peer={candidate.peer_id};"
                                    f"applied={applied};tag_valid={tag_valid};"
                                    f"configured_variance={configured};expected={expected_floor}"
                                )
                        stage_privacy["verified"] = bool(verified)
                        privacy_audit_records.append(stage_privacy)
                    traces.append(
                        StageTrace(
                            peer_id=candidate.peer_id,
                            latency_ms=latency_ms,
                            stage_index=stage_index,
                            attempt=attempt,
                            failed_peer_id=failed_peer_id,
                        )
                    )
                    if compressed_transfer and self._autoencoder is not None:
                        compression_input += len(stage_input)
                        compression_latent += stage_latent_dim
                        compression_hops += 1
                    if encrypted_transfer:
                        encryption_hops += 1
                    break
                except (grpc.RpcError, RuntimeError) as exc:
                    failed_peer_id = candidate.peer_id
                    errors.append(f"{candidate.peer_id}: {exc}")
                    # PR-3 (B2) fast-fail: on an UNAVAILABLE / DEADLINE_EXCEEDED
                    # RpcError we publish a ``PEER_DEAD`` gossip event (via
                    # the coordinator-supplied callback) so the swarm can
                    # drop this peer from its routing tables *before* the
                    # next 60 s DHT tick. The local failover retry keeps
                    # running regardless — gossip is a complementary
                    # broadcast, not a replacement for the immediate
                    # single-request re-route.
                    _rpc_code_name = ""
                    if isinstance(exc, grpc.RpcError):
                        try:
                            _rpc_code_name = exc.code().name  # type: ignore[attr-defined]
                        except Exception:  # pragma: no cover — defensive
                            _rpc_code_name = ""
                    _is_dead_signal = _rpc_code_name in {
                        "UNAVAILABLE", "DEADLINE_EXCEEDED", "UNKNOWN"
                    }
                    if self._peer_dead_callback is not None and _is_dead_signal:
                        _target_libp2p = str(
                            getattr(candidate, "libp2p_peer_id", "") or ""
                        )
                        if _target_libp2p:
                            try:
                                self._peer_dead_callback(
                                    _target_libp2p,
                                    _rpc_code_name or "rpc_error",
                                )
                            except Exception:  # pragma: no cover — never derail
                                logging.debug(
                                    "peer_dead_callback_failed", exc_info=True
                                )
            else:
                detail = "; ".join(errors)
                raise RuntimeError(f"stage {stage_index} failed after retries: {detail}")

        decode_model_id: str | None = None
        last_stage = self.pipeline[-1]
        _backend = str(getattr(last_stage, "runtime_backend", "")).strip().lower()
        if _backend.startswith("pytorch") or _backend == "mlx":
            candidate_model = str(getattr(last_stage, "runtime_model_id", "")).strip()
            if not candidate_model:
                candidate_model = str(getattr(last_stage, "model_id", "")).strip()
            decode_model_id = candidate_model or None
        logging.info(
            "chain_decode: backend=%s runtime_model_id=%s decode_model_id=%s activation_len=%d",
            _backend, getattr(last_stage, "runtime_model_id", ""), decode_model_id, len(activation),
        )

        _t_decode_start = time.perf_counter()
        output = ModelShard.decode_text(
            activation,
            max_tokens=max_tokens,
            tokenizer_model_id=decode_model_id,
        )
        _t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000
        logging.info("PROFILE phase_D_decode_text=%.1fms", _t_decode_ms)
        if privacy_audit_violations:
            raise RuntimeError("privacy_audit_failed:" + ";".join(privacy_audit_violations))
        total_ms = (time.perf_counter() - started) * 1000.0
        ratio = float(compression_latent / compression_input) if compression_input else 1.0
        compression = {
            "enabled": self._autoencoder is not None,
            "method": ("tensor_autoencoder_mean_pool" if self._autoencoder is not None else "none"),
            "hops_compressed": compression_hops,
            "total_input_elements": compression_input,
            "total_latent_elements": compression_latent,
            "avg_compression_ratio": round(ratio, 6),
            "approx_reduction_pct": round((1.0 - ratio) * 100.0, 6),
        }
        encryption = {
            "enabled": self.advanced_encryption_enabled,
            "level": (self.advanced_encryption_level if self.advanced_encryption_enabled else "off"),
            "suite": (_suite_name(self.advanced_encryption_level) if self.advanced_encryption_enabled else "none"),
            "layers_per_hop": (
                required_layers_for_level(self.advanced_encryption_level)
                if self.advanced_encryption_enabled
                else 0
            ),
            "encrypted_hops": encryption_hops,
            "onion_routing": onion_enabled,
            "onion_layers": (len(self.pipeline) if onion_enabled else 0),
            "onion_layers_peeled": onion_layers_peeled,
            "privacy_audit_required": privacy_audit_required,
            "privacy_audit_verified": (
                all(bool(item.get("verified", True)) for item in privacy_audit_records if bool(item.get("required")))
                if privacy_audit_required
                else True
            ),
            "privacy_audit_records": privacy_audit_records,
        }
        kv = {
            "enabled": bool(kv_session),
            "session_id": (kv_session or None),
            "cache_stage_index": (int(kv_cache_stage_index) if kv_session else None),
            "cache_all_stages": bool(kv_session and kv_cache_all_stages),
            "cache_requested": bool(kv_session and kv_use_cached_activation),
            "cache_hit": bool(kv_session and kv_use_cached_activation and kv_cache_hit),
            "cache_peer_id": kv_cache_peer_id,
            "store_requested": bool(kv_session and kv_store_activation),
        }
        return ChainResult(
            request_id=rid,
            text=output,
            activation=activation,
            traces=traces,
            latency_ms=total_ms,
            compression=compression,
            encryption=encryption,
            kv=kv,
            activation_hash=bytes(_last_activation_hash),
        )

    # ── Push mode: server-to-server forwarding (Petals parity) ────────────

    def run_push(
        self,
        prompt: str,
        max_tokens: int = 1,
        request_id: str | None = None,
        deadline: float | None = None,
        kv_session_id: str | None = None,
        kv_store_activation: bool = False,
        kv_use_cached_activation: bool = False,
        callback_address: str = "",
        initial_activation: list[float] | None = None,
        **decode_controls: Any,
    ) -> ChainResult:
        """Execute the pipeline in push mode: peers forward directly to each other.

        Instead of the coordinator mediating every hop, the first peer
        forwards its output to the second peer, the second to the third,
        etc.  The last peer sends the final result back to the coordinator
        via the PushResult RPC.

        This eliminates N-1 coordinator round-trips per token.
        Falls back to ``run()`` if push mode fails.
        """
        from coordinator.push_receiver import register_push, await_push
        from peer import peer_pb2

        rid = request_id or str(uuid.uuid4())
        n = len(self.pipeline)
        t_start = time.perf_counter()

        if n < 2 or not callback_address:
            # Push mode needs 2+ stages and a callback address
            return self.run(
                prompt=prompt, max_tokens=max_tokens,
                request_id=rid, deadline=deadline,
                kv_session=kv_session_id,
                kv_store_activation=kv_store_activation,
                kv_use_cached_activation=kv_use_cached_activation,
                **decode_controls,
            )

        # Build the route for remaining hops
        route_hops = []
        for i, peer in enumerate(self.pipeline):
            route_hops.append(peer_pb2.PeerHop(
                peer_id=peer.peer_id,
                address=f"{peer.host}:{peer.port}",
                stage_index=i,
                shard_layer_start=int(getattr(peer, "layer_start", 0)),
                shard_layer_end=int(getattr(peer, "layer_end", 0)),
                shard_total_layers=int(getattr(peer, "total_layers", 0)),
                libp2p_peer_id=str(getattr(peer, "libp2p_peer_id", "") or ""),
            ))

        # Register the result callback
        future = register_push(rid)

        # Build request to the first peer with full route
        first_peer = self.pipeline[0]
        next_addr = f"{self.pipeline[1].host}:{self.pipeline[1].port}" if n > 1 else ""
        next_id = self.pipeline[1].peer_id if n > 1 else ""

        # Binary-pack the initial activation if provided.
        _push_activation: list[float] = []
        _push_packed = b""
        if initial_activation:
            # PR-1: vectorised pack (numpy fast path).
            from peer.activation_codec import pack_fp32 as _push_pack_fp32
            _push_packed = _push_pack_fp32(initial_activation)

        req = peer_pb2.ForwardRequest(
            request_id=rid,
            prompt=prompt,
            activation=_push_activation,
            activation_packed=_push_packed,
            stage_index=0,
            total_stages=n,
            max_tokens=max_tokens,
            kv_session_id=kv_session_id or "",
            kv_store_activation=kv_store_activation,
            kv_use_cached_activation=kv_use_cached_activation,
            decode_do_sample=bool(decode_controls.get("decode_do_sample", False)),
            decode_temperature=float(decode_controls.get("decode_temperature", 0.0) or 0.0),
            decode_top_p=float(decode_controls.get("decode_top_p", 0.0) or 0.0),
            decode_top_k=int(decode_controls.get("decode_top_k", 0) or 0),
            decode_seed=int(decode_controls.get("decode_seed", 0) or 0),
            shard_layer_start=int(getattr(first_peer, "layer_start", 0)),
            shard_layer_end=int(getattr(first_peer, "layer_end", 0)),
            shard_total_layers=int(getattr(first_peer, "total_layers", 0)),
            push_mode=True,
            next_hop_address=next_addr,
            next_hop_peer_id=next_id,
            final_callback_address=callback_address,
            final_callback_request_id=rid,
            final_callback_libp2p_peer_id=str(
                getattr(getattr(self, '_p2p_node', None), 'libp2p_peer_id', '') or ''
            ),
            remaining_route=route_hops[1:],  # Skip first peer (that's where we're sending)
        )

        # Send to first peer in a BACKGROUND THREAD to avoid deadlock.
        # When the coordinator IS the first peer (same process), a blocking
        # Forward() call would deadlock: the gRPC server can't process the
        # PushResult callback while Forward() is waiting for the chain.
        import threading as _push_threading

        def _send_push():
            try:
                _p2p = getattr(self, '_p2p_node', None)
                _libp2p_id = str(getattr(first_peer, 'libp2p_peer_id', '') or '').strip()
                # State-aware routing: check if direct connection exists.
                _has_direct = False
                if _p2p is not None and _libp2p_id:
                    try:
                        _has_direct = _p2p.is_peer_connected(_libp2p_id)
                    except Exception:
                        pass

                if _has_direct:
                    # Direct connection (DCUtR succeeded or same LAN) — use gRPC.
                    first_addr = f"{first_peer.host}:{first_peer.port}"
                    channel = grpc.insecure_channel(
                        first_addr,
                        options=[
                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ],
                    )
                    stub = peer_pb2_grpc.PeerStub(channel)
                    stub.Forward(req, timeout=min(self.timeout_s, 60.0))
                    channel.close()
                    logging.info("push_sent_direct: peer=%s", first_peer.peer_id)
                elif _p2p is not None and _libp2p_id:
                    # Fire-and-forget: ACK instantly, inference runs async.
                    _p2p.proxy_forward(
                        target_peer_id=_libp2p_id,
                        data=b'\x03' + req.SerializeToString(),  # 0x03 = fire-and-forget
                    )
                    logging.info("push_sent_via_relay: peer=%s libp2p=%s", first_peer.peer_id, _libp2p_id[:20])
                else:
                    # No P2P — direct gRPC only (LAN/VPC).
                    first_addr = f"{first_peer.host}:{first_peer.port}"
                    channel = grpc.insecure_channel(
                        first_addr,
                        options=[
                            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                            ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ],
                    )
                    stub = peer_pb2_grpc.PeerStub(channel)
                    stub.Forward(req, timeout=min(self.timeout_s, 60.0))
                    channel.close()
            except Exception as exc:
                logging.warning("push_send_failed: %s", exc)

        _push_thread = _push_threading.Thread(target=_send_push, daemon=True)
        _push_thread.start()

        # Wait for the final result from the last peer
        timeout_s = 120.0
        if deadline:
            timeout_s = max(1.0, deadline - time.perf_counter())
        try:
            result_response = await_push(rid, timeout_s=timeout_s)
        except Exception as exc:
            logging.warning("push_await_failed: %s: %s — falling back to run()", rid, exc)
            return self.run(
                prompt=prompt, max_tokens=max_tokens,
                request_id=rid, deadline=deadline,
                kv_session=kv_session_id,
                kv_store_activation=kv_store_activation,
                kv_use_cached_activation=kv_use_cached_activation,
                **decode_controls,
            )

        total_ms = (time.perf_counter() - t_start) * 1000
        # Prefer packed bytes from response (push result).
        _push_packed = bytes(getattr(result_response, "activation_packed", b"") or b"")
        if _push_packed and len(_push_packed) >= 8:
            import struct as _push_unpack
            _n = len(_push_packed) // 4
            activation = list(_push_unpack.unpack(f'<{_n}f', _push_packed))
        else:
            activation = list(result_response.activation) if hasattr(result_response, "activation") else []
        output = ""
        if activation:
            output = ModelShard.decode_text(
                activation,
                max_tokens=max_tokens,
                tokenizer_model_id=(
                    str(getattr(self.pipeline[-1], "runtime_model_id", "") or "").strip()
                    or None
                ),
            )

        logging.info(
            "push_chain_complete: req=%s stages=%d wall=%.0fms activation_len=%d",
            rid, n, total_ms, len(activation),
        )
        return ChainResult(
            request_id=rid,
            text=output,
            activation=activation,
            traces=[],
            latency_ms=total_ms,
            activation_hash=bytes(getattr(result_response, "activation_hash", b"")),
        )


    def run_push_ring(
        self,
        *,
        initial_activation: list[float],
        max_tokens: int,
        ring_eos_ids: list[int],
        kv_session_id: str = "",
        callback_address: str = "",
        request_id: str | None = None,
        **decode_controls,
    ) -> None:
        """Kick off a ring autoregressive push — fire-and-forget.

        Sends the initial ForwardRequest to the first peer with ring_mode=True.
        Tokens circulate peer-to-peer until max_tokens or EOS. Each token is
        emitted via emit_ring_token() to the coordinator's asyncio.Queue.
        """
        n = len(self.pipeline)
        rid = request_id or str(__import__("uuid").uuid4())

        route_hops = []
        for i, peer in enumerate(self.pipeline):
            _hop_libp2p = str(getattr(peer, "libp2p_peer_id", "") or "")
            logging.info(
                "ring_route_hop: stage=%d peer=%s libp2p=%s host=%s relay=%s",
                i, peer.peer_id, _hop_libp2p[:25] or "EMPTY", peer.host,
                getattr(peer, "requires_relay", "?"),
            )
            route_hops.append(peer_pb2.PeerHop(
                peer_id=peer.peer_id,
                address=f"{peer.host}:{peer.port}",
                stage_index=i,
                shard_layer_start=int(getattr(peer, "layer_start", 0)),
                shard_layer_end=int(getattr(peer, "layer_end", 0)),
                shard_total_layers=int(getattr(peer, "total_layers", 0)),
                libp2p_peer_id=_hop_libp2p,
            ))

        first_peer = self.pipeline[0]
        next_addr = f"{self.pipeline[1].host}:{self.pipeline[1].port}" if n > 1 else ""
        next_id = self.pipeline[1].peer_id if n > 1 else ""

        import struct as _ring_struct
        _ring_packed = _ring_struct.pack(f'<{len(initial_activation)}f', *initial_activation)

        req = peer_pb2.ForwardRequest(
            request_id=rid,
            prompt="",
            activation=[],
            activation_packed=_ring_packed,
            stage_index=0,
            total_stages=n,
            max_tokens=1,
            kv_session_id=kv_session_id,
            kv_store_activation=True,
            kv_use_cached_activation=False,  # First step = prefill
            decode_do_sample=bool(decode_controls.get("decode_do_sample", False)),
            decode_temperature=float(decode_controls.get("decode_temperature", 0.0) or 0.0),
            decode_top_p=float(decode_controls.get("decode_top_p", 0.0) or 0.0),
            decode_top_k=int(decode_controls.get("decode_top_k", 0) or 0),
            decode_seed=int(decode_controls.get("decode_seed", 0) or 0),
            shard_layer_start=int(getattr(first_peer, "layer_start", 0)),
            shard_layer_end=int(getattr(first_peer, "layer_end", 0)),
            shard_total_layers=int(getattr(first_peer, "total_layers", 0)),
            push_mode=True,
            ring_mode=True,
            ring_tokens_remaining=max_tokens,
            ring_generated_ids=[],
            ring_eos_ids=ring_eos_ids,
            ring_first_hop_address=f"{first_peer.host}:{first_peer.port}",
            ring_first_hop_peer_id=first_peer.peer_id,
            ring_first_hop_libp2p_id=str(getattr(first_peer, "libp2p_peer_id", "") or ""),
            ring_full_route=route_hops,
            next_hop_address=next_addr,
            next_hop_peer_id=next_id,
            final_callback_address=callback_address,
            final_callback_request_id=rid,
            final_callback_libp2p_peer_id=str(
                getattr(getattr(self, '_p2p_node', None), 'libp2p_peer_id', '') or ''
            ),
            remaining_route=route_hops[1:],
        )

        # Fire-and-forget in background thread (same as run_push).
        import threading as _ring_threading

        def _send_ring():
            try:
                _p2p = getattr(self, '_p2p_node', None)
                _libp2p_id = str(getattr(first_peer, 'libp2p_peer_id', '') or '').strip()
                _has_direct = False
                if _p2p is not None and _libp2p_id:
                    try:
                        _has_direct = _p2p.is_peer_connected(_libp2p_id)
                    except Exception:
                        pass
                if _has_direct:
                    first_addr = f"{first_peer.host}:{first_peer.port}"
                    channel = grpc.insecure_channel(first_addr, options=[
                        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                        ("grpc.max_send_message_length", 100 * 1024 * 1024),
                    ])
                    stub = peer_pb2_grpc.PeerStub(channel)
                    stub.Forward(req, timeout=60.0)
                    channel.close()
                elif _p2p is not None and _libp2p_id:
                    _p2p.proxy_forward(target_peer_id=_libp2p_id, data=b'\x01' + req.SerializeToString())
                else:
                    first_addr = f"{first_peer.host}:{first_peer.port}"
                    channel = grpc.insecure_channel(first_addr, options=[
                        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                        ("grpc.max_send_message_length", 100 * 1024 * 1024),
                    ])
                    stub = peer_pb2_grpc.PeerStub(channel)
                    stub.Forward(req, timeout=60.0)
                    channel.close()
            except Exception as exc:
                logging.warning("ring_push_send_failed: %s", exc)
                # Emit sentinel so the coordinator doesn't hang.
                from coordinator.push_receiver import emit_ring_token
                emit_ring_token(rid, None)

        _ring_thread = _ring_threading.Thread(target=_send_ring, daemon=True)
        _ring_thread.start()
        logging.info("ring_push_started: req=%s tokens=%d stages=%d", rid, max_tokens, n)


def _suite_name(level: str) -> str:
    normalized = str(level or "standard").strip().lower()
    if normalized == "standard":
        return "x25519_hkdf_sha256_aes256_gcm"
    return f"x25519_hkdf_sha256_aes256_gcm_onion_{normalized}"
