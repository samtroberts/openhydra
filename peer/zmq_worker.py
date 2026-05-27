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

"""ZMQ IPC worker daemon — Python side of the Rust ↔ Python IPC bridge.

Replaces the ``_fwd_worker`` thread inside ``_proxy_handler_loop`` (server.py).
Connects to the Rust IPC bridge over a Unix domain socket and processes
forward requests using the existing ModelShard/MLXRuntime/PyTorchRuntime.

Wire format (matches ``network/src/ipc_codec.rs``):

  Request (Rust → Python):
    [0:4]     msg_len        (u32 LE — total bytes of the inner payload)
    [4:4+M]   inner          (IPC forward request):
      [0:4]     header_len   (u32 LE)
      [4:4+H]   header       (CBOR-encoded IpcForwardHeader)
      [4+H:4+H+4] act_len   (u32 LE)
      [4+H+4:..] activation (raw bytes)

  Response (Python → Rust):
    [0:4]     msg_len        (u32 LE)
    [4:4+M]   inner          (IPC response):
      [0:4]     header_len   (u32 LE)
      [4:4+H]   header       (CBOR-encoded IpcResponseHeader)
      [4+H:4+H+4] act_len   (u32 LE)
      [4+H+4:..] activation (raw bytes)

Uses ``cbor2`` for CBOR decoding/encoding (pip install cbor2).
"""

from __future__ import annotations

import gc
import logging
import os
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── Status codes ────────────────────────────────────────────────────────
STATUS_OK = 0
STATUS_ERROR = 1
STATUS_KV_CACHE_HIT = 2

# ── Activation dtype tags ───────────────────────────────────────────────
DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_INT8 = 2

# ── Batch wire format magic (must match BATCH_MAGIC in ipc_codec.rs) ───
BATCH_MAGIC = 0x48435442  # "BTCH" as u32 LE


@dataclass(frozen=True)
class IpcForwardHeader:
    """Decoded IPC forward request header."""

    request_id: str = ""
    stage_index: int = 0
    total_stages: int = 1
    push_mode: bool = False
    next_hop_address: str = ""
    next_hop_peer_id: str = ""

    # Layer sharding
    shard_layer_start: int = 0
    shard_layer_end: int = 0
    shard_total_layers: int = 0

    # KV cache
    kv_session_id: str = ""
    kv_store_activation: bool = False
    kv_use_cached_activation: bool = False
    kv_rollback_to: int = 0

    # Decode parameters
    decode_do_sample: bool = False
    decode_temperature: float = 0.0
    decode_top_p: float = 0.0
    decode_top_k: int = 0
    decode_seed: int = 0
    sample_on_coordinator: bool = False

    # Activation metadata
    activation_dtype: int = DTYPE_FP32
    activation_shape: tuple[int, ...] = ()
    quantized_scales: bytes = b""

    # Pipeline / speculative
    slot_id: int = 0
    pipeline_depth: int = 0
    draft_block: bool = False
    block_index: int = 0
    draft_token_ids: tuple[int, ...] = ()
    verify_batch_size: int = 0

    # Ring autoregressive
    ring_mode: bool = False
    ring_tokens_remaining: int = 0
    ring_generated_ids: tuple[int, ...] = ()
    ring_eos_ids: tuple[int, ...] = ()
    ring_first_hop_address: str = ""
    ring_first_hop_peer_id: str = ""
    ring_first_hop_libp2p_id: str = ""
    ring_full_route: bytes = b""

    # Callback routing
    final_callback_address: str = ""
    final_callback_request_id: str = ""
    final_callback_libp2p_peer_id: str = ""
    remaining_route: bytes = b""

    # Prompt (stage 0 only)
    prompt_token_ids: tuple[int, ...] = ()

    # Encryption (pass-through)
    encryption_suite: int = 0
    encryption_nonces: bytes = b""
    encryption_ephemeral_keys: bytes = b""


@dataclass(frozen=True)
class IpcResponseHeader:
    """IPC response header."""

    request_id: str = ""
    status: int = STATUS_OK
    activation_dtype: int = DTYPE_FP32
    activation_shape: tuple[int, ...] = ()
    metadata_json: str = ""
    error_message: str = ""


def _decode_header(data: bytes) -> tuple[IpcForwardHeader, bytes]:
    """Decode an IPC forward request from wire bytes.

    Returns (header, activation_bytes).
    """
    import cbor2

    if len(data) < 8:
        raise ValueError(f"IPC request too short: {len(data)} bytes")

    header_len = struct.unpack_from("<I", data, 0)[0]
    if len(data) < 4 + header_len + 4:
        raise ValueError(
            f"IPC request truncated: need {4 + header_len + 4}, have {len(data)}"
        )

    raw_header = cbor2.loads(data[4 : 4 + header_len])

    act_offset = 4 + header_len
    act_len = struct.unpack_from("<I", data, act_offset)[0]
    act_start = act_offset + 4
    act_end = act_start + act_len
    if len(data) < act_end:
        raise ValueError(
            f"IPC activation truncated: declared {act_len}, have {len(data) - act_start}"
        )

    activation = data[act_start:act_end]

    # Map CBOR dict to IpcForwardHeader fields.
    header = IpcForwardHeader(
        request_id=raw_header.get("request_id", ""),
        stage_index=raw_header.get("stage_index", 0),
        total_stages=raw_header.get("total_stages", 1),
        push_mode=raw_header.get("push_mode", False),
        next_hop_address=raw_header.get("next_hop_address", ""),
        next_hop_peer_id=raw_header.get("next_hop_peer_id", ""),
        shard_layer_start=raw_header.get("shard_layer_start", 0),
        shard_layer_end=raw_header.get("shard_layer_end", 0),
        shard_total_layers=raw_header.get("shard_total_layers", 0),
        kv_session_id=raw_header.get("kv_session_id", ""),
        kv_store_activation=raw_header.get("kv_store_activation", False),
        kv_use_cached_activation=raw_header.get("kv_use_cached_activation", False),
        kv_rollback_to=raw_header.get("kv_rollback_to", 0),
        decode_do_sample=raw_header.get("decode_do_sample", False),
        decode_temperature=raw_header.get("decode_temperature", 0.0),
        decode_top_p=raw_header.get("decode_top_p", 0.0),
        decode_top_k=raw_header.get("decode_top_k", 0),
        decode_seed=raw_header.get("decode_seed", 0),
        sample_on_coordinator=raw_header.get("sample_on_coordinator", False),
        activation_dtype=raw_header.get("activation_dtype", DTYPE_FP32),
        activation_shape=tuple(raw_header.get("activation_shape", ())),
        quantized_scales=bytes(raw_header.get("quantized_scales", b"")),
        slot_id=raw_header.get("slot_id", 0),
        pipeline_depth=raw_header.get("pipeline_depth", 0),
        draft_block=raw_header.get("draft_block", False),
        block_index=raw_header.get("block_index", 0),
        draft_token_ids=tuple(raw_header.get("draft_token_ids", ())),
        verify_batch_size=raw_header.get("verify_batch_size", 0),
        ring_mode=raw_header.get("ring_mode", False),
        ring_tokens_remaining=raw_header.get("ring_tokens_remaining", 0),
        ring_generated_ids=tuple(raw_header.get("ring_generated_ids", ())),
        ring_eos_ids=tuple(raw_header.get("ring_eos_ids", ())),
        ring_first_hop_address=raw_header.get("ring_first_hop_address", ""),
        ring_first_hop_peer_id=raw_header.get("ring_first_hop_peer_id", ""),
        ring_first_hop_libp2p_id=raw_header.get("ring_first_hop_libp2p_id", ""),
        ring_full_route=bytes(raw_header.get("ring_full_route", b"")),
        final_callback_address=raw_header.get("final_callback_address", ""),
        final_callback_request_id=raw_header.get("final_callback_request_id", ""),
        final_callback_libp2p_peer_id=raw_header.get("final_callback_libp2p_peer_id", ""),
        remaining_route=bytes(raw_header.get("remaining_route", b"")),
        prompt_token_ids=tuple(raw_header.get("prompt_token_ids", ())),
        encryption_suite=raw_header.get("encryption_suite", 0),
        encryption_nonces=bytes(raw_header.get("encryption_nonces", b"")),
        encryption_ephemeral_keys=bytes(raw_header.get("encryption_ephemeral_keys", b"")),
    )

    return header, activation


def _encode_response(header: IpcResponseHeader, activation: bytes) -> bytes:
    """Encode an IPC response into wire format."""
    import cbor2

    hdr_dict: dict[str, Any] = {"request_id": header.request_id}

    if header.status != STATUS_OK:
        hdr_dict["status"] = header.status
    if header.activation_dtype != DTYPE_FP32:
        hdr_dict["activation_dtype"] = header.activation_dtype
    if header.activation_shape:
        hdr_dict["activation_shape"] = list(header.activation_shape)
    if header.metadata_json:
        hdr_dict["metadata_json"] = header.metadata_json
    if header.error_message:
        hdr_dict["error_message"] = header.error_message

    hdr_bytes = cbor2.dumps(hdr_dict)
    header_len = len(hdr_bytes)
    act_len = len(activation)

    buf = bytearray(4 + header_len + 4 + act_len)
    struct.pack_into("<I", buf, 0, header_len)
    buf[4 : 4 + header_len] = hdr_bytes
    struct.pack_into("<I", buf, 4 + header_len, act_len)
    buf[4 + header_len + 4 :] = activation

    return bytes(buf)


def _is_batch_message(data: bytes) -> bool:
    """Check if wire bytes are a batch message (BATCH_MAGIC prefix)."""
    if len(data) < 4:
        return False
    magic = struct.unpack_from("<I", data, 0)[0]
    return magic == BATCH_MAGIC


def _decode_batch_request(
    data: bytes,
) -> list[tuple[IpcForwardHeader, bytes]]:
    """Decode a batch of IPC forward requests from wire bytes.

    Returns a list of (header, activation_bytes) pairs.
    """
    if len(data) < 8:
        raise ValueError(f"Batch request too short: {len(data)} bytes")

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != BATCH_MAGIC:
        raise ValueError(f"Invalid batch magic: {magic:#010x}")

    batch_count = struct.unpack_from("<I", data, 4)[0]
    items: list[tuple[IpcForwardHeader, bytes]] = []
    offset = 8

    for i in range(batch_count):
        if offset >= len(data):
            raise ValueError(
                f"Batch request truncated at item {i}/{batch_count}"
            )
        # Each item is a standard single-request encoding.
        header, activation = _decode_header(data[offset:])
        # Compute item wire length to advance offset.
        header_len = struct.unpack_from("<I", data, offset)[0]
        act_offset = offset + 4 + header_len
        act_len = struct.unpack_from("<I", data, act_offset)[0]
        item_len = 4 + header_len + 4 + act_len
        offset += item_len
        items.append((header, activation))

    return items


def _encode_batch_response(
    items: list[tuple[IpcResponseHeader, bytes]],
) -> bytes:
    """Encode a batch of IPC responses into the batch wire format."""
    parts = [struct.pack("<II", BATCH_MAGIC, len(items))]
    for header, activation in items:
        parts.append(_encode_response(header, activation))
    return b"".join(parts)


def _activation_to_floats(
    activation_bytes: bytes,
    dtype: int = DTYPE_FP32,
) -> list[float]:
    """Convert raw activation bytes to a list of floats."""
    if dtype == DTYPE_FP32:
        n_floats = len(activation_bytes) // 4
        return list(struct.unpack(f"<{n_floats}f", activation_bytes))
    elif dtype == DTYPE_FP16:
        # FP16: 2 bytes per float.
        n_floats = len(activation_bytes) // 2
        return list(struct.unpack(f"<{n_floats}e", activation_bytes))
    else:
        # INT8 or other — pass raw bytes (caller must handle).
        return list(activation_bytes)


def _floats_to_activation(
    floats: list[float],
    dtype: int = DTYPE_FP32,
) -> bytes:
    """Convert a list of floats to raw activation bytes."""
    if dtype == DTYPE_FP32:
        return struct.pack(f"<{len(floats)}f", *floats)
    elif dtype == DTYPE_FP16:
        return struct.pack(f"<{len(floats)}e", *floats)
    else:
        return bytes(int(f) & 0xFF for f in floats)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket, raising on EOF."""
    buf = bytearray(n)
    view = memoryview(buf)
    total = 0
    while total < n:
        nbytes = sock.recv_into(view[total:])
        if nbytes == 0:
            raise ConnectionError(
                f"IPC socket closed after {total}/{n} bytes"
            )
        total += nbytes
    return bytes(buf)


def run_worker(
    socket_path: str,
    shard: Any,
    *,
    stop_event: Any | None = None,
    gpu_keepalive: bool = False,
    keepalive_interval_s: float = 0.005,
    keepalive_idle_threshold_s: float = 0.100,
    keepalive_gil_switch_s: float = 0.0005,
) -> None:
    """Main worker loop — connects to the Rust IPC bridge and processes
    forward requests.

    Args:
        socket_path: Path to the Unix domain socket bound by the Rust bridge.
        shard: A ModelShard instance with a .forward() method.
        stop_event: Threading event to signal shutdown.
        gpu_keepalive: Enable MLX GPU keep-alive busy-polling.
        keepalive_interval_s: Busy-poll interval for GPU keep-alive.
        keepalive_idle_threshold_s: Idle threshold before disabling keep-alive.
        keepalive_gil_switch_s: GIL switch interval during active inference.
    """
    import threading

    if stop_event is None:
        stop_event = threading.Event()

    logger.info("zmq_worker: connecting to %s", socket_path)

    # Connect to the Rust IPC bridge.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2.0)  # For initial connect.

    # Retry connection for up to 10 seconds (Rust bridge may still be binding).
    connected = False
    for attempt in range(20):
        try:
            sock.connect(socket_path)
            connected = True
            break
        except (ConnectionRefusedError, FileNotFoundError):
            if stop_event.is_set():
                return
            time.sleep(0.5)

    if not connected:
        logger.error("zmq_worker: failed to connect to %s after 10s", socket_path)
        return

    logger.info("zmq_worker: connected to IPC bridge")
    sock.settimeout(1.0)  # Timeout for recv during normal operation.

    # ── GPU keep-alive state ────────────────────────────────────────────
    _last_forward_t: float = 0.0
    _observed_gap_ema: float = 0.0
    _keepalive_active = False
    _gc_disabled = False
    _default_switch = sys.getswitchinterval()

    try:
        while not stop_event.is_set():
            # ── GPU keep-alive spin ─────────────────────────────────────
            if gpu_keepalive and _last_forward_t > 0:
                _since = time.perf_counter() - _last_forward_t
                _effective_threshold = max(
                    keepalive_idle_threshold_s, 3.0 * _observed_gap_ema
                )
                if _since < _effective_threshold:
                    if not _keepalive_active:
                        sys.setswitchinterval(keepalive_gil_switch_s)
                        _keepalive_active = True
                        if not _gc_disabled:
                            gc.disable()
                            _gc_disabled = True
                else:
                    if _keepalive_active:
                        sys.setswitchinterval(_default_switch)
                        _keepalive_active = False
                        if _gc_disabled:
                            gc.enable()
                            gc.collect()
                            _gc_disabled = False

            # ── Read a request ──────────────────────────────────────────
            try:
                len_buf = _recv_exactly(sock, 4)
            except socket.timeout:
                continue
            except ConnectionError:
                logger.info("zmq_worker: IPC connection closed")
                break

            msg_len = struct.unpack("<I", len_buf)[0]
            if msg_len > 100 * 1024 * 1024:
                logger.error("zmq_worker: message too large: %d bytes", msg_len)
                break

            try:
                body = _recv_exactly(sock, msg_len)
            except ConnectionError:
                logger.info("zmq_worker: IPC connection closed mid-read")
                break

            # ── Decode and process ──────────────────────────────────────
            # CP-4: detect batch vs single request.
            if _is_batch_message(body):
                # ── Batch path ─────────────────────────────────────────
                t0 = time.perf_counter()
                try:
                    batch_items = _decode_batch_request(body)
                except Exception as e:
                    logger.error("zmq_worker: batch decode failed: %s", e)
                    err_resp = _encode_batch_response([
                        (IpcResponseHeader(
                            request_id="unknown",
                            status=STATUS_ERROR,
                            error_message=str(e),
                        ), b""),
                    ])
                    _send_response(sock, err_resp)
                    continue

                batch_responses: list[tuple[IpcResponseHeader, bytes]] = []
                for header, activation_bytes in batch_items:
                    batch_responses.append(
                        _process_single_forward(shard, header, activation_bytes)
                    )

                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    "zmq_worker: batch done count=%d elapsed=%.1fms",
                    len(batch_items),
                    elapsed_ms,
                )

                resp_wire = _encode_batch_response(batch_responses)
                _send_response(sock, resp_wire)

            else:
                # ── Single-item path (legacy, unchanged) ───────────────
                t0 = time.perf_counter()
                try:
                    header, activation_bytes = _decode_header(body)
                except Exception as e:
                    logger.error("zmq_worker: decode failed: %s", e)
                    err_resp = _encode_response(
                        IpcResponseHeader(
                            request_id="unknown",
                            status=STATUS_ERROR,
                            error_message=str(e),
                        ),
                        b"",
                    )
                    _send_response(sock, err_resp)
                    continue

                resp_header, result_bytes = _process_single_forward(
                    shard, header, activation_bytes
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.debug(
                    "zmq_worker: forward done req=%s elapsed=%.1fms",
                    resp_header.request_id,
                    elapsed_ms,
                )
                resp_wire = _encode_response(resp_header, result_bytes)
                _send_response(sock, resp_wire)

            # ── GPU keep-alive tracking ─────────────────────────────────
            _now = time.perf_counter()
            if _last_forward_t > 0:
                _gap = _now - _last_forward_t
                if _gap < 2.0:
                    _alpha = 0.3
                    _observed_gap_ema = _alpha * _gap + (1 - _alpha) * _observed_gap_ema
            _last_forward_t = _now

    except KeyboardInterrupt:
        logger.info("zmq_worker: interrupted")
    finally:
        # Restore GIL interval.
        if _keepalive_active:
            sys.setswitchinterval(_default_switch)
        if _gc_disabled:
            gc.enable()
        sock.close()
        logger.info("zmq_worker: stopped")


def _process_single_forward(
    shard: Any,
    header: IpcForwardHeader,
    activation_bytes: bytes,
) -> tuple[IpcResponseHeader, bytes]:
    """Process a single forward request and return (response_header, result_bytes).

    Extracted from the main loop to support both single and batch paths.
    """
    try:
        activation_floats = _activation_to_floats(
            activation_bytes, header.activation_dtype
        )

        result = shard.forward(
            prompt="",  # Not used for sharded decode.
            activation=activation_floats,
            max_tokens=1,
            stage_index=header.stage_index,
            total_stages=header.total_stages,
            kv_session_id=header.kv_session_id or None,
            kv_store_activation=header.kv_store_activation,
            kv_use_cached_activation=header.kv_use_cached_activation,
            request_id=header.request_id or None,
            decode_do_sample=header.decode_do_sample or None,
            decode_temperature=header.decode_temperature or None,
            decode_top_p=header.decode_top_p or None,
            decode_top_k=header.decode_top_k or None,
            decode_seed=header.decode_seed or None,
            prompt_token_ids=(
                list(header.prompt_token_ids) if header.prompt_token_ids else None
            ),
        )

        result_floats = list(result) if not isinstance(result, list) else result
        result_bytes = _floats_to_activation(result_floats, DTYPE_FP32)

        resp_header = IpcResponseHeader(
            request_id=header.request_id,
            status=STATUS_OK,
            activation_dtype=DTYPE_FP32,
            activation_shape=(1, 1, len(result_floats)),
        )
        return resp_header, result_bytes

    except Exception as e:
        logger.error(
            "zmq_worker: forward failed req=%s err=%s",
            header.request_id,
            e,
            exc_info=True,
        )
        return (
            IpcResponseHeader(
                request_id=header.request_id,
                status=STATUS_ERROR,
                error_message=str(e),
            ),
            b"",
        )


def _send_response(sock: socket.socket, resp_wire: bytes) -> None:
    """Send a length-prefixed response over the IPC socket."""
    resp_len = struct.pack("<I", len(resp_wire))
    sock.sendall(resp_len + resp_wire)


def main() -> None:
    """CLI entry point for standalone worker testing."""
    import argparse

    parser = argparse.ArgumentParser(description="OpenHydra ZMQ IPC Worker")
    parser.add_argument(
        "--socket-path",
        required=True,
        help="Path to the Unix domain socket",
    )
    parser.add_argument(
        "--model-id",
        default="tinyllama-15M",
        help="Model ID for ModelShard",
    )
    parser.add_argument(
        "--runtime-backend",
        default="toy_auto",
        help="Runtime backend",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from peer.model_shard import ModelShard, ToyShardConfig

    config = ToyShardConfig(
        model_id=args.model_id,
        runtime_backend=args.runtime_backend,
    )
    shard = ModelShard(config)

    run_worker(args.socket_path, shard)


if __name__ == "__main__":
    main()
