# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Integration tests for CP-0: ZMQ IPC bridge + Python worker daemon.

Tests the full Rust ↔ Python IPC wire format round-trip using the Python
worker daemon (peer/zmq_worker.py) with a mock ModelShard.

Run:  pytest tests/test_zmq_worker.py -v
"""

from __future__ import annotations

import os
import socket
import struct
import tempfile
import threading
import time
from unittest.mock import MagicMock

import pytest

# ── Module imports ──────────────────────────────────────────────────────

def _worker_mod():
    from peer import zmq_worker
    return zmq_worker


def _cbor2():
    import cbor2
    return cbor2


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_socket_path() -> str:
    """Create a unique temporary socket path."""
    return os.path.join(
        tempfile.gettempdir(),
        f"openhydra-test-{os.getpid()}-{threading.current_thread().ident}.sock",
    )


def _encode_request_wire(header_dict: dict, activation: bytes) -> bytes:
    """Encode a forward request in the IPC wire format (Python side).

    Returns the full message including the outer msg_len prefix.
    """
    cbor2 = _cbor2()

    hdr_bytes = cbor2.dumps(header_dict)
    header_len = len(hdr_bytes)
    act_len = len(activation)

    inner = bytearray(4 + header_len + 4 + act_len)
    struct.pack_into("<I", inner, 0, header_len)
    inner[4:4 + header_len] = hdr_bytes
    struct.pack_into("<I", inner, 4 + header_len, act_len)
    inner[4 + header_len + 4:] = activation

    # Outer length prefix.
    msg_len = struct.pack("<I", len(inner))
    return msg_len + bytes(inner)


def _decode_response_wire(data: bytes) -> tuple[dict, bytes]:
    """Decode an IPC response from wire bytes (skipping outer msg_len)."""
    cbor2 = _cbor2()

    header_len = struct.unpack_from("<I", data, 0)[0]
    header = cbor2.loads(data[4:4 + header_len])

    act_offset = 4 + header_len
    act_len = struct.unpack_from("<I", data, act_offset)[0]
    act_start = act_offset + 4
    activation = data[act_start:act_start + act_len]

    return header, activation


def _floats_to_bytes(floats: list[float]) -> bytes:
    return struct.pack(f"<{len(floats)}f", *floats)


def _bytes_to_floats(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


class _MockShard:
    """Mock ModelShard that doubles all activation values."""

    def forward(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        **kwargs,
    ) -> list[float]:
        return [v * 2.0 for v in activation]


class _ErrorShard:
    """Mock ModelShard that always raises."""

    def forward(self, prompt, activation, max_tokens, **kwargs):
        raise RuntimeError("GPU on fire")


# ── Codec tests ─────────────────────────────────────────────────────────


class TestIpcCodec:
    """Test the Python IPC codec (decode_header + encode_response)."""

    def test_decode_header_minimal(self):
        zmq = _worker_mod()
        header_dict = {"request_id": "test-001"}
        activation = _floats_to_bytes([1.0, 2.0, 3.0])

        # Build inner wire bytes (without outer msg_len).
        cbor2 = _cbor2()
        hdr_bytes = cbor2.dumps(header_dict)
        inner = bytearray(4 + len(hdr_bytes) + 4 + len(activation))
        struct.pack_into("<I", inner, 0, len(hdr_bytes))
        inner[4:4 + len(hdr_bytes)] = hdr_bytes
        struct.pack_into("<I", inner, 4 + len(hdr_bytes), len(activation))
        inner[4 + len(hdr_bytes) + 4:] = activation

        header, act = zmq._decode_header(bytes(inner))
        assert header.request_id == "test-001"
        assert header.stage_index == 0
        assert header.total_stages == 1
        assert _bytes_to_floats(act) == [1.0, 2.0, 3.0]

    def test_decode_header_full_fields(self):
        zmq = _worker_mod()
        header_dict = {
            "request_id": "full-test",
            "stage_index": 2,
            "total_stages": 4,
            "push_mode": True,
            "shard_layer_start": 8,
            "shard_layer_end": 16,
            "shard_total_layers": 32,
            "kv_session_id": "session-xyz",
            "kv_store_activation": True,
            "kv_use_cached_activation": False,
            "decode_do_sample": True,
            "decode_temperature": 0.7,
            "decode_top_p": 0.9,
            "decode_top_k": 50,
            "ring_mode": True,
            "ring_tokens_remaining": 100,
            "prompt_token_ids": [1, 2, 3, 4],
            "activation_dtype": 0,
            "activation_shape": [1, 1, 896],
        }
        activation = _floats_to_bytes([0.5])

        cbor2 = _cbor2()
        hdr_bytes = cbor2.dumps(header_dict)
        inner = bytearray(4 + len(hdr_bytes) + 4 + len(activation))
        struct.pack_into("<I", inner, 0, len(hdr_bytes))
        inner[4:4 + len(hdr_bytes)] = hdr_bytes
        struct.pack_into("<I", inner, 4 + len(hdr_bytes), len(activation))
        inner[4 + len(hdr_bytes) + 4:] = activation

        header, act = zmq._decode_header(bytes(inner))
        assert header.request_id == "full-test"
        assert header.stage_index == 2
        assert header.total_stages == 4
        assert header.push_mode is True
        assert header.shard_layer_start == 8
        assert header.shard_layer_end == 16
        assert header.shard_total_layers == 32
        assert header.kv_session_id == "session-xyz"
        assert header.kv_store_activation is True
        assert header.decode_do_sample is True
        assert abs(header.decode_temperature - 0.7) < 1e-6
        assert abs(header.decode_top_p - 0.9) < 1e-6
        assert header.decode_top_k == 50
        assert header.ring_mode is True
        assert header.ring_tokens_remaining == 100
        assert header.prompt_token_ids == (1, 2, 3, 4)

    def test_decode_header_too_short(self):
        zmq = _worker_mod()
        with pytest.raises(ValueError, match="too short"):
            zmq._decode_header(b"\x00\x00")

    def test_decode_header_truncated(self):
        zmq = _worker_mod()
        # header_len says 100 but data is only 14 bytes.
        data = b"\x64\x00\x00\x00" + b"\x00" * 10
        with pytest.raises(ValueError, match="truncated"):
            zmq._decode_header(data)

    def test_encode_response_minimal(self):
        zmq = _worker_mod()
        header = zmq.IpcResponseHeader(
            request_id="resp-001",
            status=zmq.STATUS_OK,
        )
        resp_wire = zmq._encode_response(header, b"")
        resp_hdr, resp_act = _decode_response_wire(resp_wire)
        assert resp_hdr["request_id"] == "resp-001"
        assert resp_act == b""

    def test_encode_response_with_activation(self):
        zmq = _worker_mod()
        activation = _floats_to_bytes([1.0, 2.0, 3.0])
        header = zmq.IpcResponseHeader(
            request_id="resp-002",
            status=zmq.STATUS_OK,
            activation_dtype=zmq.DTYPE_FP32,
            activation_shape=(1, 1, 3),
        )
        resp_wire = zmq._encode_response(header, activation)
        resp_hdr, resp_act = _decode_response_wire(resp_wire)
        assert resp_hdr["request_id"] == "resp-002"
        assert _bytes_to_floats(resp_act) == [1.0, 2.0, 3.0]
        assert resp_hdr["activation_shape"] == [1, 1, 3]

    def test_encode_response_error(self):
        zmq = _worker_mod()
        header = zmq.IpcResponseHeader(
            request_id="err-001",
            status=zmq.STATUS_ERROR,
            error_message="model not loaded",
        )
        resp_wire = zmq._encode_response(header, b"")
        resp_hdr, _ = _decode_response_wire(resp_wire)
        assert resp_hdr["status"] == zmq.STATUS_ERROR
        assert resp_hdr["error_message"] == "model not loaded"


# ── Activation conversion tests ─────────────────────────────────────────


class TestActivationConversion:
    def test_floats_roundtrip_fp32(self):
        zmq = _worker_mod()
        values = [0.5, -0.3, 1.0, -1.0, 0.0, 42.5]
        raw = zmq._floats_to_activation(values, zmq.DTYPE_FP32)
        restored = zmq._activation_to_floats(raw, zmq.DTYPE_FP32)
        assert len(restored) == len(values)
        for orig, rec in zip(values, restored):
            assert abs(orig - rec) < 1e-6

    def test_floats_roundtrip_fp16(self):
        zmq = _worker_mod()
        values = [0.5, -0.25, 1.0, -1.0]
        raw = zmq._floats_to_activation(values, zmq.DTYPE_FP16)
        restored = zmq._activation_to_floats(raw, zmq.DTYPE_FP16)
        assert len(restored) == len(values)
        for orig, rec in zip(values, restored):
            assert abs(orig - rec) < 0.01

    def test_empty_activation(self):
        zmq = _worker_mod()
        raw = zmq._floats_to_activation([], zmq.DTYPE_FP32)
        assert raw == b""
        restored = zmq._activation_to_floats(b"", zmq.DTYPE_FP32)
        assert restored == []


# ── IPC integration tests ───────────────────────────────────────────────


class TestIpcIntegration:
    """End-to-end IPC tests: simulate the Rust bridge side in Python,
    connect the worker, and verify request/response flow.
    """

    def test_worker_forward_doubles_values(self):
        """Full round-trip: bridge sends activation, mock shard doubles it."""
        zmq = _worker_mod()
        sock_path = _make_socket_path()

        # Clean up stale socket.
        if os.path.exists(sock_path):
            os.unlink(sock_path)

        # Create a Unix listener (simulating the Rust bridge).
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(sock_path)
        listener.listen(1)
        listener.settimeout(5.0)

        stop_event = threading.Event()
        shard = _MockShard()

        # Start the worker in a background thread.
        worker_thread = threading.Thread(
            target=zmq.run_worker,
            kwargs={
                "socket_path": sock_path,
                "shard": shard,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        worker_thread.start()

        try:
            # Accept the worker connection.
            conn, _ = listener.accept()
            conn.settimeout(5.0)

            # Send a forward request.
            activation = _floats_to_bytes([1.0, 2.0, 3.0, 4.0])
            request = _encode_request_wire(
                {
                    "request_id": "ipc-test-001",
                    "stage_index": 0,
                    "total_stages": 2,
                    "shard_layer_start": 0,
                    "shard_layer_end": 16,
                    "shard_total_layers": 32,
                },
                activation,
            )
            conn.sendall(request)

            # Read response.
            len_buf = conn.recv(4)
            assert len(len_buf) == 4
            resp_len = struct.unpack("<I", len_buf)[0]
            resp_body = b""
            while len(resp_body) < resp_len:
                chunk = conn.recv(resp_len - len(resp_body))
                assert chunk, "connection closed"
                resp_body += chunk

            resp_hdr, resp_act = _decode_response_wire(resp_body)
            assert resp_hdr["request_id"] == "ipc-test-001"
            assert resp_hdr.get("status", 0) == 0  # STATUS_OK

            result_floats = _bytes_to_floats(resp_act)
            assert result_floats == [2.0, 4.0, 6.0, 8.0]

            conn.close()
        finally:
            stop_event.set()
            worker_thread.join(timeout=3.0)
            listener.close()
            if os.path.exists(sock_path):
                os.unlink(sock_path)

    def test_worker_error_response(self):
        """Worker sends error response when shard.forward() raises."""
        zmq = _worker_mod()
        sock_path = _make_socket_path()

        if os.path.exists(sock_path):
            os.unlink(sock_path)

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(sock_path)
        listener.listen(1)
        listener.settimeout(5.0)

        stop_event = threading.Event()
        shard = _ErrorShard()

        worker_thread = threading.Thread(
            target=zmq.run_worker,
            kwargs={
                "socket_path": sock_path,
                "shard": shard,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        worker_thread.start()

        try:
            conn, _ = listener.accept()
            conn.settimeout(5.0)

            activation = _floats_to_bytes([1.0])
            request = _encode_request_wire(
                {"request_id": "err-test-001"},
                activation,
            )
            conn.sendall(request)

            # Read error response.
            len_buf = conn.recv(4)
            resp_len = struct.unpack("<I", len_buf)[0]
            resp_body = b""
            while len(resp_body) < resp_len:
                chunk = conn.recv(resp_len - len(resp_body))
                assert chunk
                resp_body += chunk

            resp_hdr, _ = _decode_response_wire(resp_body)
            assert resp_hdr["request_id"] == "err-test-001"
            assert resp_hdr["status"] == 1  # STATUS_ERROR
            assert "GPU on fire" in resp_hdr["error_message"]

            conn.close()
        finally:
            stop_event.set()
            worker_thread.join(timeout=3.0)
            listener.close()
            if os.path.exists(sock_path):
                os.unlink(sock_path)

    def test_worker_multiple_requests(self):
        """Worker handles multiple sequential requests on one connection."""
        zmq = _worker_mod()
        sock_path = _make_socket_path()

        if os.path.exists(sock_path):
            os.unlink(sock_path)

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(sock_path)
        listener.listen(1)
        listener.settimeout(5.0)

        stop_event = threading.Event()
        shard = _MockShard()

        worker_thread = threading.Thread(
            target=zmq.run_worker,
            kwargs={
                "socket_path": sock_path,
                "shard": shard,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        worker_thread.start()

        try:
            conn, _ = listener.accept()
            conn.settimeout(5.0)

            for i in range(5):
                values = [float(i + j) for j in range(4)]
                activation = _floats_to_bytes(values)
                request = _encode_request_wire(
                    {"request_id": f"multi-{i}"},
                    activation,
                )
                conn.sendall(request)

                # Read response.
                len_buf = b""
                while len(len_buf) < 4:
                    len_buf += conn.recv(4 - len(len_buf))
                resp_len = struct.unpack("<I", len_buf)[0]
                resp_body = b""
                while len(resp_body) < resp_len:
                    chunk = conn.recv(resp_len - len(resp_body))
                    assert chunk
                    resp_body += chunk

                resp_hdr, resp_act = _decode_response_wire(resp_body)
                assert resp_hdr["request_id"] == f"multi-{i}"
                result = _bytes_to_floats(resp_act)
                expected = [v * 2.0 for v in values]
                assert result == expected, f"request {i}: {result} != {expected}"

            conn.close()
        finally:
            stop_event.set()
            worker_thread.join(timeout=3.0)
            listener.close()
            if os.path.exists(sock_path):
                os.unlink(sock_path)

    def test_worker_kv_cache_params_forwarded(self):
        """Verify that KV cache parameters are correctly forwarded to shard.forward()."""
        zmq = _worker_mod()
        sock_path = _make_socket_path()

        if os.path.exists(sock_path):
            os.unlink(sock_path)

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(sock_path)
        listener.listen(1)
        listener.settimeout(5.0)

        stop_event = threading.Event()

        # Spy shard that records kwargs.
        class _SpyShard:
            calls: list[dict] = []

            def forward(self, prompt, activation, max_tokens, **kwargs):
                self.calls.append(kwargs)
                return activation  # Echo back.

        shard = _SpyShard()
        shard.calls = []

        worker_thread = threading.Thread(
            target=zmq.run_worker,
            kwargs={
                "socket_path": sock_path,
                "shard": shard,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        worker_thread.start()

        try:
            conn, _ = listener.accept()
            conn.settimeout(5.0)

            activation = _floats_to_bytes([1.0])
            request = _encode_request_wire(
                {
                    "request_id": "kv-test",
                    "stage_index": 1,
                    "total_stages": 4,
                    "kv_session_id": "session-abc",
                    "kv_store_activation": True,
                    "kv_use_cached_activation": True,
                    "decode_do_sample": True,
                    "decode_temperature": 0.8,
                    "decode_top_k": 40,
                },
                activation,
            )
            conn.sendall(request)

            # Read response.
            len_buf = b""
            while len(len_buf) < 4:
                len_buf += conn.recv(4 - len(len_buf))
            resp_len = struct.unpack("<I", len_buf)[0]
            resp_body = b""
            while len(resp_body) < resp_len:
                chunk = conn.recv(resp_len - len(resp_body))
                assert chunk
                resp_body += chunk

            # Verify the shard received correct parameters.
            assert len(shard.calls) == 1
            call = shard.calls[0]
            assert call["stage_index"] == 1
            assert call["total_stages"] == 4
            assert call["kv_session_id"] == "session-abc"
            assert call["kv_store_activation"] is True
            assert call["kv_use_cached_activation"] is True
            assert call["decode_do_sample"] is True
            assert abs(call["decode_temperature"] - 0.8) < 1e-6
            assert call["decode_top_k"] == 40

            conn.close()
        finally:
            stop_event.set()
            worker_thread.join(timeout=3.0)
            listener.close()
            if os.path.exists(sock_path):
                os.unlink(sock_path)

    def test_ipc_latency_under_threshold(self):
        """Verify IPC round-trip overhead is under 1ms (pure codec, no model)."""
        zmq = _worker_mod()

        # Measure codec encode + decode round-trip.
        header_dict = {
            "request_id": "latency-test",
            "stage_index": 0,
            "total_stages": 2,
            "shard_layer_start": 0,
            "shard_layer_end": 16,
            "shard_total_layers": 32,
            "activation_dtype": 0,
            "activation_shape": [1, 1, 896],
        }

        # Simulate a 896-dim FP32 hidden state (~3.5KB).
        activation = _floats_to_bytes([0.1] * 896)

        cbor2 = _cbor2()

        # Warm up.
        for _ in range(10):
            hdr_bytes = cbor2.dumps(header_dict)
            inner = bytearray(4 + len(hdr_bytes) + 4 + len(activation))
            struct.pack_into("<I", inner, 0, len(hdr_bytes))
            inner[4:4 + len(hdr_bytes)] = hdr_bytes
            struct.pack_into("<I", inner, 4 + len(hdr_bytes), len(activation))
            inner[4 + len(hdr_bytes) + 4:] = activation
            zmq._decode_header(bytes(inner))

        # Measure.
        n_iterations = 1000
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            hdr_bytes = cbor2.dumps(header_dict)
            inner = bytearray(4 + len(hdr_bytes) + 4 + len(activation))
            struct.pack_into("<I", inner, 0, len(hdr_bytes))
            inner[4:4 + len(hdr_bytes)] = hdr_bytes
            struct.pack_into("<I", inner, 4 + len(hdr_bytes), len(activation))
            inner[4 + len(hdr_bytes) + 4:] = activation
            zmq._decode_header(bytes(inner))
        elapsed = time.perf_counter() - t0

        per_iter_us = (elapsed / n_iterations) * 1e6
        # Target: < 200μs per encode+decode (generous — actual should be <50μs).
        assert per_iter_us < 200, f"IPC codec too slow: {per_iter_us:.0f}μs/iter"
        # Log for visibility.
        print(f"\nIPC codec round-trip: {per_iter_us:.1f}μs/iter ({n_iterations} iterations)")


# ── Cross-compatibility tests ───────────────────────────────────────────


class TestCrossCompatibility:
    """Test that the Python codec can decode Rust-encoded messages and vice versa."""

    def test_rust_encoded_request_decoded_by_python(self):
        """Verify CBOR field names match between Rust and Python."""
        zmq = _worker_mod()
        cbor2 = _cbor2()

        # Simulate what Rust's ciborium would produce — a CBOR map with
        # the exact field names from ipc_codec.rs IpcForwardHeader.
        rust_header = {
            "request_id": "cross-compat-001",
            "stage_index": 1,
            "total_stages": 4,
            "push_mode": True,
            "shard_layer_start": 8,
            "shard_layer_end": 16,
            "shard_total_layers": 32,
            "kv_session_id": "sess-cross",
            "activation_dtype": 0,  # Fp32
            "activation_shape": [1, 1, 896],
            "ring_mode": True,
            "ring_tokens_remaining": 50,
        }
        activation = _floats_to_bytes([1.5, 2.5])

        hdr_bytes = cbor2.dumps(rust_header)
        inner = bytearray(4 + len(hdr_bytes) + 4 + len(activation))
        struct.pack_into("<I", inner, 0, len(hdr_bytes))
        inner[4:4 + len(hdr_bytes)] = hdr_bytes
        struct.pack_into("<I", inner, 4 + len(hdr_bytes), len(activation))
        inner[4 + len(hdr_bytes) + 4:] = activation

        header, act = zmq._decode_header(bytes(inner))
        assert header.request_id == "cross-compat-001"
        assert header.stage_index == 1
        assert header.total_stages == 4
        assert header.push_mode is True
        assert header.shard_layer_start == 8
        assert header.ring_mode is True
        assert header.ring_tokens_remaining == 50

    def test_python_encoded_response_has_correct_structure(self):
        """Verify the Python response wire format matches what Rust expects."""
        zmq = _worker_mod()
        cbor2 = _cbor2()

        activation = _floats_to_bytes([10.0, 20.0])
        header = zmq.IpcResponseHeader(
            request_id="resp-cross",
            status=zmq.STATUS_OK,
            activation_dtype=zmq.DTYPE_FP32,
            activation_shape=(1, 1, 2),
            metadata_json='{"elapsed_ms":5.2}',
        )
        resp_wire = zmq._encode_response(header, activation)

        # Parse manually — same as what Rust's decode_response would do.
        header_len = struct.unpack_from("<I", resp_wire, 0)[0]
        hdr_dict = cbor2.loads(resp_wire[4:4 + header_len])

        assert hdr_dict["request_id"] == "resp-cross"
        assert "status" not in hdr_dict  # STATUS_OK (0) is omitted
        assert hdr_dict["activation_shape"] == [1, 1, 2]
        assert hdr_dict["metadata_json"] == '{"elapsed_ms":5.2}'

        act_offset = 4 + header_len
        act_len = struct.unpack_from("<I", resp_wire, act_offset)[0]
        act_bytes = resp_wire[act_offset + 4:act_offset + 4 + act_len]
        assert _bytes_to_floats(act_bytes) == [10.0, 20.0]

    def test_unknown_cbor_fields_ignored(self):
        """Verify Python decoder ignores unknown CBOR fields (forward compat)."""
        zmq = _worker_mod()
        cbor2 = _cbor2()

        # Include a field that doesn't exist in the current schema.
        header_dict = {
            "request_id": "future-field-test",
            "some_future_field": "should be ignored",
            "another_future_field": 42,
        }
        activation = b""

        hdr_bytes = cbor2.dumps(header_dict)
        inner = bytearray(4 + len(hdr_bytes) + 4)
        struct.pack_into("<I", inner, 0, len(hdr_bytes))
        inner[4:4 + len(hdr_bytes)] = hdr_bytes
        struct.pack_into("<I", inner, 4 + len(hdr_bytes), 0)

        # Should not raise — unknown fields are silently ignored.
        header, act = zmq._decode_header(bytes(inner))
        assert header.request_id == "future-field-test"
        assert act == b""


# ── CP-4: Batch wire format tests ─────────────────────────────────────


class TestBatchWireFormat:
    """Test the batch IPC wire format (BATCH_MAGIC prefix)."""

    def test_batch_magic_detection(self):
        zmq = _worker_mod()
        # Raw batch message starts with BATCH_MAGIC.
        batch_prefix = struct.pack("<II", zmq.BATCH_MAGIC, 0)
        assert zmq._is_batch_message(batch_prefix)
        # Single message does NOT match BATCH_MAGIC.
        single_prefix = struct.pack("<I", 42)  # header_len = 42
        assert not zmq._is_batch_message(single_prefix)
        # Too short.
        assert not zmq._is_batch_message(b"\x00")

    def test_batch_decode_roundtrip(self):
        zmq = _worker_mod()
        cbor2 = _cbor2()

        h1 = {"request_id": "batch-1", "stage_index": 0, "shard_layer_start": 0}
        h2 = {"request_id": "batch-2", "stage_index": 1, "shard_layer_start": 16}
        act1 = _floats_to_bytes([1.0, 2.0])
        act2 = _floats_to_bytes([3.0, 4.0, 5.0])

        # Build batch wire format.
        items_wire = b""
        for hdr_dict, activation in [(h1, act1), (h2, act2)]:
            hdr_bytes = cbor2.dumps(hdr_dict)
            inner = bytearray(4 + len(hdr_bytes) + 4 + len(activation))
            struct.pack_into("<I", inner, 0, len(hdr_bytes))
            inner[4:4 + len(hdr_bytes)] = hdr_bytes
            struct.pack_into("<I", inner, 4 + len(hdr_bytes), len(activation))
            inner[4 + len(hdr_bytes) + 4:] = activation
            items_wire += bytes(inner)

        batch_wire = struct.pack("<II", zmq.BATCH_MAGIC, 2) + items_wire

        # Decode.
        items = zmq._decode_batch_request(batch_wire)
        assert len(items) == 2
        assert items[0][0].request_id == "batch-1"
        assert items[0][0].shard_layer_start == 0
        assert _bytes_to_floats(items[0][1]) == [1.0, 2.0]
        assert items[1][0].request_id == "batch-2"
        assert items[1][0].shard_layer_start == 16
        assert _bytes_to_floats(items[1][1]) == [3.0, 4.0, 5.0]

    def test_batch_response_encode(self):
        zmq = _worker_mod()
        cbor2 = _cbor2()

        resp1 = zmq.IpcResponseHeader(request_id="resp-1", status=zmq.STATUS_OK,
                                      activation_shape=(1, 1, 2))
        resp2 = zmq.IpcResponseHeader(request_id="resp-2", status=zmq.STATUS_OK,
                                      activation_shape=(1, 1, 3))
        act1 = _floats_to_bytes([10.0, 20.0])
        act2 = _floats_to_bytes([30.0, 40.0, 50.0])

        batch_resp = zmq._encode_batch_response([(resp1, act1), (resp2, act2)])

        # Verify batch magic.
        assert zmq._is_batch_message(batch_resp)
        batch_count = struct.unpack_from("<I", batch_resp, 4)[0]
        assert batch_count == 2

        # Parse items.
        offset = 8
        for expected_id, expected_act in [("resp-1", [10.0, 20.0]),
                                          ("resp-2", [30.0, 40.0, 50.0])]:
            hdr_len = struct.unpack_from("<I", batch_resp, offset)[0]
            hdr = cbor2.loads(batch_resp[offset + 4:offset + 4 + hdr_len])
            assert hdr["request_id"] == expected_id
            act_off = offset + 4 + hdr_len
            act_len = struct.unpack_from("<I", batch_resp, act_off)[0]
            act_bytes = batch_resp[act_off + 4:act_off + 4 + act_len]
            assert _bytes_to_floats(act_bytes) == expected_act
            offset = act_off + 4 + act_len

    def test_batch_worker_integration(self):
        """End-to-end: send a batch to the worker, verify all items processed."""
        zmq = _worker_mod()
        cbor2 = _cbor2()
        sock_path = _make_socket_path()

        if os.path.exists(sock_path):
            os.unlink(sock_path)

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(sock_path)
        listener.listen(1)
        listener.settimeout(5.0)

        stop_event = threading.Event()
        shard = _MockShard()

        worker_thread = threading.Thread(
            target=zmq.run_worker,
            kwargs={
                "socket_path": sock_path,
                "shard": shard,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        worker_thread.start()

        try:
            conn, _ = listener.accept()
            conn.settimeout(5.0)

            # Build a batch of 3 items.
            items_wire = b""
            expected_results = []
            for i in range(3):
                values = [float(i + j) for j in range(4)]
                expected_results.append([v * 2.0 for v in values])
                activation = _floats_to_bytes(values)
                hdr_dict = {"request_id": f"batch-{i}"}
                hdr_bytes = cbor2.dumps(hdr_dict)
                inner = bytearray(4 + len(hdr_bytes) + 4 + len(activation))
                struct.pack_into("<I", inner, 0, len(hdr_bytes))
                inner[4:4 + len(hdr_bytes)] = hdr_bytes
                struct.pack_into("<I", inner, 4 + len(hdr_bytes), len(activation))
                inner[4 + len(hdr_bytes) + 4:] = activation
                items_wire += bytes(inner)

            batch_wire = struct.pack("<II", zmq.BATCH_MAGIC, 3) + items_wire

            # Send with outer length prefix.
            msg_len = struct.pack("<I", len(batch_wire))
            conn.sendall(msg_len + batch_wire)

            # Read batch response.
            len_buf = b""
            while len(len_buf) < 4:
                len_buf += conn.recv(4 - len(len_buf))
            resp_len = struct.unpack("<I", len_buf)[0]
            resp_body = b""
            while len(resp_body) < resp_len:
                chunk = conn.recv(resp_len - len(resp_body))
                assert chunk, "connection closed"
                resp_body += chunk

            # Verify it's a batch response.
            assert zmq._is_batch_message(resp_body)
            batch_count = struct.unpack_from("<I", resp_body, 4)[0]
            assert batch_count == 3

            # Parse each response.
            offset = 8
            for i in range(3):
                hdr_len = struct.unpack_from("<I", resp_body, offset)[0]
                hdr = cbor2.loads(resp_body[offset + 4:offset + 4 + hdr_len])
                assert hdr["request_id"] == f"batch-{i}"

                act_off = offset + 4 + hdr_len
                act_len = struct.unpack_from("<I", resp_body, act_off)[0]
                act_bytes = resp_body[act_off + 4:act_off + 4 + act_len]
                result = _bytes_to_floats(act_bytes)
                assert result == expected_results[i], (
                    f"item {i}: {result} != {expected_results[i]}"
                )
                offset = act_off + 4 + act_len

            conn.close()
        finally:
            stop_event.set()
            worker_thread.join(timeout=3.0)
            listener.close()
            if os.path.exists(sock_path):
                os.unlink(sock_path)
