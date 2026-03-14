import pytest

from coordinator.transport import TransportConfig, create_channel


class _DummyChannel:
    def __init__(self, address: str):
        self.address = address

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_create_channel_uses_insecure_by_default(monkeypatch):
    called: dict[str, str] = {}

    def fake_insecure(address, options=None):
        called["address"] = address
        called["options"] = str(options)
        return _DummyChannel(address)

    monkeypatch.setattr("coordinator.transport.grpc.insecure_channel", fake_insecure)

    channel = create_channel("127.0.0.1:50051", TransportConfig())
    assert isinstance(channel, _DummyChannel)
    assert called["address"] == "127.0.0.1:50051"
    assert "grpc.max_receive_message_length" in called["options"]


def test_create_channel_uses_secure_when_tls_enabled(monkeypatch, tmp_path):
    root = tmp_path / "ca.pem"
    cert = tmp_path / "client.pem"
    key = tmp_path / "client.key"
    root.write_bytes(b"ca")
    cert.write_bytes(b"cert")
    key.write_bytes(b"key")

    called: dict[str, object] = {}

    def fake_creds(root_certificates=None, private_key=None, certificate_chain=None):
        called["root"] = root_certificates
        called["key"] = private_key
        called["cert"] = certificate_chain
        return "creds"

    def fake_secure(address, credentials, options=None):
        called["address"] = address
        called["credentials"] = credentials
        called["options"] = options
        return _DummyChannel(address)

    monkeypatch.setattr("coordinator.transport.grpc.ssl_channel_credentials", fake_creds)
    monkeypatch.setattr("coordinator.transport.grpc.secure_channel", fake_secure)

    cfg = TransportConfig(
        tls_enabled=True,
        root_cert_path=str(root),
        client_cert_path=str(cert),
        client_key_path=str(key),
        server_name_override="peer-a",
    )

    channel = create_channel("127.0.0.1:50051", cfg)
    assert isinstance(channel, _DummyChannel)
    assert called["address"] == "127.0.0.1:50051"
    assert called["credentials"] == "creds"
    assert ("grpc.ssl_target_name_override", "peer-a") in called["options"]


def test_transport_config_requires_cert_and_key_together():
    cfg = TransportConfig(tls_enabled=True, client_cert_path="/tmp/cert.pem", client_key_path=None)
    with pytest.raises(ValueError, match="client_cert_path and client_key_path"):
        cfg.validate()
