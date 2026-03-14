from peer.tls import load_server_credentials


def test_load_server_credentials_calls_grpc(monkeypatch, tmp_path):
    cert = tmp_path / "server.cert.pem"
    key = tmp_path / "server.key.pem"
    ca = tmp_path / "ca.cert.pem"

    cert.write_bytes(b"cert-bytes")
    key.write_bytes(b"key-bytes")
    ca.write_bytes(b"ca-bytes")

    called: dict[str, object] = {}

    def fake_ssl_server_credentials(private_key_certificate_chain_pairs, root_certificates=None, require_client_auth=False):
        called["pairs"] = private_key_certificate_chain_pairs
        called["root"] = root_certificates
        called["require"] = require_client_auth
        return "server-creds"

    monkeypatch.setattr("peer.tls.grpc.ssl_server_credentials", fake_ssl_server_credentials)

    creds = load_server_credentials(
        cert_path=str(cert),
        key_path=str(key),
        client_ca_path=str(ca),
        require_client_auth=True,
    )

    assert creds == "server-creds"
    assert called["pairs"][0] == (b"key-bytes", b"cert-bytes")
    assert called["root"] == b"ca-bytes"
    assert called["require"] is True
