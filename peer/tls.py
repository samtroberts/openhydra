from __future__ import annotations

from pathlib import Path

import grpc


def load_server_credentials(
    *,
    cert_path: str,
    key_path: str,
    client_ca_path: str | None = None,
    require_client_auth: bool = False,
) -> grpc.ServerCredentials:
    cert_chain = Path(cert_path).read_bytes()
    private_key = Path(key_path).read_bytes()
    client_ca = Path(client_ca_path).read_bytes() if client_ca_path else None

    return grpc.ssl_server_credentials(
        private_key_certificate_chain_pairs=[(private_key, cert_chain)],
        root_certificates=client_ca,
        require_client_auth=require_client_auth,
    )
