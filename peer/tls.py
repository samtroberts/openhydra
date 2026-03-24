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
