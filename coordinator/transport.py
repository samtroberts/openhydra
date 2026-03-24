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
from pathlib import Path

import grpc

GRPC_MAX_MESSAGE_BYTES = 100 * 1024 * 1024
GRPC_CHANNEL_OPTIONS: list[tuple[str, int]] = [
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
]


@dataclass(frozen=True)
class TransportConfig:
    tls_enabled: bool = False
    root_cert_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = None
    server_name_override: str | None = None

    def is_secure(self) -> bool:
        return self.tls_enabled or bool(self.root_cert_path or self.client_cert_path or self.client_key_path)

    def validate(self) -> None:
        has_client_cert = bool(self.client_cert_path)
        has_client_key = bool(self.client_key_path)
        if has_client_cert != has_client_key:
            raise ValueError("client_cert_path and client_key_path must be provided together")


def _read_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    return Path(path).read_bytes()


def create_channel(address: str, transport_config: TransportConfig | None = None) -> grpc.Channel:
    config = transport_config or TransportConfig()
    if not config.is_secure():
        return grpc.insecure_channel(address, options=GRPC_CHANNEL_OPTIONS)

    config.validate()
    root_cert = _read_bytes(config.root_cert_path)
    client_cert = _read_bytes(config.client_cert_path)
    client_key = _read_bytes(config.client_key_path)

    credentials = grpc.ssl_channel_credentials(
        root_certificates=root_cert,
        private_key=client_key,
        certificate_chain=client_cert,
    )

    options: list[tuple[str, str | int]] = list(GRPC_CHANNEL_OPTIONS)
    if config.server_name_override:
        options.append(("grpc.ssl_target_name_override", config.server_name_override))
        options.append(("grpc.default_authority", config.server_name_override))

    return grpc.secure_channel(address, credentials, options=options)
