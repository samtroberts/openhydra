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

"""dht.signpost — Hivemind DHT signpost daemon for Linode Nanodes.

This is the entry-point for Hivemind DHT bootstrap nodes.  Each signpost
runs a ``hivemind.DHT`` node with ``host_maddrs`` but **no** ``initial_peers``
(because it IS the bootstrap).

The signpost is meant to run alongside the existing HTTP bootstrap service
(``dht/bootstrap.py``) on the same Nanode, providing a dual-stack DHT:
peers can discover each other via HTTP (legacy) or Hivemind Kademlia (new).

Usage
-----
::

    python -m dht.signpost --host 0.0.0.0 --port 38751

Or via systemd::

    [Service]
    ExecStart=/usr/bin/python3 -m dht.signpost --host 0.0.0.0 --port 38751
"""
from __future__ import annotations

import argparse
import logging
import signal
import threading
import time

logger = logging.getLogger(__name__)


def serve(
    host: str = "0.0.0.0",
    port: int = 38751,
    identity_path: str | None = None,
) -> None:
    """Start a Hivemind DHT signpost (bootstrap) node.

    This node runs with no initial_peers — other peers connect TO it.
    The node listens on ``/ip4/{host}/tcp/{port}``.

    Parameters
    ----------
    identity_path:
        Path to a persistent libp2p private key file.  If the file does not
        exist, hivemind creates it.  On subsequent starts the same peer ID
        is reused, keeping multiaddrs stable across restarts.
    """
    try:
        import hivemind
    except ImportError:
        logger.error(
            "signpost: hivemind not installed. "
            "Install with: pip install hivemind>=1.1.0"
        )
        return

    host_maddrs = [f"/ip4/{host}/tcp/{port}"]

    # Resolve default identity path.
    if identity_path is None:
        identity_path = "/opt/openhydra/.hivemind_identity.key"

    logger.info(
        "signpost: starting Hivemind DHT on %s (identity=%s)",
        host_maddrs[0], identity_path,
    )
    dht = hivemind.DHT(
        initial_peers=None,
        host_maddrs=host_maddrs,
        identity_path=identity_path,
        start=True,
    )
    peer_id = str(getattr(dht, "peer_id", "unknown"))
    logger.info("signpost: running peer_id=%s maddrs=%s", peer_id, host_maddrs)

    # Block until SIGTERM/SIGINT.
    stop = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        logger.info("signpost: shutdown signal=%d", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    try:
        while not stop.is_set():
            stop.wait(timeout=1.0)
    finally:
        dht.shutdown()
        logger.info("signpost: shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenHydra Hivemind DHT signpost (bootstrap) daemon"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=38751,
        help="TCP port for Hivemind Kademlia (default: 38751)",
    )
    parser.add_argument(
        "--identity-path",
        default="/opt/openhydra/.hivemind_identity.key",
        help=(
            "Path to a persistent libp2p identity key file. "
            "Created on first run; reused on restarts to keep peer ID stable. "
            "(default: /opt/openhydra/.hivemind_identity.key)"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    serve(host=args.host, port=args.port, identity_path=args.identity_path)


if __name__ == "__main__":
    main()
