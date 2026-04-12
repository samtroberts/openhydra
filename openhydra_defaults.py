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

"""
openhydra_defaults.py — well-known production defaults for OpenHydra.

Bootstrap nodes are the DHT phone book: peers announce into them, coordinators
query them for peer discovery.  They do not participate in inference.

These three nodes are geographically distributed (US-East, EU-Central,
AP-South) so that the closest one is always fast.  Any peer or coordinator
that does not explicitly pass --dht-url will use all three automatically.

Operators running private networks can override every default here via CLI
flags; passing even one --dht-url flag replaces the entire default list.
"""

# ---------------------------------------------------------------------------
# DHT bootstrap nodes
# ---------------------------------------------------------------------------

PRODUCTION_BOOTSTRAP_URLS: tuple[str, ...] = (
    "http://bootstrap-us.openhydra.co:8468",
    "http://bootstrap-eu.openhydra.co:8468",
    "http://bootstrap-ap.openhydra.co:8468",
)

# gRPC relay port for NAT-traversal. The relay service runs on the same
# hosts as the DHT bootstrap but on a separate port. NATted peers connect
# outbound to the relay and coordinators proxy Forward() calls through it.
DEFAULT_RELAY_PORT: int = 50052

# ---------------------------------------------------------------------------
# libp2p bootstrap peers (Kademlia DHT + Circuit Relay v2)
# ---------------------------------------------------------------------------
# Same three Linode servers, running the Rust openhydra-bootstrap binary
# on TCP port 4001. These provide Kademlia DHT bootstrap, Circuit Relay v2
# (for NATted peers), and AutoNAT probing.
#
# Format: /ip4/<ip>/tcp/4001/p2p/<PeerId>
# Passing even one --p2p-bootstrap flag replaces the entire default list.

PRODUCTION_LIBP2P_BOOTSTRAP_PEERS: tuple[str, ...] = (
    # US (Dallas)
    "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb",
    # EU (London)
    "/ip4/172.105.69.49/tcp/4001/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT",
    # AP (Singapore)
    "/ip4/172.104.164.98/tcp/4001/p2p/12D3KooWPgqZBgLZ1f94AQ7sbeyEz5UJ4jiT4d3zuQp2t61VLPZo",
)
