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
