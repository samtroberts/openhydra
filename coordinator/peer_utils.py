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

"""Shared utilities for PeerEndpoint construction and normalization.

Canonical implementations of tag/layer-index normalization, previously
duplicated across coordinator/path_finder.py and dht/bootstrap.py.
"""

from __future__ import annotations


def normalize_tags(raw: object) -> tuple[str, ...]:
    """Deduplicate and lowercase a tags field (str, list, tuple, or None).

    Returns a tuple of unique, non-empty, lowercased strings preserving
    insertion order.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        try:
            values = [str(item).strip() for item in list(raw)]
        except TypeError:
            values = []
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        norm = value.lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return tuple(deduped)


def normalize_layer_indices(raw: object) -> tuple[int, ...]:
    """Deduplicate, sort, and validate a layer-indices field.

    Returns a tuple of unique, non-negative integers in ascending order.
    Non-numeric or negative values are silently dropped.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        try:
            tokens = [str(item).strip() for item in list(raw)]
        except TypeError:
            tokens = []
    out: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value < 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(sorted(out))
