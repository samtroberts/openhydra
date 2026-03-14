from __future__ import annotations

import zlib


class Lz4Codec:
    """Tier 2 placeholder using zlib fallback where lz4 isn't guaranteed."""

    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=6)

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)
