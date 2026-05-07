# Copyright 2026 OpenHydra contributors — Apache 2.0

from __future__ import annotations


def format_address(host: str, port: int | str) -> str:
    """Format host:port for network addresses, wrapping IPv6 in brackets."""
    h = str(host).strip()
    if ":" in h and not h.startswith("["):
        return f"[{h}]:{port}"
    return f"{h}:{port}"
