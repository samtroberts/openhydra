from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any
from urllib import request


DEFAULT_PIECE_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True)
class GenesisResult:
    model_id: str
    source: str
    artifact_path: str
    artifact_sha256: str
    artifact_bytes: int
    manifest_path: str
    torrent_meta_path: str


@dataclass(frozen=True)
class PieceDigest:
    index: int
    offset: int
    size: int
    sha256: str


class IntegrityError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _piece_digests(path: Path, piece_bytes: int = DEFAULT_PIECE_BYTES) -> list[PieceDigest]:
    if piece_bytes <= 0:
        raise ValueError("piece_bytes must be positive")

    out: list[PieceDigest] = []
    with path.open("rb") as handle:
        offset = 0
        index = 0
        while True:
            block = handle.read(piece_bytes)
            if not block:
                break
            out.append(
                PieceDigest(
                    index=index,
                    offset=offset,
                    size=len(block),
                    sha256=hashlib.sha256(block).hexdigest(),
                )
            )
            offset += len(block)
            index += 1

    return out


def _download_file(url: str, destination: Path, timeout_s: float = 30.0) -> None:
    req = request.Request(url=url, method="GET", headers={"User-Agent": "OpenHydra/0.1 genesis"})
    with request.urlopen(req, timeout=timeout_s) as response:
        with destination.open("wb") as out:
            shutil.copyfileobj(response, out)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def bootstrap_weights(
    model_id: str,
    *,
    cache_dir: str = ".cache/openhydra",
    local_path: str | None = None,
    source_url: str | None = None,
    expected_sha256: str | None = None,
    force_refresh: bool = False,
    piece_bytes: int = DEFAULT_PIECE_BYTES,
) -> GenesisResult:
    """Bootstrap model weights and produce manifest + pseudo torrent metadata.

    Acquisition order:
    1) local_path if provided
    2) source_url if provided
    3) deterministic placeholder
    """
    root = Path(cache_dir) / model_id
    root.mkdir(parents=True, exist_ok=True)

    artifact = root / f"{model_id}.safetensors"
    source = "cache"

    if force_refresh and artifact.exists():
        artifact.unlink()

    if not artifact.exists():
        if local_path:
            src = Path(local_path)
            if not src.exists():
                raise FileNotFoundError(f"local_path not found: {src}")
            shutil.copyfile(src, artifact)
            source = "local-copy"
        elif source_url:
            _download_file(source_url, artifact)
            source = "http-download"
        else:
            artifact.write_bytes(b"openhydra-placeholder-weights")
            source = "local-placeholder"

    artifact_sha256 = sha256_file(artifact)
    if expected_sha256 and artifact_sha256.lower() != expected_sha256.lower():
        raise IntegrityError(
            f"checksum mismatch for {model_id}: expected {expected_sha256}, got {artifact_sha256}"
        )

    pieces = _piece_digests(artifact, piece_bytes=piece_bytes)
    now_unix_ms = int(time.time() * 1000)

    manifest_payload = {
        "schema": "openhydra-model-manifest/v1",
        "model_id": model_id,
        "artifact": {
            "path": str(artifact),
            "bytes": artifact.stat().st_size,
            "sha256": artifact_sha256,
            "piece_bytes": piece_bytes,
            "piece_count": len(pieces),
        },
        "created_unix_ms": now_unix_ms,
        "source": source,
    }

    torrent_payload = {
        "schema": "openhydra-genesis-torrent/v1",
        "model_id": model_id,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_sha256,
        "piece_bytes": piece_bytes,
        "piece_count": len(pieces),
        "pieces": [asdict(piece) for piece in pieces],
        "created_unix_ms": now_unix_ms,
    }

    manifest_path = root / "manifest.json"
    torrent_meta_path = root / "genesis.torrent.json"
    _write_manifest(manifest_path, manifest_payload)
    _write_manifest(torrent_meta_path, torrent_payload)

    return GenesisResult(
        model_id=model_id,
        source=source,
        artifact_path=str(artifact),
        artifact_sha256=artifact_sha256,
        artifact_bytes=artifact.stat().st_size,
        manifest_path=str(manifest_path),
        torrent_meta_path=str(torrent_meta_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenHydra Genesis weight bootstrap")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", default=".cache/openhydra")
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--force-refresh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--piece-bytes", type=int, default=DEFAULT_PIECE_BYTES)
    args = parser.parse_args()

    result = bootstrap_weights(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        local_path=args.local_path,
        source_url=args.source_url,
        expected_sha256=args.expected_sha256,
        force_refresh=args.force_refresh,
        piece_bytes=max(1, args.piece_bytes),
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
