from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import threading
import time

from verification.reputation import Reputation


@dataclass
class PeerHealthStats:
    peer_id: str
    pings_ok: int = 0
    pings_failed: int = 0
    inferences_ok: int = 0
    inferences_failed: int = 0
    verifications_ok: int = 0
    verifications_failed: int = 0
    latency_ema_ms: float | None = None
    latency_dev_ema: float = 0.0
    consecutive_failures: int = 0
    last_seen_unix_ms: int = 0


@dataclass
class HealthState:
    peers: dict[str, PeerHealthStats] = field(default_factory=dict)


class HealthScorer:
    """Persistent per-peer health tracker feeding routing reputation."""

    def __init__(self, store_path: str, flush_interval_s: float = 5.0):
        self.path = Path(store_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = HealthState()
        self._lock = threading.Lock()
        self._dirty = False
        self._flush_interval_s = max(0.05, float(flush_interval_s))
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="openhydra-health-flush",
            daemon=True,
        )
        self._load()
        self._flush_thread.start()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text())
        peers = {}
        for peer_id, item in dict(payload.get("peers", {})).items():
            peers[peer_id] = PeerHealthStats(
                peer_id=peer_id,
                pings_ok=int(item.get("pings_ok", 0)),
                pings_failed=int(item.get("pings_failed", 0)),
                inferences_ok=int(item.get("inferences_ok", 0)),
                inferences_failed=int(item.get("inferences_failed", 0)),
                verifications_ok=int(item.get("verifications_ok", 0)),
                verifications_failed=int(item.get("verifications_failed", 0)),
                latency_ema_ms=(float(item["latency_ema_ms"]) if item.get("latency_ema_ms") is not None else None),
                latency_dev_ema=float(item.get("latency_dev_ema", 0.0)),
                consecutive_failures=int(item.get("consecutive_failures", 0)),
                last_seen_unix_ms=int(item.get("last_seen_unix_ms", 0)),
            )
        self.state = HealthState(peers=peers)

    def _save(self) -> None:
        serializable = {
            "peers": {
                peer_id: asdict(stats)
                for peer_id, stats in self.state.peers.items()
            }
        }
        self.path.write_text(json.dumps(serializable, indent=2))

    def _peer(self, peer_id: str) -> PeerHealthStats:
        if peer_id not in self.state.peers:
            self.state.peers[peer_id] = PeerHealthStats(peer_id=peer_id)
        return self.state.peers[peer_id]

    def _peer_or_none(self, peer_id: str) -> PeerHealthStats | None:
        return self.state.peers.get(str(peer_id))

    @staticmethod
    def _default_stats(peer_id: str) -> PeerHealthStats:
        return PeerHealthStats(peer_id=str(peer_id))

    @staticmethod
    def _update_latency(stats: PeerHealthStats, latency_ms: float, alpha: float = 0.25) -> None:
        if stats.latency_ema_ms is None:
            stats.latency_ema_ms = latency_ms
            stats.latency_dev_ema = 0.0
            return

        prev = stats.latency_ema_ms
        stats.latency_ema_ms = (1 - alpha) * prev + alpha * latency_ms
        deviation = abs(latency_ms - stats.latency_ema_ms)
        stats.latency_dev_ema = (1 - alpha) * stats.latency_dev_ema + alpha * deviation

    def record_ping(self, peer_id: str, healthy: bool, latency_ms: float) -> None:
        with self._lock:
            stats = self._peer(peer_id)
            stats.last_seen_unix_ms = int(time.time() * 1000)

            if healthy:
                stats.pings_ok += 1
                stats.consecutive_failures = 0
                self._update_latency(stats, latency_ms)
            else:
                stats.pings_failed += 1
                stats.consecutive_failures += 1
            self._dirty = True

    def record_inference(self, peer_id: str, success: bool, latency_ms: float | None = None) -> None:
        with self._lock:
            stats = self._peer(peer_id)
            stats.last_seen_unix_ms = int(time.time() * 1000)

            if success:
                stats.inferences_ok += 1
                stats.consecutive_failures = 0
                if latency_ms is not None:
                    self._update_latency(stats, latency_ms)
            else:
                stats.inferences_failed += 1
                stats.consecutive_failures += 1
            self._dirty = True

    def record_verification(self, peer_id: str, success: bool) -> None:
        with self._lock:
            stats = self._peer(peer_id)
            stats.last_seen_unix_ms = int(time.time() * 1000)

            if success:
                stats.verifications_ok += 1
                stats.consecutive_failures = 0
            else:
                stats.verifications_failed += 1
                stats.consecutive_failures += 1
            self._dirty = True

    def score(self, peer_id: str) -> float:
        with self._lock:
            stats = self._peer_or_none(peer_id)
            if stats is None:
                stats = self._default_stats(peer_id)
        return self._score_from_stats(stats)

    @staticmethod
    def _score_from_stats(stats: PeerHealthStats) -> float:
        inference_total = stats.inferences_ok + stats.inferences_failed
        ping_total = stats.pings_ok + stats.pings_failed
        verification_total = stats.verifications_ok + stats.verifications_failed

        inference_success = stats.inferences_ok / inference_total if inference_total else 0.7
        verification_accuracy = (
            stats.verifications_ok / verification_total
            if verification_total
            else inference_success
        )
        verification_success = 0.65 * inference_success + 0.35 * verification_accuracy
        uptime = stats.pings_ok / ping_total if ping_total else 0.5

        if stats.latency_ema_ms is None or stats.latency_ema_ms <= 0:
            latency_consistency = 0.5
        else:
            cv = stats.latency_dev_ema / stats.latency_ema_ms
            latency_consistency = max(0.0, min(1.0, 1.0 - min(cv, 1.0)))

        base = Reputation(
            verification_success=verification_success,
            uptime=uptime,
            latency_consistency=latency_consistency,
            stake_factor=0.0,
        ).score()

        # Penalize recent instability but do not fully zero healthy peers.
        penalty = min(30.0, float(stats.consecutive_failures * 6))
        return max(0.0, min(100.0, base - penalty))

    def scores(self, peer_ids: list[str]) -> dict[str, float]:
        scores_by_peer: dict[str, float] = {}
        with self._lock:
            for peer_id in peer_ids:
                stats = self._peer_or_none(peer_id)
                if stats is None:
                    stats = self._default_stats(peer_id)
                scores_by_peer[peer_id] = self._score_from_stats(stats)
        return scores_by_peer

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {peer_id: asdict(stats) for peer_id, stats in self.state.peers.items()}

    def _wait_for_flush_tick(self) -> bool:
        return self._stop_event.wait(self._flush_interval_s)

    def _flush_loop(self) -> None:
        while True:
            if self._wait_for_flush_tick():
                return
            self.flush()

    def flush(self) -> bool:
        with self._lock:
            if not self._dirty:
                return False
            self._save()
            self._dirty = False
            return True

    def close(self) -> None:
        self._stop_event.set()
        if self._flush_thread.is_alive() and threading.current_thread() is not self._flush_thread:
            self._flush_thread.join(timeout=max(0.1, self._flush_interval_s * 2.0))
        self.flush()
