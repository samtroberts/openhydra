#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import statistics
import subprocess
import threading
import time
from typing import Any
from urllib import error as urlerror
from urllib import request


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if q <= 0.0:
        return ordered[0]
    if q >= 1.0:
        return ordered[-1]
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def _post_completion(
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    pipeline_width: int,
    grounding: bool,
    timeout_s: float,
    request_id: int,
) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": f"SLO chaos probe request #{request_id}",
        "max_tokens": int(max_tokens),
        "pipeline_width": int(pipeline_width),
        "grounding": bool(grounding),
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        method="POST",
        data=body,
        headers={"content-type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with request.urlopen(req, timeout=max(0.1, float(timeout_s))) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            choices = list(data.get("choices", []))
            completion = str((choices[0] if choices else {}).get("text", "")).strip()
            return {
                "ok": bool(completion),
                "status": int(getattr(resp, "status", 200)),
                "latency_ms": elapsed_ms,
                "error": "",
            }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": False,
            "status": int(getattr(exc, "code", 0) or 0),
            "latency_ms": elapsed_ms,
            "error": str(exc),
        }


def _fetch_metrics(base_url: str, timeout_s: float) -> dict[str, float]:
    endpoint = f"{base_url.rstrip('/')}/metrics"
    try:
        with request.urlopen(endpoint, timeout=max(0.1, float(timeout_s))) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return {}
    metrics: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _run_chaos_restart(
    *,
    compose_project: str | None,
    service: str,
) -> dict[str, Any]:
    cmd = ["docker", "compose"]
    if compose_project:
        cmd.extend(["-p", str(compose_project)])
    cmd.extend(["restart", str(service)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "command": cmd,
        "exit_code": int(proc.returncode),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": (proc.returncode == 0),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    duration_s = max(5.0, float(args.duration_s))
    workers = max(1, int(args.workers))
    chaos_at_s = max(0.0, float(args.chaos_at_s))
    timeout_s = max(0.1, float(args.request_timeout_s))
    base_url = str(args.base_url).strip().rstrip("/")

    results: list[dict[str, Any]] = []
    request_counter = 0
    t0 = time.perf_counter()
    end_at = t0 + duration_s

    metrics_before = _fetch_metrics(base_url, timeout_s=timeout_s)
    chaos_result: dict[str, Any] = {
        "ok": True,
        "command": [],
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "triggered": False,
    }

    chaos_lock = threading.Lock()

    def _chaos_thread() -> None:
        nonlocal chaos_result
        if args.skip_chaos or not str(args.chaos_service).strip():
            return
        delay = max(0.0, chaos_at_s)
        time.sleep(delay)
        result = _run_chaos_restart(
            compose_project=(str(args.compose_project).strip() or None),
            service=str(args.chaos_service).strip(),
        )
        result["triggered"] = True
        with chaos_lock:
            chaos_result = result

    thread = threading.Thread(target=_chaos_thread, daemon=True)
    thread.start()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        inflight = {}
        while (time.perf_counter() < end_at) or inflight:
            while time.perf_counter() < end_at and len(inflight) < workers:
                req_id = request_counter
                request_counter += 1
                fut = pool.submit(
                    _post_completion,
                    base_url=base_url,
                    model=str(args.model),
                    max_tokens=int(args.max_tokens),
                    pipeline_width=max(1, int(args.pipeline_width)),
                    grounding=bool(args.grounding),
                    timeout_s=timeout_s,
                    request_id=req_id,
                )
                inflight[fut] = req_id

            if not inflight:
                continue

            done, _ = wait(list(inflight.keys()), timeout=0.5, return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.pop(fut, None)
                try:
                    results.append(dict(fut.result()))
                except Exception as exc:  # pragma: no cover
                    results.append({"ok": False, "status": 0, "latency_ms": 0.0, "error": str(exc)})

    thread.join(timeout=2.0)
    elapsed_s = max(0.0, time.perf_counter() - t0)
    metrics_after = _fetch_metrics(base_url, timeout_s=timeout_s)

    successes = [item for item in results if bool(item.get("ok"))]
    failures = [item for item in results if not bool(item.get("ok"))]
    latencies = [float(item.get("latency_ms", 0.0)) for item in successes]
    success_rate = (float(len(successes)) / float(len(results))) if results else 0.0
    p95_ms = _percentile(latencies, 0.95) if latencies else 0.0

    summary = {
        "duration_s": round(elapsed_s, 3),
        "workers": workers,
        "total_requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "success_rate": round(success_rate, 6),
        "throughput_rps": round((len(results) / elapsed_s), 6) if elapsed_s > 0.0 else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 3) if latencies else 0.0,
            "p95": round(p95_ms, 3),
            "p99": round(_percentile(latencies, 0.99), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "chaos": chaos_result,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "slo_thresholds": {
            "min_success_rate": float(args.min_success_rate),
            "max_p95_ms": float(args.max_p95_ms),
        },
        "pass": (
            bool(results)
            and success_rate >= float(args.min_success_rate)
            and p95_ms <= float(args.max_p95_ms)
            and bool(chaos_result.get("ok", True))
        ),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenHydra sustained load + chaos SLO validation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--request-timeout-s", type=float, default=8.0)
    parser.add_argument("--model", default="openhydra-toy-345m")
    parser.add_argument("--pipeline-width", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--grounding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-success-rate", type=float, default=0.97)
    parser.add_argument("--max-p95-ms", type=float, default=1500.0)
    parser.add_argument("--chaos-at-s", type=float, default=30.0)
    parser.add_argument("--chaos-service", default="peer")
    parser.add_argument("--compose-project", default=None)
    parser.add_argument("--skip-chaos", action="store_true")
    parser.add_argument("--report-json", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_probe(args)
    out = json.dumps(summary, indent=2, sort_keys=True)
    print(out)
    if args.report_json:
        with open(str(args.report_json), "w", encoding="utf-8") as handle:
            handle.write(out + "\n")
    raise SystemExit(0 if bool(summary.get("pass")) else 1)


if __name__ == "__main__":
    main()
