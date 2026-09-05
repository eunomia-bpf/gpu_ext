#!/usr/bin/env python3
"""Continuously retain real nvidia-smi compute-client PID observations."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time


stopping = False


def stop(signo: int, frame: object) -> None:
    del signo, frame
    global stopping
    stopping = True


def sample() -> dict[str, object]:
    started = time.monotonic_ns()
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    finished = time.monotonic_ns()
    pids: list[int] = []
    error: str | None = None
    if completed.returncode:
        error = f"nvidia-smi exited {completed.returncode}: {completed.stderr.strip()}"
    else:
        for line in completed.stdout.splitlines():
            value = line.strip()
            if not value or value == "No running processes found":
                continue
            if not value.isdigit():
                error = f"unexpected compute PID row: {value}"
                break
            pids.append(int(value))
    return {
        "query_started_mono_ns": started,
        "query_finished_mono_ns": finished,
        "pids": sorted(set(pids)),
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-ms", type=int, default=200)
    args = parser.parse_args()
    if args.interval_ms < 50 or args.interval_ms > 1000:
        parser.error("--interval-ms must be in [50, 1000]")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not stopping:
        print(json.dumps(sample(), separators=(",", ":")), flush=True)
        time.sleep(args.interval_ms / 1000.0)
    print(json.dumps(sample(), separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
