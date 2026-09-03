#!/usr/bin/env python3
"""Continuously record GPU compute-process PIDs for a functional safety cell."""
from __future__ import annotations

import json
import signal
import subprocess
import time

running = True


def stop(_signum, _frame):
    global running
    running = False


def emit(row):
    print(json.dumps(row, sort_keys=True), flush=True)


def main():
    errors = 0
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while running:
        query_started_mono_ns = time.monotonic_ns()
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                text=True, capture_output=True, timeout=2, check=False)
            query_finished_mono_ns = time.monotonic_ns()
            if result.returncode:
                if not running:
                    break
                errors += 1
                emit({"event": "sample", "wall_time_ns": time.time_ns(),
                      "query_started_mono_ns": query_started_mono_ns,
                      "query_finished_mono_ns": query_finished_mono_ns, "pids": [],
                      "error": f"nvidia-smi exit {result.returncode}"})
            else:
                pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
                emit({"event": "sample", "wall_time_ns": time.time_ns(),
                      "query_started_mono_ns": query_started_mono_ns,
                      "query_finished_mono_ns": query_finished_mono_ns,
                      "pids": sorted(pids)})
        except BaseException as error:
            if not running:
                break
            errors += 1
            emit({"event": "sample", "wall_time_ns": time.time_ns(),
                  "query_started_mono_ns": query_started_mono_ns,
                  "query_finished_mono_ns": time.monotonic_ns(), "pids": [],
                  "error": f"{type(error).__name__}: {error}"})
        time.sleep(0.1)
    emit({"event": "final", "wall_time_ns": time.time_ns(),
          "monotonic_ns": time.monotonic_ns(), "errors": errors})
    return int(errors != 0)


if __name__ == "__main__":
    raise SystemExit(main())
