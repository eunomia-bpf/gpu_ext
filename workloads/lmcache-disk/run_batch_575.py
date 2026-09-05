#!/usr/bin/env python3
"""Run the fixed first ten LMCache 575 blocks once, only when invoked by root.

Usage, after traced preflight/correctness and runtime review are complete:
  sudo -n env HF_HOME=/home/yunwei37/.cache/huggingface taskset -c 8-16 \
    ./current-venv/bin/python -B run_batch_575.py

No resume, retries, replacement blocks, runtime changes, or GPU safety logic.
SIGINT/SIGTERM request a deferred stop: finish/reap the current child, including
a successful cell's offline validation, then stop before another GPU cell.
No signal is forwarded and no child is force-killed. A stuck child therefore
requires coordinator inspection; do not wrap this script in a forceful timeout.
SIGKILL, host loss, and external termination of the child are not recoverable
by this coordinator. The existing run-cell remains responsible for GPU cleanup.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

from run_lmcache_disk import CONFIGS, PREFIXES, SCHEDULE, validate_schedule


HERE = Path(__file__).resolve().parent
PYTHON = HERE / "current-venv/bin/python"
RUNNER = HERE / "run_lmcache_disk.py"
OUTPUT = HERE / "raw/storage-575-v3-full-01"
TRACED = HERE / "raw/storage-575-v3-preflight-03/disk"
CORRECTNESS = HERE / "raw/storage-575-v3-correctness-01"
HF_CACHE = "/home/yunwei37/.cache/huggingface"


def record(stream, message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


class DeferredStop:
    def __init__(self):
        self.signum = None
        self.announced = False

    def request(self, signum, _frame):
        # Do not raise or perform I/O inside the signal handler.
        if self.signum is None:
            self.signum = signum

    def announce(self, stream):
        if self.signum is not None and not self.announced:
            self.announced = True
            record(stream, f"PENDING STOP {signal.Signals(self.signum).name}: "
                   "waiting for the owned child and cell validation; no new GPU cell")

    def check(self):
        if self.signum is not None:
            raise InterruptedError(f"deferred {signal.Signals(self.signum).name}; current child reaped")


def run_step(label, arguments, cpus, output, journal, stop, allow_pending=False):
    if not allow_pending:
        stop.check()
    command = ["/usr/bin/taskset", "-c", cpus, str(PYTHON), "-u", "-B",
               str(RUNNER), *map(str, arguments)]
    environment = {**os.environ, "HF_HOME": HF_CACHE}
    record(journal, f"START {label} HF_HOME={HF_CACHE} command={shlex.join(command)}")
    process = None
    returncode = None
    with (output / "logs" / f"{label}.log").open("x") as log:
        try:
            process = subprocess.Popen(command, cwd=HERE, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record(journal, f"CHILD {label} owned_pid_pgid={process.pid}")
            while True:
                stop.announce(journal)
                try:
                    returncode = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process is not None:
                # Also wait on unexpected coordinator-side exceptions; never
                # leave a live child merely because writing/processing failed.
                returncode = process.wait()
            stop.announce(journal)
            record(journal, f"END {label} returncode={returncode}")
    if returncode != 0:
        raise RuntimeError(f"{label} exited {returncode}; see logs/{label}.log")


def check_prerequisite_scope():
    # Detailed raw/output/safety validation is exclusively the existing CLI's
    # job. These checks just exclude smaller smokes or a differently named arm.
    cells = [(TRACED, "lmcache_disk")] + [(CORRECTNESS / name, name) for name in CONFIGS]
    for path, config in cells:
        result = json.loads((path / "result.json").read_text())
        environment = json.loads((path / "environment.json").read_text())
        if (result.get("config") != config or result.get("prefix_count") != PREFIXES
                or environment.get("gpu", {}).get("driver") != "575.57.08"):
            raise ValueError(f"prerequisite must be the full eight-prefix 575 {config}: {path}")
        if path != TRACED and (path / "strace").exists():
            raise ValueError(f"correctness prerequisite must be untraced: {path}")


def run_batch():
    # Atomic refusal of existing output: no overwrite or automatic continuation.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "logs").mkdir()
    stop = DeferredStop()
    previous = {sig: signal.signal(sig, stop.request) for sig in (signal.SIGINT, signal.SIGTERM)}
    active_attempt = None
    try:
        with (OUTPUT / "batch.log").open("x") as journal:
            try:
                run_step("preflight", ["validate-cell", TRACED, "--require-trace"],
                         "17", OUTPUT, journal, stop)
                run_step("correctness", ["compare-outputs", *[CORRECTNESS / name for name in CONFIGS]],
                         "17", OUTPUT, journal, stop)
                stop.check()
                check_prerequisite_scope()
                schedule = json.loads(SCHEDULE.read_text())
                validate_schedule(schedule)
                for row in schedule["attempts"][:10]:
                    stop.check()
                    attempt = row["attempt"]
                    active_attempt = OUTPUT / f"attempt-{attempt:02d}"
                    active_attempt.mkdir(exist_ok=False)
                    for position, config in enumerate(row["order"]):
                        cell = active_attempt / f"position-{position}-{config}"
                        label = f"block-{attempt:02d}-position-{position}-{config}"
                        record(journal, f"PROGRESS cell={attempt * 3 + position + 1}/30 block={attempt + 1}/10")
                        run_step(label + "-run", ["run-cell", "--expected-driver", "575.57.08",
                                 "--prefix-limit", str(PREFIXES), "--config", config, "--output", cell],
                                 "8-16", OUTPUT, journal, stop)
                        run_step(label + "-validate", ["validate-cell", cell],
                                 "17", OUTPUT, journal, stop, allow_pending=True)
                        stop.check()
                    active_attempt = None
                run_step("analysis", ["analyze", OUTPUT], "17", OUTPUT, journal, stop)
                stop.check()
                record(journal, "FINISHED 30 cells and existing analysis completed")
                return 0
            except Exception as error:
                failure = (active_attempt or OUTPUT) / "failure.md"
                with failure.open("x") as stream:
                    stream.write(f"Batch stopped: {type(error).__name__}: {error}\n"
                                 "No automatic retry, skip, or replacement block was launched.\n"
                                 "Review batch.log, child logs, and the existing cell cleanup evidence before continuing.\n")
                record(journal, f"STOPPED {type(error).__name__}: {error}; preserved {failure}")
                return 128 + stop.signum if stop.signum is not None else 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        return run_batch()
    except OSError as error:
        print(f"Batch not started or output unavailable: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
