import argparse
import json
import os
import statistics
import subprocess
import sys

MODES = ("fifo", "native", "bpf")
EXTRA_KEYS = ("mode", "block", "order")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GDS-control executor over a rotated per-block schedule.")
    parser.add_argument("--executor", required=True, help="executor program to run")
    parser.add_argument("--file", default=None, help="input file passed to the executor")
    parser.add_argument("--requests", type=int, default=None, help="request count passed to the executor")
    parser.add_argument("--blocks", type=int, default=5, help="number of rotated blocks (default: 5)")
    parser.add_argument("--raw", default="raw.jsonl", help="raw JSONL file records are appended to")
    parser.add_argument("--summary", default="summary.json", help="compact JSON summary file to write")
    return parser.parse_args()


def executor_command(args, mode):
    command = [args.executor, "--mode", mode]
    if args.file is not None:
        command += ["--file", str(args.file)]
    if args.requests is not None:
        command += ["--requests", str(args.requests)]
    return command


def run_executor(args, mode):
    proc = subprocess.run(executor_command(args, mode), capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise SystemExit(f"executor failed for mode {mode!r} (exit {proc.returncode})")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"executor printed no stdout for mode {mode!r}")
    try:
        record = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"executor last stdout line is not JSON for mode {mode!r}: {exc}")
    if not isinstance(record, dict):
        raise SystemExit(f"executor last stdout line is not a JSON object for mode {mode!r}")
    return record


def build_schedule(blocks):
    schedule = []
    for block in range(blocks):
        offset = block % len(MODES)
        for order in range(len(MODES)):
            schedule.append((block, order, MODES[(offset + order) % len(MODES)]))
    return schedule


def executor_numeric_fields(record):
    return {
        key: value
        for key, value in record.items()
        if key not in EXTRA_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def main():
    args = parse_args()
    records_by_mode = {mode: [] for mode in MODES}
    runs = 0
    with open(args.raw, "a", encoding="utf-8") as raw:
        for block, order, mode in build_schedule(args.blocks):
            record = run_executor(args, mode)
            record["mode"] = mode
            record["block"] = block
            record["order"] = order
            raw.write(json.dumps(record) + "\n")
            raw.flush()
            os.fsync(raw.fileno())
            records_by_mode[mode].append(record)
            runs += 1

    per_mode = {}
    for mode in MODES:
        values = {}
        for record in records_by_mode[mode]:
            for key, value in executor_numeric_fields(record).items():
                values.setdefault(key, []).append(value)
        medians = {key: statistics.median(vals) for key, vals in sorted(values.items())}
        per_mode[mode] = {"runs": len(records_by_mode[mode]), "medians": medians}

    summary = {"runs": runs, "per_mode": per_mode}
    compact = json.dumps(summary, separators=(",", ":"))
    with open(args.summary, "w", encoding="utf-8") as summary_file:
        summary_file.write(compact + "\n")
    print(compact)


if __name__ == "__main__":
    main()
