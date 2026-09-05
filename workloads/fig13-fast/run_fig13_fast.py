#!/usr/bin/env python3
"""Fig.13 fast four-arm memory/scheduling matrix (minimal runnable).

Arms: baseline, memory_only, sched_only, combined.
Workload: two concurrent uvmbench HotSpot tenants, size_factor 0.6,
mode uvm, iterations 1.
Memory policy: extension/prefetch_eviction_pid (high param 20, low param 80,
keyed by tenant PID).
Scheduling policy: extension/gpu_sched_set_timeslices
(uvmbench_high 1000000 us, uvmbench_low 200 us, keyed by comm via /tmp
symlinks).

Per-arm ordering:
  Each tenant is spawned through tenant_launcher.py, which execs its
  interpreter normally so Popen returns, then SIGSTOPs itself before
  exec'ing the real tenant image; the PID is preserved by that exec and
  the harness confirms the stopped state.
  1. Both tenants are spawned SIGSTOPped before exec, so both PIDs exist
     before any policy starts and before any tenant CUDA initialization.
  2. The sched policy (sched_only, combined) starts while tenants are
     stopped, i.e. before tenant CUDA initialization.
  3. The memory policy (memory_only, combined) starts after tenant PIDs
     exist and before SIGCONT.
  4. Both tenants are resumed; each tenant's completion latency is timed
     independently from its own SIGCONT to its own exit.

Five interleaved blocks: each block runs all four arms back to back with
the arm order rotated, so every arm occupies every position across blocks.

Engagement counters are recorded as metadata only and are never a gate.
No correctness/review/verifier/hash/checksum/digest gate, no retry, no
filtering; failures and raw numbers are preserved in the CSV, per-arm
meta.json, events.log, and tenant/tool logs.

CPU dry-run: --dry-run replaces tenants with stubs/stub_tenant.py and
policy tools with stubs/stub_policy.py so the orchestration (stop-before-
exec, PID capture, policy ordering, independent timing, block/CSV bookkeeping)
can be validated without a GPU or kernel BPF attachment.
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

ARMS = ("baseline", "memory_only", "sched_only", "combined")
ARM_TOOLS = {
    "baseline": (),
    "memory_only": ("mem",),
    "sched_only": ("sched",),
    "combined": ("sched", "mem"),
}

KERNEL = "hotspot"
SIZE_FACTOR = "0.6"
ITERATIONS = 1
MEM_HIGH_PARAM = "20"
MEM_LOW_PARAM = "80"
SCHED_HIGH_TS = "1000000"
SCHED_LOW_TS = "200"
HIGH_NAME = "uvmbench_high"
LOW_NAME = "uvmbench_low"
TOOL_SETTLE_S = 1.0
DRY_RUN_SLEEP = {"high": 1.2, "low": 1.8}

CSV_COLUMNS = [
    "block", "arm", "high_pid", "low_pid",
    "high_latency_s", "low_latency_s",
    "high_median_ms", "low_median_ms",
    "high_bw_gbps", "low_bw_gbps",
    "high_rc", "low_rc",
    "high_timeout", "low_timeout",
    "sched_meta", "mem_meta", "notes",
]


class RunLog:
    def __init__(self, fh):
        self.fh = fh

    def info(self, msg):
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def proc_state(pid):
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read().decode(errors="replace")
        return data.rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError, UnicodeDecodeError):
        return "?"


def wait_stopped(pid, log, label, timeout_s=3.0):
    deadline = time.time() + timeout_s
    state = proc_state(pid)
    while state not in ("T", "Z", "X") and time.time() < deadline:
        time.sleep(0.05)
        state = proc_state(pid)
    log.info(f"tenant {label} state after spawn: {state}")
    return state


def ensure_symlinks(args, log):
    if args.dry_run:
        target = HERE / "stubs" / "stub_tenant.py"
    else:
        target = REPO_ROOT / "microbench" / "memory" / "uvmbench"
    for name in (HIGH_NAME, LOW_NAME):
        link = Path("/tmp") / name
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
            log.info(f"symlink /tmp/{name} -> {target}")
        except OSError as exc:
            log.info(f"symlink /tmp/{name} failed: {exc} (continuing)")


def spawn_tenant(name, role, arm_dir, args, log):
    link = Path("/tmp") / name
    env = os.environ.copy()
    if args.dry_run:
        env["FIG13_FAST_STUB_SLEEP"] = f"{DRY_RUN_SLEEP[role]:.1f}"
        target_argv = [str(link)]
    else:
        target_argv = [
            str(link),
            f"--kernel={KERNEL}",
            f"--size_factor={SIZE_FACTOR}",
            "--mode=uvm",
            f"--iterations={ITERATIONS}",
            str(arm_dir / f"uvmbench_{name}_results.csv"),
        ]
    cmd = [sys.executable, str(HERE / "tenant_launcher.py"), *target_argv]
    log_path = arm_dir / f"tenant_{name}.log"
    logf = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        cwd=str(arm_dir),
        env=env,
    )
    log.info(f"tenant {name} spawned pid={proc.pid} via tenant_launcher (target={link}) log={log_path}")
    return proc, logf


def tool_command(label, args, high_pid, low_pid):
    ext = REPO_ROOT / "extension"
    if label == "sched":
        if args.dry_run:
            return [sys.executable, str(HERE / "stubs" / "stub_policy.py"), "sched"]
        return [
            "sudo", str(ext / "gpu_sched_set_timeslices"),
            "-p", f"{HIGH_NAME}:{SCHED_HIGH_TS}",
            "-p", f"{LOW_NAME}:{SCHED_LOW_TS}",
        ]
    if args.dry_run:
        return [sys.executable, str(HERE / "stubs" / "stub_policy.py"), "mem"]
    return [
        "sudo", str(ext / "prefetch_eviction_pid"),
        "-p", str(high_pid), "-P", MEM_HIGH_PARAM,
        "-l", str(low_pid), "-L", MEM_LOW_PARAM,
    ]


def spawn_tool(cmd, log_path, log, label):
    logf = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(HERE))
    log.info(f"{label} tool started pid={proc.pid} cmd={' '.join(cmd)} log={log_path}")
    time.sleep(TOOL_SETTLE_S)
    if proc.poll() is not None:
        log.info(f"{label} tool exited early rc={proc.returncode} (continuing; no gate)")
    return proc, logf


def stop_tool(proc, log, label):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    log.info(f"{label} tool stopped rc={proc.returncode}")


def watch_tenant(proc, role, timeout_s, results, log):
    try:
        rc = proc.wait(timeout=timeout_s)
        results[role] = {"rc": rc, "t_done": time.time(), "timeout": False}
        log.info(f"tenant {role} finished rc={rc}")
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            rc = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rc = None
        results[role] = {"rc": rc, "t_done": time.time(), "timeout": True}
        log.info(f"tenant {role} killed after timeout rc={rc}")


def parse_tenant_metrics(path):
    median = None
    bw = None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return median, bw
    m = re.search(r"Median time:\s*([\d.]+)", text)
    if m:
        median = m.group(1)
    m = re.search(r"Bandwidth:\s*([\d.]+)\s*GB/s", text)
    if m:
        bw = m.group(1)
    return median, bw


def parse_sched_meta(path):
    meta = {"hit": None, "miss": None, "mod": None}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return meta
    for key, pat in (
        ("hit", r"policy_hit:\s*(\d+)"),
        ("miss", r"policy_miss:\s*(\d+)"),
        ("mod", r"timeslice_mod:\s*(\d+)"),
    ):
        m = re.search(pat, text)
        if m:
            meta[key] = m.group(1)
    return meta


def parse_mem_meta(path):
    meta = {"act": None, "allow": None, "deny": None}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return meta
    for key, pat in (
        ("act", r"Total activated:\s*(\d+)"),
        ("allow", r"Policy allow \(moved\):\s*(\d+)"),
        ("deny", r"Policy deny \(not moved\):\s*(\d+)"),
    ):
        m = re.search(pat, text)
        if m:
            meta[key] = m.group(1)
    return meta


def format_meta(label, arm, meta, args):
    if label not in ARM_TOOLS[arm]:
        return "n/a"
    if not any(v is not None for v in meta.values()):
        return "missing"
    prefix = "stub:" if args.dry_run else ""
    body = ";".join(f"{k}={v if v is not None else 'na'}" for k, v in meta.items())
    return prefix + body


def run_cleanup(args, log, phase):
    tool = REPO_ROOT / "extension" / "cleanup_struct_ops_tool"
    if not tool.exists():
        log.info(f"{phase} cleanup_struct_ops_tool missing (skipped)")
        return
    try:
        r = subprocess.run(["sudo", str(tool)], capture_output=True, text=True, timeout=60)
        log.info(f"{phase} cleanup_struct_ops_tool rc={r.returncode} (advisory)")
    except subprocess.SubprocessError as exc:
        log.info(f"{phase} cleanup_struct_ops_tool failed: {exc} (advisory)")


def pre_clean(args, log):
    for name in (HIGH_NAME, LOW_NAME):
        subprocess.run(["pkill", "-9", "-f", f"/tmp/{name}"], capture_output=True)
    if not args.dry_run:
        run_cleanup(args, log, "pre")
    time.sleep(0.5)


def run_arm(block, arm, args, out_root, log):
    arm_dir = out_root / f"block{block:02d}_{arm}"
    arm_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "block": block,
        "arm": arm,
        "mode": "dry_run" if args.dry_run else "gpu",
        "kernel": KERNEL,
        "size_factor": SIZE_FACTOR,
        "iterations": ITERATIONS,
        "high_name": HIGH_NAME,
        "low_name": LOW_NAME,
        "mem_params": f"high={MEM_HIGH_PARAM}/low={MEM_LOW_PARAM}",
        "sched_params": f"high={SCHED_HIGH_TS}us/low={SCHED_LOW_TS}us",
    }
    row = {c: "" for c in CSV_COLUMNS}
    row["block"] = block
    row["arm"] = arm
    notes = []

    high = low = None
    tools = {}
    files = []
    t_cont = {"high": None, "low": None}
    results = {}

    def kill_quiet(proc, label):
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
                notes.append(f"{label}_killed_by_harness")
        except Exception:
            pass

    def close_files():
        for f in files:
            try:
                f.close()
            except OSError:
                pass

    try:
        pre_clean(args, log)
        ensure_symlinks(args, log)

        high, high_logf = spawn_tenant(HIGH_NAME, "high", arm_dir, args, log)
        t_spawn_high = time.time()
        low, low_logf = spawn_tenant(LOW_NAME, "low", arm_dir, args, log)
        t_spawn_low = time.time()
        files.extend([high_logf, low_logf])

        row["high_pid"] = high.pid
        row["low_pid"] = low.pid
        meta["high_pid"] = high.pid
        meta["low_pid"] = low.pid
        meta["t_spawn_high"] = t_spawn_high
        meta["t_spawn_low"] = t_spawn_low

        state_high = wait_stopped(high.pid, log, HIGH_NAME)
        state_low = wait_stopped(low.pid, log, LOW_NAME)
        meta["states_after_spawn"] = {"high": state_high, "low": state_low}
        if state_high != "T":
            notes.append(f"{HIGH_NAME}_not_stopped_pre_policy:{state_high}")
        if state_low != "T":
            notes.append(f"{LOW_NAME}_not_stopped_pre_policy:{state_low}")

        for label in ARM_TOOLS[arm]:
            cmd = tool_command(label, args, high.pid, low.pid)
            proc, logf = spawn_tool(cmd, arm_dir / f"{label}_tool.log", log, label)
            files.append(logf)
            tools[label] = proc
            meta[f"t_{label}_start"] = time.time()
            meta[f"command_{label}"] = cmd

        t_cont["high"] = time.time()
        os.kill(high.pid, signal.SIGCONT)
        time.sleep(0.02)
        t_cont["low"] = time.time()
        os.kill(low.pid, signal.SIGCONT)
        meta["t_cont_high"] = t_cont["high"]
        meta["t_cont_low"] = t_cont["low"]
        log.info(f"SIGCONT sent: high={t_cont['high']} low={t_cont['low']}")

        threads = []
        for proc, role in ((high, "high"), (low, "low")):
            th = threading.Thread(
                target=watch_tenant,
                args=(proc, role, args.timeout, results, log),
            )
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        for label in ("mem", "sched"):
            if label in tools:
                stop_tool(tools[label], log, label)
                meta[f"rc_{label}_tool"] = tools[label].returncode

        if not args.dry_run:
            run_cleanup(args, log, "post")

        for role, name in (("high", HIGH_NAME), ("low", LOW_NAME)):
            r = results.get(role)
            if r is None:
                notes.append(f"{role}_no_result")
                continue
            latency = r["t_done"] - t_cont[role]
            row[f"{role}_latency_s"] = f"{latency:.6f}"
            row[f"{role}_rc"] = r["rc"] if r["rc"] is not None else ""
            row[f"{role}_timeout"] = int(r["timeout"])
            meta[f"t_done_{role}"] = r["t_done"]
            meta[f"rc_{role}"] = r["rc"]
            median, bw = parse_tenant_metrics(arm_dir / f"tenant_{name}.log")
            row[f"{role}_median_ms"] = median if median is not None else ""
            row[f"{role}_bw_gbps"] = bw if bw is not None else ""

        row["sched_meta"] = format_meta("sched", arm, parse_sched_meta(arm_dir / "sched_tool.log"), args)
        row["mem_meta"] = format_meta("mem", arm, parse_mem_meta(arm_dir / "mem_tool.log"), args)
        meta["engagement_metadata"] = {
            "sched": parse_sched_meta(arm_dir / "sched_tool.log") if "sched" in ARM_TOOLS[arm] else None,
            "mem": parse_mem_meta(arm_dir / "mem_tool.log") if "mem" in ARM_TOOLS[arm] else None,
            "note": "metadata only; never a gate",
        }
    except Exception as exc:
        notes.append(f"harness_error:{type(exc).__name__}:{exc}")
        log.info(f"block{block:02d}_{arm} harness error: {exc!r}")
    finally:
        kill_quiet(high, HIGH_NAME)
        kill_quiet(low, LOW_NAME)
        for label, p in tools.items():
            kill_quiet(p, f"{label}_tool")
        close_files()
        meta["timeouts"] = {
            role: bool(results.get(role, {}).get("timeout")) for role in ("high", "low")
        }
        meta["notes"] = notes
        row["notes"] = ";".join(notes)
        try:
            (arm_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        except OSError as exc:
            log.info(f"meta.json write failed: {exc}")
    return row


def print_summary(rows, log):
    log.info("SUMMARY (descriptive only; no gates)")
    header = f"{'arm':<12} {'n':<3} {'high_lat(s)':>12} {'low_lat(s)':>12} {'high_med(ms)':>13} {'low_med(ms)':>12}"
    print(header, flush=True)
    log.info(header)
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        n = len(arm_rows)
        def mean_of(key):
            vals = []
            for r in arm_rows:
                v = r.get(key, "")
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            return sum(vals) / len(vals) if vals else None
        h_lat = mean_of("high_latency_s")
        l_lat = mean_of("low_latency_s")
        h_med = mean_of("high_median_ms")
        l_med = mean_of("low_median_ms")
        line = (
            f"{arm:<12} {n:<3} "
            f"{h_lat:>12.2f} {l_lat:>12.2f} "
            f"{h_med:>13.1f} {l_med:>12.1f}"
            if all(v is not None for v in (h_lat, l_lat, h_med, l_med))
            else f"{arm:<12} {n:<3} (incomplete rows; see CSV)"
        )
        print(line, flush=True)
        log.info(line)


def main():
    ap = argparse.ArgumentParser(description="Fig.13 fast four-arm matrix")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU-only orchestration test with stub tenants/policies")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-tenant wall-clock seconds from SIGCONT")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    if args.dry_run:
        for stub in ("stub_tenant.py", "stub_policy.py"):
            if not (HERE / "stubs" / stub).exists():
                print(f"missing stub: {stub}")
                sys.exit(1)
    else:
        required = [
            REPO_ROOT / "microbench" / "memory" / "uvmbench",
            REPO_ROOT / "extension" / "gpu_sched_set_timeslices",
            REPO_ROOT / "extension" / "prefetch_eviction_pid",
            REPO_ROOT / "extension" / "cleanup_struct_ops_tool",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            print("missing required tools:\n  " + "\n  ".join(missing))
            sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if args.dry_run else ""
    out_root = Path(args.results_dir) / f"fig13_fast_{ts}{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)

    arm_orders = {}
    for b in range(args.blocks):
        arm_orders[b] = list(ARMS[b:] + ARMS[:b])

    run_cfg = {
        "mode": "dry_run" if args.dry_run else "gpu",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "kernel": KERNEL,
        "size_factor": SIZE_FACTOR,
        "iterations": ITERATIONS,
        "tenants": 2,
        "arms": list(ARMS),
        "arm_orders": arm_orders,
        "timeout_s": args.timeout,
        "mem_policy": "extension/prefetch_eviction_pid -p HIGH_PID -P 20 -l LOW_PID -L 80",
        "sched_policy": "extension/gpu_sched_set_timeslices -p uvmbench_high:1000000 -p uvmbench_low:200",
        "uvmbench": str(REPO_ROOT / "microbench" / "memory" / "uvmbench"),
        "engagement": "counters recorded as metadata only; never a gate",
    }
    (out_root / "run.json").write_text(json.dumps(run_cfg, indent=2))

    events_fh = open(out_root / "events.log", "w", buffering=1)
    log = RunLog(events_fh)
    log.info(f"fig13-fast start mode={'dry_run' if args.dry_run else 'gpu'} blocks={args.blocks} out={out_root}")

    csv_path = out_root / "fig13_fast.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    rows = []
    for block, order in arm_orders.items():
        log.info(f"block {block} arm order: {','.join(order)}")
        for arm in order:
            log.info(f"--- block {block} arm {arm} ---")
            row = run_arm(block, arm, args, out_root, log)
            rows.append(row)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

    print_summary(rows, log)
    log.info(f"fig13-fast done csv={csv_path}")
    events_fh.close()
    print(f"\nResults: {out_root}")
    print(f"CSV: {csv_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
