#!/usr/bin/env python3
"""
Policy Comparison Evaluation Script

Tests different eviction/prefetch policies with uvmbench workloads.

Usage:
  python run_policy_comparison.py                           # Default: hotspot, size_factor=0.6
  python run_policy_comparison.py --resume file.csv         # Resume from existing CSV
  python run_policy_comparison.py --output results_gemm     # Custom output directory

Example configurations:
  sudo python run_policy_comparison.py --kernel hotspot --size-factor 0.6 --output results_hotspot
  sudo python run_policy_comparison.py --kernel gemm --size-factor 0.6 --output results_gemm
  sudo python run_policy_comparison.py --kernel kmeans_sparse --size-factor 0.9 --output results_kmeans
"""

import subprocess
import tempfile
import time
import re
import os
import signal
import sys
import argparse
import csv
import json
import threading
from pathlib import Path
from datetime import datetime

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "extension"
UVM = REPO_ROOT / "microbench" / "memory" / "uvmbench"
TENANT_LAUNCHER = REPO_ROOT / "workloads" / "fig13-fast" / "tenant_launcher.py"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"

# Benchmark parameters (defaults, can be overridden via command line)
DEFAULT_SIZE_FACTOR = 0.6
DEFAULT_KERNEL = "hotspot"
ITERATIONS = 1
NUM_ROUNDS = 1

# Available kernels: rand_stream, seq_stream, hotspot, gemm, kmeans_sparse

# Policy configurations to test
POLICIES = [
    # (policy_name, policy_binary, configs)
    # configs = [(high_param, low_param), ...]
    ("no_policy", None, [(50, 50)]),
    # ("eviction_pid_quota", "eviction_pid_quota", [(50, 50), (80, 20), (90, 10)]),
    # ("eviction_fifo_chance", "eviction_fifo_chance", [(0, 0), (3, 0), (5, 0), (8, 1)]),
    # ("eviction_fifo_chance", "eviction_fifo_chance", [(0, 0), (5, 0)]),
    # eviction_freq_pid_decay: -P = high decay (1=always protected), -L = low decay (larger=less protected)
    ("eviction_freq_pid_decay", "eviction_freq_pid_decay", [(1, 1), (1, 10)]),
    # ("eviction_freq_pid_decay", "eviction_freq_pid_decay", [(1, 1), (1, 10)]),
    # ("prefetch_pid_tree", "prefetch_pid_tree", [(0, 0), (50, 50), (20, 80), (0, 40), (40, 40), (60, 60), (80, 80)]),
    ("prefetch_pid_tree", "prefetch_pid_tree", [(0, 0), (0, 20), (20, 80)]),
    # ("prefetch_pid_tree", "prefetch_pid_tree", [(20, 80)]),
    ("prefetch_eviction_pid", "prefetch_eviction_pid", [(20, 80)]),
]

# Single process configurations (no policy needed)
# (config_name, size_factor_multiplier)
SINGLE_PROCESS_CONFIGS = [
    ("single_1x", 1),      # SIZE_FACTOR * 1
    ("single_2x", 2),      # SIZE_FACTOR * 2
]

# ---------------------------------------------------------------------------
# Opt-in four-arm combined memory/scheduling comparison (ORIGINAL Fig.13
# follow-up). Arms: no policy (baseline), memory-only, scheduling-only, and
# their combination. Each arm runs two concurrent uvmbench tenants; five
# interleaved blocks rotate the arm order so every arm occupies every
# position. All arms share the same settings for a given run.
# ---------------------------------------------------------------------------
COMBINED_ARMS = ("baseline", "memory_only", "sched_only", "combined")
# Tools to start while tenants are stopped before CUDA init, in start order.
# "sched" is comm-keyed (via /tmp symlinks), "mem" is PID-keyed.
COMBINED_ARM_TOOLS = {
    "baseline": (),
    "memory_only": ("mem",),
    "sched_only": ("sched",),
    "combined": ("sched", "mem"),
}
COMBINED_BLOCKS = 5
COMBINED_MEM_HIGH = 20
COMBINED_MEM_LOW = 80
COMBINED_SCHED_HIGH_TS = 1000000
COMBINED_SCHED_LOW_TS = 200
HIGH_NAME = "uvmbench_high"
LOW_NAME = "uvmbench_low"
# Seconds to let a freshly started policy tool settle before resuming tenants.
COMBINED_TOOL_SETTLE_S = 1.0
# uvmbench argument order shared by both tenants.
COMBINED_UVM_ARGS = ["--mode=uvm"]
# Combined-mode CSV schema. Latencies are measured from the single common
# release origin; the raw spawn/attach/release/exit timestamps are kept in
# the per-arm meta.json and events.log so setup and release skew stay explicit.
COMBINED_CSV_COLUMNS = [
    "block", "arm",
    "high_pid", "low_pid",
    "t_spawn_high", "t_spawn_low",
    "t_attach_sched", "t_attach_mem",
    "t_release", "t_cont_high", "t_cont_low",
    "t_exit_high", "t_exit_low",
    "high_median_ms", "low_median_ms",
    "high_bw_gbps", "low_bw_gbps",
    "high_latency_s", "low_latency_s",
    "high_rc", "low_rc",
    "notes",
]


# Processes this runner started. Cleanup targets only these (never a global
# pkill or a global struct-ops sweep, which could remove another session's
# live policy, e.g. LMCache).
_OWN_PROCS = set()


def _track(proc):
    """Register a Popen handle so cleanup only ever touches our own children."""
    if proc is not None:
        _OWN_PROCS.add(proc)
    return proc


def stop_proc(proc, label="", notes=None):
    """Stop one of our own processes: SIGINT, then SIGKILL if it lingers."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass
    if notes is not None:
        notes.append(f"{label}_stopped rc={proc.returncode}")
    _OWN_PROCS.discard(proc)


def cleanup_processes():
    """Stop only the processes this runner started. No global pkill / struct-ops."""
    for proc in list(_OWN_PROCS):
        stop_proc(proc)
    _OWN_PROCS.clear()


def run_uvmbench(output_file, size_factor, kernel):
    """Start a uvmbench process."""
    cmd = [
        str(UVM),
        f"--size_factor={size_factor}",
        "--mode=uvm",
        f"--iterations={ITERATIONS}",
        f"--kernel={kernel}",
    ]
    with open(output_file, 'w') as f:
        proc = _track(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT))
    return proc


def parse_uvmbench_output(output_file):
    """Parse uvmbench output to extract median time and bandwidth."""
    median_ms = 0.0
    bw_gbps = 0.0

    try:
        with open(output_file, 'r') as f:
            content = f.read()

        # Parse "Median time: X.XXX ms"
        match = re.search(r'Median time:\s+([\d.]+)', content)
        if match:
            median_ms = float(match.group(1))

        # Parse "Bandwidth: X.XX GB/s"
        match = re.search(r'Bandwidth:\s+([\d.]+)', content)
        if match:
            bw_gbps = float(match.group(1))
    except Exception as e:
        print(f"  Warning: Failed to parse {output_file}: {e}")

    return median_ms, bw_gbps


def proc_state(pid):
    """Return the kernel state letter for pid from /proc/<pid>/stat."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read().decode(errors="replace")
        return data.rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return "?"


def wait_stopped(pid, timeout_s=5.0):
    """Wait until a spawned tenant reaches the stopped (pre-exec) state."""
    deadline = time.time() + timeout_s
    state = proc_state(pid)
    while state not in ("T", "Z", "X") and time.time() < deadline:
        time.sleep(0.05)
        state = proc_state(pid)
    return state


def ensure_symlinks(link_dir, uvm_path):
    """Create run-owned tenant symlinks whose basenames key the scheduler comm.

    The links live in this run's own directory, so we never unlink or clobber a
    foreign /tmp file. The scheduler policy matches on the link basename only,
    so the directory location is irrelevant to it.
    """
    link_dir.mkdir(parents=True, exist_ok=True)
    for name in (HIGH_NAME, LOW_NAME):
        link = link_dir / name
        try:
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise FileExistsError(f"{link} exists and is not a symlink")
            link.symlink_to(uvm_path)
        except OSError as exc:
            print(f"  Warning: symlink {link} failed: {exc}")


def spawn_stopped_tenant(name, link_dir, arm_dir, size_factor, kernel, tenant_launcher):
    """Spawn a tenant that stops before exec; the PID is preserved across exec."""
    link = link_dir / name
    # Distinct per-tenant result file; both children share cwd=arm_dir, so the
    # default results.csv would otherwise be written by both (as in fig13-fast).
    out_csv = arm_dir / f"uvmbench_{name}_results.csv"
    argv = [
        str(link),
        f"--size_factor={size_factor}",
        *COMBINED_UVM_ARGS,
        f"--iterations={ITERATIONS}",
        f"--kernel={kernel}",
        f"--output={out_csv}",
    ]
    cmd = [sys.executable, str(tenant_launcher)] + argv
    log_path = arm_dir / f"tenant_{name}.log"
    logf = open(log_path, "w", buffering=1)
    proc = _track(subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT,
        cwd=str(arm_dir), start_new_session=True,
    ))
    return proc, logf, cmd


def combined_tool_command(label, high_pid, low_pid, args):
    """Build the argv for a policy tool (comm-keyed sched, PID-keyed mem)."""
    if label == "sched":
        return [
            "sudo", str(args.sched_tool),
            "-p", f"{HIGH_NAME}:{COMBINED_SCHED_HIGH_TS}",
            "-p", f"{LOW_NAME}:{COMBINED_SCHED_LOW_TS}",
        ]
    return [
        "sudo", str(args.mem_tool),
        "-p", str(high_pid), "-P", str(COMBINED_MEM_HIGH),
        "-l", str(low_pid), "-L", str(COMBINED_MEM_LOW),
    ]


def spawn_tool(cmd, log_path, label):
    """Start a policy tool in its own session and let it settle."""
    logf = open(log_path, "w", buffering=1)
    proc = _track(subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
    ))
    time.sleep(COMBINED_TOOL_SETTLE_S)
    if proc.poll() is not None:
        print(f"  Warning: {label} tool exited early rc={proc.returncode} (continuing)")
    return proc, logf


def watch_tenant(proc, role, results):
    """Independently observe one tenant's completion in its own thread."""
    rc = proc.wait()
    results[role] = {"rc": rc, "t_done": time.time()}


def run_combined_arm(block, arm, size_factor, kernel, run_root, link_dir, args, log):
    """Run one arm: spawn stopped, attach policies, common release, observe exits."""
    arm_dir = run_root / f"block{block:02d}_{arm}"
    arm_dir.mkdir(parents=True, exist_ok=True)

    row = {c: "" for c in COMBINED_CSV_COLUMNS}
    row["block"] = block
    row["arm"] = arm
    meta = {
        "block": block,
        "arm": arm,
        "kernel": kernel,
        "size_factor": size_factor,
        "iterations": ITERATIONS,
        "high_name": HIGH_NAME,
        "low_name": LOW_NAME,
        "mem_params": f"high={COMBINED_MEM_HIGH}/low={COMBINED_MEM_LOW}",
        "sched_params": f"high={COMBINED_SCHED_HIGH_TS}us/low={COMBINED_SCHED_LOW_TS}us",
        "mem_tool": str(args.mem_tool),
        "sched_tool": str(args.sched_tool),
        "tenant_launcher": str(args.tenant_launcher),
        "uvmbench": str(args.uvmbench),
        "common_release_origin": True,
        "independent_exit_observation": True,
    }
    notes = []
    high = low = None
    tools = {}
    files = []
    results = {}
    t_release = None
    failed = False

    try:
        ensure_symlinks(link_dir, args.uvmbench)

        high, high_logf, high_cmd = spawn_stopped_tenant(HIGH_NAME, link_dir, arm_dir, size_factor, kernel, args.tenant_launcher)
        t_spawn_high = time.time()
        low, low_logf, low_cmd = spawn_stopped_tenant(LOW_NAME, link_dir, arm_dir, size_factor, kernel, args.tenant_launcher)
        t_spawn_low = time.time()
        files.extend([high_logf, low_logf])
        meta["command_high"] = " ".join(high_cmd)
        meta["command_low"] = " ".join(low_cmd)

        row["high_pid"] = high.pid
        row["low_pid"] = low.pid
        row["t_spawn_high"] = f"{t_spawn_high:.6f}"
        row["t_spawn_low"] = f"{t_spawn_low:.6f}"
        meta["t_spawn_high"] = t_spawn_high
        meta["t_spawn_low"] = t_spawn_low

        state_high = wait_stopped(high.pid)
        state_low = wait_stopped(low.pid)
        meta["states_after_spawn"] = {"high": state_high, "low": state_low}
        if state_high != "T":
            notes.append(f"{HIGH_NAME}_not_stopped_pre_policy:{state_high}")
        if state_low != "T":
            notes.append(f"{LOW_NAME}_not_stopped_pre_policy:{state_low}")

        # Start policy tools while tenants are still stopped (before CUDA init).
        for label in COMBINED_ARM_TOOLS[arm]:
            cmd = combined_tool_command(label, high.pid, low.pid, args)
            proc, logf = spawn_tool(cmd, arm_dir / f"{label}_tool.log", label)
            files.append(logf)
            tools[label] = proc
            t_attach = time.time()
            meta[f"t_{label}_start"] = t_attach
            meta[f"command_{label}"] = " ".join(cmd)
            row[f"t_attach_{label}"] = f"{t_attach:.6f}"

        # Single common release origin: one timestamp, then resume both tenants.
        t_release = time.time()
        os.kill(high.pid, signal.SIGCONT)
        t_cont_high = time.time()
        os.kill(low.pid, signal.SIGCONT)
        t_cont_low = time.time()
        row["t_release"] = f"{t_release:.6f}"
        row["t_cont_high"] = f"{t_cont_high:.6f}"
        row["t_cont_low"] = f"{t_cont_low:.6f}"
        meta["t_release"] = t_release
        meta["t_cont_high"] = t_cont_high
        meta["t_cont_low"] = t_cont_low
        log.info(f"block{block:02d}_{arm} release t={t_release:.6f} "
                 f"cont_high={t_cont_high:.6f} cont_low={t_cont_low:.6f}")

        # Observe each child's completion independently (no sequential wait).
        threads = []
        for proc, role in ((high, "high"), (low, "low")):
            th = threading.Thread(target=watch_tenant, args=(proc, role, results))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        # Stop our own policy tools (own attachments only).
        for label in ("sched", "mem"):
            if label in tools:
                stop_proc(tools[label], f"{label}_tool", notes)
                meta[f"rc_{label}_tool"] = tools[label].returncode

        for role in ("high", "low"):
            r = results.get(role)
            if r is None or t_release is None:
                notes.append(f"{role}_no_result")
                continue
            row[f"t_exit_{role}"] = f"{r['t_done']:.6f}"
            row[f"{role}_latency_s"] = f"{r['t_done'] - t_release:.6f}"
            row[f"{role}_rc"] = r["rc"] if r["rc"] is not None else ""
            meta[f"t_done_{role}"] = r["t_done"]
            meta[f"rc_{role}"] = r["rc"]

        # Parse per-tenant stdout (retained in the arm dir) for median/bandwidth.
        for role, name in (("high", HIGH_NAME), ("low", LOW_NAME)):
            median, bw = parse_uvmbench_output(arm_dir / f"tenant_{name}.log")
            row[f"{role}_median_ms"] = median
            row[f"{role}_bw_gbps"] = bw

        row["notes"] = ";".join(notes)
        meta["notes"] = notes
    except Exception as exc:
        failed = True
        notes.append(f"harness_error:{type(exc).__name__}:{exc}")
        print(f"  harness error in {arm_dir}: {exc!r}")
    finally:
        # Clean up only our own processes/attachments.
        stop_proc(high, "high")
        stop_proc(low, "low")
        for label, p in tools.items():
            stop_proc(p, f"{label}_tool")
        for f in files:
            try:
                f.close()
            except OSError:
                pass
        # Always persist the final notes and failure status, even on error.
        meta["status"] = "error" if failed else "ok"
        meta["notes"] = notes
        row["notes"] = ";".join(notes)
        try:
            (arm_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        except OSError as exc:
            print(f"  meta.json write failed: {exc}")
    return row


def run_combined_mode(args, kernel, size_factor, output_dir):
    """Run the opt-in four-arm combined comparison for one kernel.

    Writes into a fresh timestamped run subdirectory under output_dir so a
    reused --output never overwrites earlier raw results (CSV, run.json,
    events.log, per-arm tenant/tool logs, meta.json).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / f"combined_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    link_dir = run_root / "tenant_links"
    csv_path = run_root / "combined_comparison.csv"

    # Rotate the arm order so every arm occupies every position across blocks.
    arm_orders = {}
    for b in range(args.blocks):
        start = b % len(COMBINED_ARMS)
        arm_orders[b] = list(COMBINED_ARMS[start:] + COMBINED_ARMS[:start])

    run_cfg = {
        "mode": "combined",
        "started_utc": datetime.now().isoformat(),
        "kernel": kernel,
        "size_factor": size_factor,
        "iterations": ITERATIONS,
        "tenants": 2,
        "blocks": args.blocks,
        "arms": list(COMBINED_ARMS),
        "arm_orders": arm_orders,
        "output_dir": str(output_dir),
        "run_root": str(run_root),
        "link_dir": str(link_dir),
        "mem_params": f"high={COMBINED_MEM_HIGH}/low={COMBINED_MEM_LOW}",
        "sched_params": f"high={COMBINED_SCHED_HIGH_TS}us/low={COMBINED_SCHED_LOW_TS}us",
        "mem_tool": str(args.mem_tool),
        "sched_tool": str(args.sched_tool),
        "tenant_launcher": str(args.tenant_launcher),
        "uvmbench": str(args.uvmbench),
        "common_release_origin": True,
        "independent_exit_observation": True,
        "cleanup": "own processes/attachments only; no global pkill/struct-ops",
    }
    (run_root / "run.json").write_text(json.dumps(run_cfg, indent=2))

    events_fh = open(run_root / "events.log", "w", buffering=1)

    class RunLog:
        def info(self, msg):
            line = f"{datetime.now().isoformat()} {msg}"
            print(line, flush=True)
            events_fh.write(line + "\n")
            events_fh.flush()

    log = RunLog()
    log.info(f"combined start kernel={kernel} size_factor={size_factor} "
             f"blocks={args.blocks} out={run_root}")

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=COMBINED_CSV_COLUMNS).writeheader()

    for block, order in arm_orders.items():
        log.info(f"block {block} arm order: {','.join(order)}")
        for arm in order:
            log.info(f"--- block {block} arm {arm} ---")
            row = run_combined_arm(block, arm, size_factor, kernel, run_root, link_dir, args, log)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=COMBINED_CSV_COLUMNS).writerow(row)

    events_fh.close()
    print(f"\nResults saved to: {run_root}")
    print(f"CSV: {csv_path}")
    return csv_path


def run_experiment(policy_name, policy_binary, high_param, low_param, round_idx, size_factor, kernel, output_dir):
    """Run a single experiment with the given policy configuration."""

    cleanup_processes()

    # Create temp files for output
    high_output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    low_output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    high_output.close()
    low_output.close()

    policy_proc = None
    policy_output = None

    try:
        # Record start time
        start_time = time.time()

        # Start both uvmbench processes
        high_proc = run_uvmbench(high_output.name, size_factor, kernel)
        low_proc = run_uvmbench(low_output.name, size_factor, kernel)

        # Start policy process if needed
        if policy_binary:
            policy_path = SRC / policy_binary
            policy_output = output_dir / f"{policy_binary}_{high_param}_{low_param}_r{round_idx+1}.txt"

            cmd = [
                "sudo", str(policy_path),
                "-p", str(high_proc.pid), "-P", str(high_param),
                "-l", str(low_proc.pid), "-L", str(low_param),
            ]

            with open(policy_output, 'w') as f:
                policy_proc = _track(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT))
            time.sleep(1)

        # Wait for uvmbench to complete, record end times
        high_end_time = None
        low_end_time = None

        while high_proc.poll() is None or low_proc.poll() is None:
            if high_end_time is None and high_proc.poll() is not None:
                high_end_time = time.time()
            if low_end_time is None and low_proc.poll() is not None:
                low_end_time = time.time()
            time.sleep(0.01)

        # Ensure end times are recorded
        if high_end_time is None:
            high_end_time = time.time()
        if low_end_time is None:
            low_end_time = time.time()

        # Calculate latency (seconds)
        high_latency = high_end_time - start_time
        low_latency = low_end_time - start_time

        # Calculate throughput (iterations per second)
        high_throughput = ITERATIONS / high_latency if high_latency > 0 else 0
        low_throughput = ITERATIONS / low_latency if low_latency > 0 else 0

        # Stop policy process
        if policy_proc:
            policy_proc.send_signal(signal.SIGINT)
            try:
                policy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                policy_proc.kill()
                policy_proc.wait()

        # Parse results
        high_median, high_bw = parse_uvmbench_output(high_output.name)
        low_median, low_bw = parse_uvmbench_output(low_output.name)

        return {
            'high_median_ms': high_median,
            'high_bw_gbps': high_bw,
            'high_latency_s': high_latency,
            'high_throughput': high_throughput,
            'low_median_ms': low_median,
            'low_bw_gbps': low_bw,
            'low_latency_s': low_latency,
            'low_throughput': low_throughput,
        }

    finally:
        # Cleanup temp files
        os.unlink(high_output.name)
        os.unlink(low_output.name)


def run_single_experiment(config_name, size_multiplier, round_idx, size_factor, kernel):
    """Run a single process experiment without any policy."""

    cleanup_processes()

    # Create temp file for output
    output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    output.close()

    try:
        # Calculate actual size factor
        actual_size_factor = size_factor * size_multiplier

        # Record start time
        start_time = time.time()

        # Start single uvmbench process
        proc = run_uvmbench(output.name, actual_size_factor, kernel)

        # Wait for uvmbench to complete
        proc.wait()
        end_time = time.time()

        # Calculate latency (seconds)
        latency = end_time - start_time

        # Calculate throughput (iterations per second)
        throughput = ITERATIONS / latency if latency > 0 else 0

        # Parse results
        median_ms, bw_gbps = parse_uvmbench_output(output.name)

        return {
            'median_ms': median_ms,
            'bw_gbps': bw_gbps,
            'latency_s': latency,
            'throughput': throughput,
        }

    finally:
        # Cleanup temp file
        os.unlink(output.name)


def warmup(size_factor, kernel):
    """Run warmup iteration."""
    print("=== WARMUP ===")
    cmd = [
        str(UVM),
        f"--size_factor={size_factor}",
        "--mode=uvm",
        "--iterations=2",
        f"--kernel={kernel}",
    ]
    subprocess.run(cmd, capture_output=True)
    time.sleep(2)


def load_completed_tests(csv_path):
    """Load completed tests from existing CSV file.

    Returns a set of (policy_name, high_param, low_param, round) tuples.
    For single process tests, high_param and low_param will be empty strings.
    """
    completed = set()
    if not csv_path or not Path(csv_path).exists():
        return completed

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 12:
            policy = parts[0]
            high_param = parts[1]  # Keep as string (may be empty for single process)
            low_param = parts[2]
            round_num = int(parts[11])
            completed.add((policy, high_param, low_param, round_num))

    return completed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Policy Comparison Evaluation')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to existing CSV file to resume from')
    parser.add_argument('--size-factor', type=float, default=DEFAULT_SIZE_FACTOR,
                        help=f'Size factor for uvmbench (default: {DEFAULT_SIZE_FACTOR})')
    parser.add_argument('--kernel', type=str, default=DEFAULT_KERNEL,
                        choices=['rand_stream', 'seq_stream', 'hotspot', 'gemm', 'kmeans_sparse'],
                        help=f'Kernel to run (default: {DEFAULT_KERNEL})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help=f'Output directory for results (default: results)')
    parser.add_argument('--combined', action='store_true',
                        help='Run the opt-in four-arm combined memory/scheduling '
                             'comparison (baseline, memory-only, sched-only, combined)')
    parser.add_argument('--blocks', type=int, default=COMBINED_BLOCKS,
                        help=f'Interleaved blocks for --combined (default: {COMBINED_BLOCKS})')
    parser.add_argument('--mem-tool', type=str, default=str(SRC / "prefetch_eviction_pid"),
                        help='Path to the memory policy tool')
    parser.add_argument('--sched-tool', type=str, default=str(SRC / "gpu_sched_set_timeslices"),
                        help='Path to the scheduler policy tool')
    parser.add_argument('--uvmbench', type=str, default=str(UVM),
                        help='Path to the uvmbench binary')
    parser.add_argument('--tenant-launcher', type=str, default=str(TENANT_LAUNCHER),
                        help='Path to the stop-before-exec tenant launcher')
    return parser.parse_args()


def run_combined(args):
    """Dispatch the opt-in four-arm combined comparison for one kernel."""
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir
    else:
        output_dir = Path(__file__).parent / "results_combined"

    required = {
        "uvmbench": Path(args.uvmbench),
        "mem_tool": Path(args.mem_tool),
        "sched_tool": Path(args.sched_tool),
        "tenant_launcher": Path(args.tenant_launcher),
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        print("Error: combined mode is missing required tools:\n  " + "\n  ".join(missing))
        sys.exit(1)

    size_factor = args.size_factor
    kernel = args.kernel
    print("=" * 60)
    print("Combined Four-Arm Comparison (ORIGINAL Fig.13 follow-up)")
    print("=" * 60)
    print(f"Kernel: {kernel}, Size Factor: {size_factor}, Iterations: {ITERATIONS}")
    print(f"Arms: {', '.join(COMBINED_ARMS)}, Blocks: {args.blocks}")
    print(f"Memory policy: prefetch_eviction_pid high={COMBINED_MEM_HIGH}/low={COMBINED_MEM_LOW}")
    print(f"Sched policy:  gpu_sched_set_timeslices high={COMBINED_SCHED_HIGH_TS}us/low={COMBINED_SCHED_LOW_TS}us")
    print(f"Output: {output_dir}")
    print()

    run_combined_mode(args, kernel, size_factor, output_dir)


def main():
    args = parse_args()

    # Check uvmbench exists
    if not UVM.exists():
        print(f"Error: {UVM} not found")
        sys.exit(1)

    if args.combined:
        run_combined(args)
        return

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle resume mode
    completed_tests = set()
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Error: Resume file not found: {args.resume}")
            sys.exit(1)
        completed_tests = load_completed_tests(resume_path)
        csv_path = resume_path
        print(f"Resuming from: {csv_path}")
        print(f"Found {len(completed_tests)} completed tests")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"policy_comparison_{timestamp}.csv"
        # Write CSV header for new file
        with open(csv_path, 'w') as f:
            f.write("policy,high_param,low_param,high_median_ms,high_bw_gbps,high_latency_s,high_throughput,low_median_ms,low_bw_gbps,low_latency_s,low_throughput,round\n")

    # Get configuration from args
    size_factor = args.size_factor
    kernel = args.kernel

    print("=" * 60)
    print("Policy Comparison Evaluation")
    print("=" * 60)
    print(f"Kernel: {kernel}, Size Factor: {size_factor}")
    print(f"Output: {csv_path}")
    print()

    # Count remaining tests
    remaining = 0
    # Count dual-process tests
    for policy_name, policy_binary, configs in POLICIES:
        for high_param, low_param in configs:
            for round_idx in range(NUM_ROUNDS):
                if (policy_name, str(high_param), str(low_param), round_idx + 1) not in completed_tests:
                    remaining += 1
    # Count single-process tests
    for config_name, size_multiplier in SINGLE_PROCESS_CONFIGS:
        for round_idx in range(NUM_ROUNDS):
            if (config_name, "", "", round_idx + 1) not in completed_tests:
                remaining += 1
    print(f"Remaining tests: {remaining}")
    print()

    # Warmup
    warmup(size_factor, kernel)

    # Run all experiments
    results = []

    # Run single-process experiments first
    for config_name, size_multiplier in SINGLE_PROCESS_CONFIGS:
        for round_idx in range(NUM_ROUNDS):
            # Skip completed tests
            if (config_name, "", "", round_idx + 1) in completed_tests:
                print(f"=== SKIP {config_name} R{round_idx+1} (already done) ===")
                continue

            exp_name = f"{config_name} (size={size_factor * size_multiplier}) R{round_idx+1}"
            print(f"=== {exp_name} ===")

            result = run_single_experiment(config_name, size_multiplier, round_idx, size_factor, kernel)

            # Print result
            print(f"  {result['median_ms']:.2f}ms {result['bw_gbps']:.2f}GB/s "
                  f"latency={result['latency_s']:.2f}s throughput={result['throughput']:.2f}/s")

            # Write to CSV (leave high_param, low_param, low_* fields empty)
            with open(csv_path, 'a') as f:
                f.write(f"{config_name},,,"
                       f"{result['median_ms']},{result['bw_gbps']},"
                       f"{result['latency_s']},{result['throughput']},"
                       f",,,,{round_idx+1}\n")

            results.append({
                'policy': config_name,
                'high_param': '',
                'low_param': '',
                'round': round_idx + 1,
                'high_median_ms': result['median_ms'],
                'high_bw_gbps': result['bw_gbps'],
                'high_latency_s': result['latency_s'],
                'high_throughput': result['throughput'],
                'low_median_ms': 0,
                'low_bw_gbps': 0,
                'low_latency_s': 0,
                'low_throughput': 0,
            })

    # Run dual-process experiments
    for policy_name, policy_binary, configs in POLICIES:
        for high_param, low_param in configs:
            for round_idx in range(NUM_ROUNDS):
                # Skip completed tests
                if (policy_name, str(high_param), str(low_param), round_idx + 1) in completed_tests:
                    print(f"=== SKIP {policy_name} {high_param}/{low_param} R{round_idx+1} (already done) ===")
                    continue

                exp_name = f"{policy_name} {high_param}/{low_param} R{round_idx+1}"
                print(f"=== {exp_name} ===")

                result = run_experiment(policy_name, policy_binary, high_param, low_param, round_idx, size_factor, kernel, output_dir)

                # Print result
                print(f"  H:{result['high_median_ms']:.2f}ms {result['high_bw_gbps']:.2f}GB/s "
                      f"lat={result['high_latency_s']:.2f}s tput={result['high_throughput']:.2f}/s")
                print(f"  L:{result['low_median_ms']:.2f}ms {result['low_bw_gbps']:.2f}GB/s "
                      f"lat={result['low_latency_s']:.2f}s tput={result['low_throughput']:.2f}/s")

                # Write to CSV
                with open(csv_path, 'a') as f:
                    f.write(f"{policy_name},{high_param},{low_param},"
                           f"{result['high_median_ms']},{result['high_bw_gbps']},"
                           f"{result['high_latency_s']},{result['high_throughput']},"
                           f"{result['low_median_ms']},{result['low_bw_gbps']},"
                           f"{result['low_latency_s']},{result['low_throughput']},"
                           f"{round_idx+1}\n")

                results.append({
                    'policy': policy_name,
                    'high_param': high_param,
                    'low_param': low_param,
                    'round': round_idx + 1,
                    **result,
                })

    # Cleanup
    cleanup_processes()

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Load all results from CSV for summary (including previously completed)
    from collections import defaultdict
    all_results = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 12:
            policy = parts[0]
            # Handle single process tests (empty params)
            high_param = parts[1] if parts[1] else ''
            low_param = parts[2] if parts[2] else ''
            high_median = float(parts[3]) if parts[3] else 0.0
            high_bw = float(parts[4]) if parts[4] else 0.0
            high_latency = float(parts[5]) if parts[5] else 0.0
            high_throughput = float(parts[6]) if parts[6] else 0.0
            low_median = float(parts[7]) if parts[7] else 0.0
            low_bw = float(parts[8]) if parts[8] else 0.0
            low_latency = float(parts[9]) if parts[9] else 0.0
            low_throughput = float(parts[10]) if parts[10] else 0.0
            all_results.append({
                'policy': policy,
                'high_param': high_param,
                'low_param': low_param,
                'high_median_ms': high_median,
                'high_bw_gbps': high_bw,
                'high_latency_s': high_latency,
                'high_throughput': high_throughput,
                'low_median_ms': low_median,
                'low_bw_gbps': low_bw,
                'low_latency_s': low_latency,
                'low_throughput': low_throughput,
            })

    # Group by policy and config
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['policy'], r['high_param'], r['low_param'])
        grouped[key].append(r)

    print(f"{'Policy':<22} {'Params':<8} {'H_Lat(s)':<10} {'H_Tput':<10} {'L_Lat(s)':<10} {'L_Tput':<10}")
    print("-" * 75)

    for (policy, hp, lp), runs in grouped.items():
        h_lat_avg = sum(r['high_latency_s'] for r in runs) / len(runs)
        h_tput_avg = sum(r['high_throughput'] for r in runs) / len(runs)
        l_lat_avg = sum(r['low_latency_s'] for r in runs) / len(runs)
        l_tput_avg = sum(r['low_throughput'] for r in runs) / len(runs)

        # Format params display
        if hp == '' and lp == '':
            params_str = "(single)"
        else:
            params_str = f"{hp}/{lp}"

        print(f"{policy:<22} {params_str:<8} {h_lat_avg:<10.2f} {h_tput_avg:<10.2f} {l_lat_avg:<10.2f} {l_tput_avg:<10.2f}")

    print()
    print(f"Results saved to: {csv_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
