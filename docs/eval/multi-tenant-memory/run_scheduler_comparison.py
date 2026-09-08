#!/usr/bin/env python3
"""
Scheduler Policy Comparison Evaluation Script

Tests scheduler timeslice policies with the SAME uvmbench workloads as memory policy.
This allows comparison: Does scheduler alone solve memory contention?

Usage:
  python run_scheduler_comparison.py --kernel hotspot --size-factor 0.6 --output results_hotspot
  python run_scheduler_comparison.py --kernel gemm --size-factor 0.6 --output results_gemm
  python run_scheduler_comparison.py --kernel kmeans_sparse --size-factor 0.9 --output results_kmeans
"""

import subprocess
import tempfile
import time
import re
import os
import signal
import argparse
import sys
import json
import threading
from pathlib import Path
from datetime import datetime

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "extension"
UVM = REPO_ROOT / "microbench" / "memory" / "uvmbench"
TENANT_LAUNCHER = REPO_ROOT / "workloads" / "fig13-fast" / "tenant_launcher.py"
SCHED_POLICY = SRC / "gpu_sched_set_timeslices"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results_sched"

# Benchmark parameters
DEFAULT_SIZE_FACTOR = 0.6
DEFAULT_KERNEL = "hotspot"
ITERATIONS = 1
NUM_ROUNDS = 1

# Scheduler timeslice settings (microseconds)
HIGH_TIMESLICE = 1_000_000  # 1 second for high priority
LOW_TIMESLICE = 200         # 200 microseconds for low priority

# Scheduler policy configurations to test
SCHED_POLICIES = [
    # (policy_name, use_sched_policy, high_timeslice, low_timeslice)
    ("no_policy", False, None, None),
    ("sched_timeslice", True, HIGH_TIMESLICE, LOW_TIMESLICE),
]

# Single process configurations
SINGLE_PROCESS_CONFIGS = [
    ("single_1x", 1),
    # ("single_2x", 2),
]

# Tenant comm names (via run-owned symlinks) used to key the scheduler policy.
HIGH_NAME = "uvmbench_high"
LOW_NAME = "uvmbench_low"


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


def watch_tenant(proc, role, results):
    """Independently observe one tenant's completion in its own thread."""
    rc = proc.wait()
    results[role] = {"rc": rc, "t_done": time.time()}


def parse_output(output_file):
    """Parse uvmbench output to extract metrics."""
    median_ms = 0
    bw_gbps = 0

    try:
        with open(output_file, 'r') as f:
            content = f.read()

        match = re.search(r'Median time:\s*([\d.]+)\s*ms', content)
        if match:
            median_ms = float(match.group(1))

        match = re.search(r'Bandwidth:\s*([\d.]+)\s*GB/s', content)
        if match:
            bw_gbps = float(match.group(1))
    except Exception as e:
        print(f"  Warning: Failed to parse {output_file}: {e}")

    return median_ms, bw_gbps


def spawn_stopped_tenant(name, link_dir, run_root, size_factor, kernel):
    """Spawn a tenant that stops before exec; the PID is preserved across exec."""
    link = link_dir / name
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a symlink")
    link.symlink_to(UVM)
    out_csv = run_root / f"uvmbench_{name}_results.csv"
    argv = [
        str(link),
        f"--size_factor={size_factor}",
        "--mode=uvm",
        f"--iterations={ITERATIONS}",
        f"--kernel={kernel}",
        f"--output={out_csv}",
    ]
    cmd = [sys.executable, str(TENANT_LAUNCHER)] + argv
    log_path = run_root / f"tenant_{name}.log"
    logf = open(log_path, "w", buffering=1)
    proc = _track(subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT,
        cwd=str(run_root), start_new_session=True,
    ))
    return proc, logf


def run_experiment(policy_name, use_sched_policy, high_ts, low_ts, round_idx, size_factor, kernel, output_dir):
    """Run one scheduler-policy experiment: stop-before-exec, common release,
    independent completion, own-process cleanup only.

    Latency is measured from the single common release origin (SIGCONT), not the
    old pre-Popen start; raw tenant/tool logs and timing metadata are kept in a
    fresh run_root subdirectory.
    """
    cleanup_processes()

    run_root = output_dir / f"round{round_idx + 1}_{policy_name}"
    run_root.mkdir(parents=True, exist_ok=True)
    link_dir = run_root / "tenant_links"
    link_dir.mkdir(parents=True, exist_ok=True)

    sched_proc = None
    high = low = None
    files = []
    results = {}
    t_release = None
    notes = []
    failed = False
    meta = {
        "policy": policy_name,
        "use_sched_policy": use_sched_policy,
        "high_timeslice": high_ts,
        "low_timeslice": low_ts,
        "round": round_idx + 1,
        "kernel": kernel,
        "size_factor": size_factor,
        "iterations": ITERATIONS,
        "common_release_origin": True,
        "independent_exit_observation": True,
        "timing_note": "latency measured from the common release origin, not the pre-Popen start",
    }

    try:
        high, high_logf = spawn_stopped_tenant(HIGH_NAME, link_dir, run_root, size_factor, kernel)
        t_spawn_high = time.time()
        low, low_logf = spawn_stopped_tenant(LOW_NAME, link_dir, run_root, size_factor, kernel)
        t_spawn_low = time.time()
        files.extend([high_logf, low_logf])
        meta["high_pid"] = high.pid
        meta["low_pid"] = low.pid
        meta["t_spawn_high"] = t_spawn_high
        meta["t_spawn_low"] = t_spawn_low

        state_high = wait_stopped(high.pid)
        state_low = wait_stopped(low.pid)
        meta["states_after_spawn"] = {"high": state_high, "low": state_low}
        if state_high != "T":
            notes.append(f"{HIGH_NAME}_not_stopped_pre_policy:{state_high}")
        if state_low != "T":
            notes.append(f"{LOW_NAME}_not_stopped_pre_policy:{state_low}")

        # Start the scheduler policy while tenants are stopped (before CUDA init).
        if use_sched_policy and SCHED_POLICY.exists():
            cmd = [
                "sudo", str(SCHED_POLICY),
                "-p", f"{HIGH_NAME}:{high_ts}",
                "-p", f"{LOW_NAME}:{low_ts}",
            ]
            sched_log = open(run_root / "sched_tool.log", "w", buffering=1)
            files.append(sched_log)
            sched_proc = _track(subprocess.Popen(
                cmd, stdout=sched_log, stderr=subprocess.STDOUT, start_new_session=True,
            ))
            time.sleep(1.0)
            meta["t_sched_start"] = time.time()
            meta["command_sched"] = " ".join(cmd)
            if sched_proc.poll() is not None:
                notes.append(f"sched_tool_exited_early rc={sched_proc.returncode}")

        # Single common release origin: one timestamp, then resume both tenants.
        t_release = time.time()
        os.kill(high.pid, signal.SIGCONT)
        t_cont_high = time.time()
        os.kill(low.pid, signal.SIGCONT)
        t_cont_low = time.time()
        meta["t_release"] = t_release
        meta["t_cont_high"] = t_cont_high
        meta["t_cont_low"] = t_cont_low

        # Observe each child's completion independently (no sequential wait).
        threads = []
        for proc, role in ((high, "high"), (low, "low")):
            th = threading.Thread(target=watch_tenant, args=(proc, role, results))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        # Stop our own scheduler tool (own attachment only).
        if sched_proc is not None:
            stop_proc(sched_proc, "sched_tool", notes)
            meta["rc_sched_tool"] = sched_proc.returncode

        if "high" not in results or "low" not in results or t_release is None:
            raise RuntimeError("missing tenant completion or release timestamp")

        high_latency = results["high"]["t_done"] - t_release
        low_latency = results["low"]["t_done"] - t_release
        meta["t_done_high"] = results["high"]["t_done"]
        meta["t_done_low"] = results["low"]["t_done"]
        meta["rc_high"] = results["high"]["rc"]
        meta["rc_low"] = results["low"]["rc"]

        high_median, high_bw = parse_output(run_root / f"tenant_{HIGH_NAME}.log")
        low_median, low_bw = parse_output(run_root / f"tenant_{LOW_NAME}.log")

        return {
            'policy': policy_name,
            'high_param': high_ts if high_ts else '',
            'low_param': low_ts if low_ts else '',
            'high_median_ms': high_median,
            'high_bw_gbps': high_bw,
            'high_latency_s': high_latency,
            'high_throughput': 1.0 / high_latency if high_latency > 0 else 0,
            'low_median_ms': low_median,
            'low_bw_gbps': low_bw,
            'low_latency_s': low_latency,
            'low_throughput': 1.0 / low_latency if low_latency > 0 else 0,
            'round': round_idx + 1,
        }

    except Exception as exc:
        failed = True
        notes.append(f"harness_error:{type(exc).__name__}:{exc}")
        print(f"  harness error in {run_root}: {exc!r}")
        return None
    finally:
        # Clean up only our own processes/attachments.
        stop_proc(high, "high")
        stop_proc(low, "low")
        stop_proc(sched_proc, "sched_tool")
        for f in files:
            try:
                f.close()
            except OSError:
                pass
        meta["status"] = "error" if failed else "ok"
        meta["notes"] = notes
        try:
            (run_root / "meta.json").write_text(json.dumps(meta, indent=2))
        except OSError as exc:
            print(f"  meta.json write failed: {exc}")


def run_single_experiment(config_name, size_multiplier, round_idx, size_factor, kernel):
    """Run a single process experiment."""
    cleanup_processes()

    output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    output.close()
    proc = None

    try:
        actual_size_factor = size_factor * size_multiplier
        start_time = time.time()

        # No policy is attached in the single-process baseline, so run uvmbench
        # directly (no comm-keying symlink, no stop-before-exec needed).
        cmd = [
            str(UVM),
            f"--size_factor={actual_size_factor}",
            "--mode=uvm",
            f"--iterations={ITERATIONS}",
            f"--kernel={kernel}",
        ]
        with open(output.name, 'w') as f:
            proc = _track(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT))
        proc.wait()
        end_time = time.time()

        latency = end_time - start_time
        median_ms, bw_gbps = parse_output(output.name)

        return {
            'policy': config_name,
            'high_param': '',
            'low_param': '',
            'high_median_ms': median_ms,
            'high_bw_gbps': bw_gbps,
            'high_latency_s': latency,
            'high_throughput': 1.0 / latency if latency > 0 else 0,
            'low_median_ms': '',
            'low_bw_gbps': '',
            'low_latency_s': '',
            'low_throughput': '',
            'round': round_idx + 1,
        }
    finally:
        stop_proc(proc, "single")
        try:
            os.unlink(output.name)
        except OSError:
            pass


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Scheduler Policy Comparison Evaluation')
    parser.add_argument('--size-factor', type=float, default=DEFAULT_SIZE_FACTOR,
                        help=f'Size factor for uvmbench (default: {DEFAULT_SIZE_FACTOR})')
    parser.add_argument('--kernel', type=str, default=DEFAULT_KERNEL,
                        choices=['rand_stream', 'seq_stream', 'hotspot', 'gemm', 'kmeans_sparse'],
                        help=f'Kernel to run (default: {DEFAULT_KERNEL})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results (default: results_sched)')
    return parser.parse_args()


def main():
    args = parse_args()

    if not UVM.exists():
        print(f"Error: {UVM} not found")
        sys.exit(1)

    if not SCHED_POLICY.exists():
        print(f"Error: {SCHED_POLICY} not found")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    size_factor = args.size_factor
    kernel = args.kernel

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"sched_comparison_{timestamp}.csv"

    with open(csv_path, 'w') as f:
        f.write("policy,high_param,low_param,high_median_ms,high_bw_gbps,high_latency_s,high_throughput,low_median_ms,low_bw_gbps,low_latency_s,low_throughput,round\n")

    print("=" * 60)
    print("Scheduler Policy Comparison Evaluation")
    print("=" * 60)
    print(f"Kernel: {kernel}, Size Factor: {size_factor}")
    print(f"Timeslice: High={HIGH_TIMESLICE}µs, Low={LOW_TIMESLICE}µs")
    print(f"Output: {csv_path}")
    print()

    warmup(size_factor, kernel)

    # Run single process baselines
    print("\n=== SINGLE PROCESS BASELINES ===")
    for config_name, size_multiplier in SINGLE_PROCESS_CONFIGS:
        for round_idx in range(NUM_ROUNDS):
            exp_name = f"{config_name} (size={size_factor * size_multiplier}) R{round_idx+1}"
            print(f"=== {exp_name} ===")

            result = run_single_experiment(config_name, size_multiplier, round_idx, size_factor, kernel)

            print(f"  {result['high_latency_s']:.2f}s, {result['high_median_ms']:.2f}ms, {result['high_bw_gbps']:.2f}GB/s")

            with open(csv_path, 'a') as f:
                f.write(f"{result['policy']},{result['high_param']},{result['low_param']},"
                       f"{result['high_median_ms']},{result['high_bw_gbps']},{result['high_latency_s']},{result['high_throughput']},"
                       f"{result['low_median_ms']},{result['low_bw_gbps']},{result['low_latency_s']},{result['low_throughput']},"
                       f"{result['round']}\n")

    # Run scheduler policy experiments
    print("\n=== SCHEDULER POLICY EXPERIMENTS ===")
    for policy_name, use_sched, high_ts, low_ts in SCHED_POLICIES:
        for round_idx in range(NUM_ROUNDS):
            ts_str = f"{high_ts}/{low_ts}" if high_ts else "none"
            exp_name = f"{policy_name} (ts={ts_str}) R{round_idx+1}"
            print(f"=== {exp_name} ===")

            result = run_experiment(policy_name, use_sched, high_ts, low_ts, round_idx, size_factor, kernel, output_dir)

            print(f"  H:{result['high_median_ms']:.2f}ms {result['high_bw_gbps']:.2f}GB/s "
                  f"lat={result['high_latency_s']:.2f}s")
            print(f"  L:{result['low_median_ms']:.2f}ms {result['low_bw_gbps']:.2f}GB/s "
                  f"lat={result['low_latency_s']:.2f}s")

            with open(csv_path, 'a') as f:
                f.write(f"{result['policy']},{result['high_param']},{result['low_param']},"
                       f"{result['high_median_ms']},{result['high_bw_gbps']},{result['high_latency_s']},{result['high_throughput']},"
                       f"{result['low_median_ms']},{result['low_bw_gbps']},{result['low_latency_s']},{result['low_throughput']},"
                       f"{result['round']}\n")

    print()
    print(f"Results saved to: {csv_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
