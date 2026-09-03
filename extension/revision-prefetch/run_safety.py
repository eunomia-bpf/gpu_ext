#!/usr/bin/env python3
"""Three closed functional controls; never reload a module or run a benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "workloads/uvm-policy-mechanism"))
import run_safe_policy_smoke as control
sys.path.insert(0, str(ROOT / "workloads/gpreempt"))
import run_three_way as owned

safety = control.shared
MODES = {"native": 0, "bypass": 1, "invalid99": 99}
OBSERVER_COUNT = 3
COMPUTE_MAX_GAP_NS = 1_000_000_000
LOADER = HERE / "build/prefetch_safety"
COMPUTE_MONITOR = HERE / "monitor_compute_apps.py"
HOOKS = ("uvm_bpf_call_gpu_page_prefetch", "uvm_bpf_prefetch_diagnostic")
INTERRUPTED_SIGNALS = []


def note_interrupt(signum, _frame):
    """Queue cancellation without ever interrupting owned-child cleanup."""
    INTERRUPTED_SIGNALS.append(signum)


def raise_if_interrupted():
    if INTERRUPTED_SIGNALS:
        raise InterruptedError(f"signal {INTERRUPTED_SIGNALS[0]}")


def stop_owned(process):
    """EB's poll-and-PGID check, retaining the existing Q2 grace periods."""
    if process is None:
        return
    for sig, seconds in ((signal.SIGINT, 8), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if process.poll() is not None and not owned.group_members(process.pid):
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if process.poll() is not None and not owned.group_members(process.pid):
                return
            time.sleep(0.05)
    if process.poll() is not None and not owned.group_members(process.pid):
        return
    raise RuntimeError(f"owned Q2 group {process.pid} survived cleanup")


def stop_monitors(monitors, record):
    """Attempt every owned monitor, recording failures before rejecting the cell."""
    attempts = record["monitor_cleanup"] = []
    for process in reversed(monitors):
        attempt = {"pid": process.pid, "error": None}
        try:
            stop_owned(process)
        except BaseException as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
        attempt["returncode"] = process.poll()
        attempts.append(attempt)
    if any(attempt["error"] for attempt in attempts):
        record["complete"] = False
        raise RuntimeError(f"owned monitor cleanup failed: {attempts}")


def runtime_files():
    result = []
    for path in (LOADER, HERE / "build/fixture.bpf.o", HERE / "fixture.bpf.c",
                 HERE / "fixture.h", HERE / "loader.c", Path(__file__), COMPUTE_MONITOR,
                 HERE.parent / "uvm_types.h", HERE.parent / "bpf_testmod.h",
                 control.WORKLOAD, control.HERE / "uvm_fault_stream.cu"):
        metric = control._file_metric(path)
        metric["mtime_ns"] = path.stat().st_mtime_ns
        result.append(metric)
    return result


def checked(argv, cwd=None, timeout=60):
    process = subprocess.Popen(argv, cwd=cwd, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    finally:
        stop_owned(process)
    if process.returncode:
        raise RuntimeError(f"command exited {process.returncode}: {argv!r}\n{stderr[-3000:]}")
    return stdout.strip()


def validate_layout(raw, name, size, fields):
    alias = re.search(r"^\[\d+\] TYPEDEF '" + re.escape(name) + r"' type_id=(\d+)$", raw, re.M)
    if not alias:
        raise RuntimeError(f"loaded BTF missing {name}")
    layout = re.search(r"^\[" + alias[1] + r"\] STRUCT '[^']*' size=" + str(size) +
                       r" vlen=" + str(len(fields)) + r"\n((?:\t[^\n]+\n)+)", raw, re.M)
    observed = re.findall(r"\t'([^']+)' type_id=\d+ bits_offset=(\d+)", layout[1]) if layout else []
    if observed != [(key, str(offset)) for key, offset in fields]:
        raise RuntimeError(f"loaded BTF layout differs: {name}")


def validate_diagnostic_interface(raw):
    fields = [("raw_action", 0), ("requested_first", 64), ("requested_outer", 128),
              ("max_first", 192), ("max_outer", 256), ("output_first", 320),
              ("output_outer", 384), ("phase", 448), ("request_attempted", 480),
              ("request_conflict", 512), ("initial_region_result", 544),
              ("initial_effect", 576), ("native_iterations", 608),
              ("native_completed", 640)]
    match = re.search(
        r"^\[(\d+)\] STRUCT 'uvm_bpf_prefetch_diagnostic_ctx' size=88 vlen=14\n"
        r"((?:\t[^\n]+\n){14})", raw, re.M)
    members = re.findall(r"\t'([^']+)' type_id=\d+ bits_offset=(\d+)", match[2]) if match else []
    if members != [
            (key, str(offset)) for key, offset in fields]:
        raise RuntimeError("loaded BTF lacks the exact address-free diagnostic context")
    struct_id = match[1]
    const_ids = re.findall(r"^\[(\d+)\] CONST '\(anon\)' type_id=" + struct_id + r"$", raw, re.M)
    pointer_ids = []
    for const_id in const_ids:
        pointer_ids.extend(re.findall(r"^\[(\d+)\] PTR '\(anon\)' type_id=" + const_id + r"$", raw, re.M))
    prototype_ids = []
    for pointer_id in pointer_ids:
        prototype_ids.extend(re.findall(
            r"^\[(\d+)\] FUNC_PROTO '\(anon\)' ret_type_id=0 vlen=1\n"
            r"\t'ctx' type_id=" + pointer_id + r"$", raw, re.M))
    if not any(re.search(r"^\[\d+\] FUNC 'uvm_bpf_prefetch_diagnostic' type_id=" +
                         prototype_id + r" linkage=", raw, re.M)
               for prototype_id in prototype_ids):
        raise RuntimeError("loaded diagnostic is not void(const context *)")
    enum = re.search(
        r"ENUM 'uvm_bpf_prefetch_diagnostic_phase'[^\n]*vlen=2\n"
        r"\t'UVM_BPF_PREFETCH_DIAG_SELECTED' val=1\n"
        r"\t'UVM_BPF_PREFETCH_DIAG_FINISHED' val=2", raw)
    if not enum:
        raise RuntimeError("loaded diagnostic phase enum differs")


def validate_metrics(mode, row):
    """Independently recheck the saved counters, not just the loader's flag."""
    def count(key):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid counter {key}")
        return value
    if (row.get("valid") is not True or row.get("empty_frames") is not True or
        type(row.get("action")) is not int or row["action"] != MODES[mode]):
        raise RuntimeError("loader did not produce a closed valid frame set")
    decisions = count("wrapper_enter")
    if not decisions or decisions != count("wrapper_exit"):
        raise RuntimeError("unmatched/non-engaged entries and exits")
    if (decisions != count("selected_events") or decisions != count("finished_events") or
        decisions != count("decisions_complete") or count("diagnostic_calls") != 2 * decisions):
        raise RuntimeError("diagnostic phase accounting does not reconcile")
    if (decisions != count("region_noop_default") + count("region_apply") or
        decisions != count("native_effects") + count("bypass_effects") or
        decisions != count("empty_outputs") + count("nonempty_outputs")):
        raise RuntimeError("diagnostic result/effect/output accounting does not reconcile")
    returned = {"native": "returned_default", "bypass": "returned_bypass", "invalid99": "returned_invalid99"}
    if count(returned[mode]) != decisions or sum(count(key) for key in returned.values()) != decisions:
        raise RuntimeError("actual wrapper-return values disagree with the selected control")
    expected_policy_calls = 0 if mode == "native" else decisions
    if count("policy_calls") != expected_policy_calls or count("setter_ok") != expected_policy_calls:
        raise RuntimeError("actual policy/setter calls differ from matched wrappers")
    if mode == "bypass":
        if (count("region_apply") != decisions or count("region_noop_default") or
            count("bypass_effects") != decisions or count("native_effects") or
            count("native_completions") or count("native_iterations") or
            count("empty_outputs") != decisions):
            raise RuntimeError("legal empty BYPASS did not remain empty and traversal-free")
    elif mode == "native":
        if (count("region_noop_default") != decisions or count("region_apply") or
            count("native_effects") != decisions or count("bypass_effects") or
            count("native_completions") != decisions or count("native_iterations") < decisions):
            raise RuntimeError("native control did not execute native traversal")
    elif (count("region_apply") != decisions or count("region_noop_default") or
          count("native_effects") != decisions or count("bypass_effects") or
          count("native_completions") != decisions or count("native_iterations") < decisions):
        raise RuntimeError("invalid action did not fall back to native traversal")
    for key in ("map_errors", "nesting_errors", "missing_frame", "order_errors",
                "read_errors", "request_errors", "action_errors", "state_errors",
                "phase_errors", "traversal_errors", "output_errors"):
        if count(key):
            raise RuntimeError(f"observer failure: {key}")
    expected = {"wrapper_enter", "wrapper_exit", "diagnostic_enter", "gpu_page_prefetch"}
    programs = row.get("programs", [])
    if (len(programs) != 4 or {p.get("name") for p in programs} != expected or
        len({p.get("id") for p in programs}) != 4):
        raise RuntimeError("missing actual program statistics")
    for p in programs:
        if type(p.get("id")) is not int or p["id"] <= 0 or type(p.get("recursion_misses")) is not int or p["recursion_misses"] != 0:
            raise RuntimeError("invalid program identity or missed recursion events")
        runs = p.get("run_count")
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < 0:
            raise RuntimeError("missing actual program run count")
        name = p["name"]
        if name in {"wrapper_enter", "wrapper_exit"} and runs != count(name):
            raise RuntimeError(f"BPF run count disagrees with observed {name}")
        if name == "gpu_page_prefetch" and runs != expected_policy_calls:
            raise RuntimeError("actual struct_ops runs disagree with callback counter")
        if name == "diagnostic_enter" and runs != count("diagnostic_calls"):
            raise RuntimeError("actual diagnostic runs disagree with both phases")


def read_compute_samples(path, *, final=False):
    text = path.read_text(errors="replace")
    if not final and text and not text.endswith("\n"):
        newline = text.rfind("\n")
        text = text[:newline + 1] if newline >= 0 else ""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if row.get("event") == "sample"], rows


def wait_compute_sample(path, process, predicate, *, after_ns=0, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raise_if_interrupted()
        if process.poll() is not None:
            raise RuntimeError(f"compute-process monitor exited early: {process.returncode}")
        samples, _ = read_compute_samples(path)
        for row in reversed(samples):
            started = row.get("query_started_mono_ns")
            finished = row.get("query_finished_mono_ns")
            if (type(started) is int and type(finished) is int and
                    0 < started <= finished and started > after_ns and
                    "error" not in row and predicate(row.get("pids"))):
                return row
        time.sleep(0.05)
    raise RuntimeError("compute-process monitor did not cover the required lifecycle point")


def wait_telemetry_sample(path, process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raise_if_interrupted()
        if process.poll() is not None:
            raise RuntimeError(f"GPU telemetry exited early: {process.returncode}")
        if len([line for line in path.read_text(errors="replace").splitlines() if line.strip()]) >= 2:
            return
        time.sleep(0.05)
    raise RuntimeError("GPU telemetry did not produce a pre-release sample")


def validate_sample_cadence(samples, max_gap):
    query_durations = [row["query_finished_mono_ns"] - row["query_started_mono_ns"]
                       for row in samples]
    idle_gaps = [later["query_started_mono_ns"] - earlier["query_finished_mono_ns"]
                 for earlier, later in zip(samples, samples[1:])]
    start_gaps = [later["query_started_mono_ns"] - earlier["query_started_mono_ns"]
                  for earlier, later in zip(samples, samples[1:])]
    finish_gaps = [later["query_finished_mono_ns"] - earlier["query_finished_mono_ns"]
                   for earlier, later in zip(samples, samples[1:])]
    if (any(duration > max_gap for duration in query_durations) or
            any(gap > max_gap for gap in idle_gaps + start_gaps + finish_gaps)):
        raise RuntimeError("compute-process monitor has an uncovered sampling gap")
    return {
        "max_query_duration_ns": max(query_durations),
        "max_idle_gap_ns": max(idle_gaps) if idle_gaps else 0,
        "max_start_gap_ns": max(start_gaps) if start_gaps else 0,
        "max_finish_gap_ns": max(finish_gaps) if finish_gaps else 0,
    }


def validate_compute_monitor(path, target_pid, window):
    samples, rows = read_compute_samples(path, final=True)
    final = [row for row in rows if row.get("event") == "final"]
    if not samples or len(final) != 1 or final[0].get("errors") != 0:
        raise RuntimeError("continuous compute-process monitor was incomplete")
    starts = [row.get("query_started_mono_ns") for row in samples]
    finishes = [row.get("query_finished_mono_ns") for row in samples]
    if (any(type(value) is not int or value <= 0 for value in starts + finishes) or
            any(started > finished for started, finished in zip(starts, finishes)) or
            starts != sorted(starts) or len(set(starts)) != len(starts) or
            any(later <= earlier for earlier, later in zip(finishes, starts[1:]))):
        raise RuntimeError("invalid compute-process monitor timestamps")
    foreign = set()
    for row in samples:
        pids = row.get("pids")
        if ("error" in row or not isinstance(pids, list) or
                any(type(pid) is not int or pid <= 0 for pid in pids)):
            raise RuntimeError("invalid compute-process monitor sample")
        foreign.update(pid for pid in pids if pid != target_pid)
    if foreign:
        raise RuntimeError(f"foreign compute clients appeared: {sorted(foreign)}")

    required = ("pretarget_empty_query_started_mono_ns",
                "pretarget_empty_query_finished_mono_ns", "target_started_mono_ns",
                "pause_observed_mono_ns", "paused_target_query_started_mono_ns",
                "paused_target_query_finished_mono_ns", "ready_observed_mono_ns",
                "ready_target_query_started_mono_ns", "ready_target_query_finished_mono_ns",
                "released_mono_ns", "post_release_target_query_started_mono_ns",
                "post_release_target_query_finished_mono_ns", "target_exit_mono_ns",
                "loader_stopped_mono_ns", "post_exit_empty_query_started_mono_ns",
                "post_exit_empty_query_finished_mono_ns")
    if any(type(window.get(key)) is not int or window[key] <= 0 for key in required):
        raise RuntimeError("compute-process lifecycle markers are incomplete")
    marker_times = [window[key] for key in required]
    if marker_times != sorted(marker_times):
        raise RuntimeError("compute-process lifecycle markers are out of order")
    by_start = {row["query_started_mono_ns"]: row for row in samples}
    exact = {
        "pretarget_empty_query_started_mono_ns": (
            "pretarget_empty_query_finished_mono_ns", []),
        "paused_target_query_started_mono_ns": (
            "paused_target_query_finished_mono_ns", [target_pid]),
        "ready_target_query_started_mono_ns": (
            "ready_target_query_finished_mono_ns", [target_pid]),
        "post_release_target_query_started_mono_ns": (
            "post_release_target_query_finished_mono_ns", [target_pid]),
        "post_exit_empty_query_started_mono_ns": (
            "post_exit_empty_query_finished_mono_ns", []),
    }
    for started_key, (finished_key, expected_pids) in exact.items():
        row = by_start.get(window[started_key])
        if (row is None or row["query_finished_mono_ns"] != window[finished_key] or
                row["pids"] != expected_pids):
            raise RuntimeError("required compute-process lifecycle sample is absent")
    max_gap = window.get("max_sample_gap_ns")
    if type(max_gap) is not int or max_gap <= 0 or max_gap != COMPUTE_MAX_GAP_NS:
        raise RuntimeError("compute-process sampling bound changed")
    if (window["target_started_mono_ns"] - window["pretarget_empty_query_finished_mono_ns"] > max_gap or
            window["paused_target_query_started_mono_ns"] - window["pause_observed_mono_ns"] > max_gap or
            window["ready_target_query_started_mono_ns"] - window["ready_observed_mono_ns"] > max_gap or
            window["released_mono_ns"] - window["ready_target_query_finished_mono_ns"] > max_gap or
            window["post_release_target_query_started_mono_ns"] - window["released_mono_ns"] > max_gap or
            window["target_exit_mono_ns"] - window["post_release_target_query_finished_mono_ns"] < 0 or
            window["post_exit_empty_query_started_mono_ns"] - window["target_exit_mono_ns"] > max_gap or
            window["post_exit_empty_query_started_mono_ns"] - window["loader_stopped_mono_ns"] > max_gap or
            window["target_exit_mono_ns"] - max(
                row["query_finished_mono_ns"] for row in samples
                if row["query_finished_mono_ns"] <= window["target_exit_mono_ns"] and
                target_pid in row["pids"]
            ) > max_gap):
        raise RuntimeError("compute-process lifecycle point exceeded the sampling bound")
    covered = [row for row in samples
               if window["pretarget_empty_query_started_mono_ns"] <=
               row["query_started_mono_ns"] <= window["post_exit_empty_query_started_mono_ns"]]
    if (not covered or
            covered[0]["query_started_mono_ns"] != window["pretarget_empty_query_started_mono_ns"] or
            covered[-1]["query_started_mono_ns"] != window["post_exit_empty_query_started_mono_ns"]):
        raise RuntimeError("compute-process monitor has an uncovered sampling gap")
    cadence = validate_sample_cadence(covered, max_gap)
    if final[0].get("monotonic_ns", 0) < window["post_exit_empty_query_finished_mono_ns"]:
        raise RuntimeError("compute-process monitor stopped before the tail sample")
    return {"samples": len(samples), "target_samples": sum(target_pid in row["pids"] for row in samples),
            "foreign_pids": sorted(foreign), "errors": final[0]["errors"], **cadence}


def run_cell(mode, directory):
    raise_if_interrupted()
    directory.mkdir(parents=True, exist_ok=False)
    record = {"mode": mode, "complete": False, "started_ns": time.time_ns(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "worker_cpu_affinity": list(range(8, 16)), "telemetry_cpu": 16,
              "loader_cpu": 17}
    streams, monitors = [], []
    target = loader = compute_monitor = None
    compute_path = directory / "compute-apps.jsonl"
    before = None
    old_checked = safety.run_checked
    safety.run_checked = checked
    try:
        if not set(range(8, 18)) <= set(os.sched_getaffinity(0)):
            raise RuntimeError("coordinator must retain CPUs 8-17; target alone is pinned to 8-15")
        if any(os.environ.get(k) for k in ("LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH")):
            raise RuntimeError("ambient injection is not admitted")
        if os.uname().release != "6.15.11-061511-generic" or os.sysconf("SC_PAGE_SIZE") != 4096:
            raise RuntimeError("requires the frozen x86 6.15/4 KiB-page runtime")
        record["interface"] = safety.verify_loaded_uvm_interface("575.57.08")
        raw_btf = checked(["sudo", "-n", "bpftool", "btf", "dump", "file", str(safety.LOADED_UVM_BTF), "format", "raw"])
        (directory / "loaded-uvm-btf.txt").write_text(raw_btf + "\n")
        for hook in HOOKS:
            if f"FUNC '{hook}'" not in raw_btf:
                raise RuntimeError(f"loaded BTF lacks named hook {hook}")
        validate_layout(raw_btf, "nv_gpu_prefetch_decision_t", 24,
                        [("attempted", 0), ("conflict", 8), ("first", 64), ("outer", 128)])
        validate_layout(raw_btf, "uvm_va_block_region_t", 4, [("first", 0), ("outer", 16)])
        validate_diagnostic_interface(raw_btf)
        if Path("/sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable").read_text().strip() != "1":
            raise RuntimeError("native prefetch must already be enabled; no parameter/module change is allowed")
        record["files"] = runtime_files()
        before = safety.safety_snapshot()
        record["safety_before"] = before
        safety.validate_pre_server_safety(before)
        if before["gpu"]["driver"] != "575.57.08":
            raise RuntimeError("actual GPU driver differs from the admitted module version")
        # Defer Python handlers until the helper's child is registered. Do not
        # mask OS signals: the child would inherit that mask across exec.
        queued = []
        launch_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for sig in launch_handlers:
            signal.signal(sig, lambda signum, frame: queued.append(signum))
        try:
            telemetry, stream, telemetry_path = safety.start_gpu_telemetry(directory)
            monitors.append(telemetry)
            streams.append(stream)
        finally:
            for sig, handler in launch_handlers.items():
                signal.signal(sig, handler)
        if queued:
            launch_handlers[queued[0]](queued[0], None)
        kernel_log = (directory / "kernel-follow.log").open("x")
        streams.append(kernel_log)
        monitors.append(subprocess.Popen(["taskset", "-c", "16", "journalctl", "-k", "-b", "-f", "-n", "0", "--no-pager"],
                                         stdout=kernel_log, stderr=subprocess.STDOUT, start_new_session=True))
        compute_log = compute_path.open("x")
        streams.append(compute_log)
        compute_monitor = subprocess.Popen(
            ["taskset", "-c", "16", sys.executable, "-B", str(COMPUTE_MONITOR)],
            stdout=compute_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        monitors.append(compute_monitor)
        pretarget = wait_compute_sample(compute_path, compute_monitor, lambda pids: pids == [])
        wait_telemetry_sample(telemetry_path, telemetry)
        record["compute_window"] = {
            "max_sample_gap_ns": COMPUTE_MAX_GAP_NS,
            "pretarget_empty_query_started_mono_ns": pretarget["query_started_mono_ns"],
            "pretarget_empty_query_finished_mono_ns": pretarget["query_finished_mono_ns"],
        }
        target_log = (directory / "target.log").open("x")
        loader_log = (directory / "loader.jsonl").open("x")
        streams.extend((target_log, loader_log))
        argv = ["taskset", "-c", "8-15", str(control.WORKLOAD), "--gib", "8",
                "--region-kib", "64", "--wait-for-monitor", "--output", str(directory / "target.json")]
        record["target_argv"] = argv
        record["compute_window"]["target_started_mono_ns"] = time.monotonic_ns()
        target = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=target_log,
                                  stderr=subprocess.STDOUT, text=True, start_new_session=True)
        record["target_pid"] = target.pid
        deadline = time.monotonic() + 60
        while f"MONITOR_PID: {target.pid}\n" not in (directory / "target.log").read_text():
            raise_if_interrupted()
            if target.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("target failed to reach its post-CPU-initialization pause")
            time.sleep(0.1)
        record["compute_window"]["pause_observed_mono_ns"] = time.monotonic_ns()
        paused = wait_compute_sample(
            compute_path, compute_monitor, lambda pids: pids == [target.pid],
            after_ns=record["compute_window"]["pause_observed_mono_ns"])
        record["compute_window"]["paused_target_query_started_mono_ns"] = paused["query_started_mono_ns"]
        record["compute_window"]["paused_target_query_finished_mono_ns"] = paused["query_finished_mono_ns"]
        argv = ["sudo", "-n", "taskset", "-c", "17", str(LOADER), mode]
        record["loader_argv"] = argv
        loader = subprocess.Popen(argv, stdout=loader_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        record["loader_group_pid"] = loader.pid
        ready = safety.wait_event(loader, directory / "loader.jsonl", "ready", 30)
        raise_if_interrupted()
        record["ready"] = ready
        if (ready.get("mode") != mode or ready.get("action") != MODES[mode] or
            ready.get("pid") not in safety.descendants(loader.pid) or os.getpgid(ready["pid"]) != loader.pid):
            raise RuntimeError("loader identity or fixed configuration mismatch")
        safety.validate_policy_ownership(ready, safety.struct_ops_inventory())
        links = json.loads(checked(["sudo", "-n", "bpftool", "link", "show", "-j"]))
        ids = ready.get("observer_link_ids", [])
        observed = [link for link in links if link.get("id") in ids]
        if (len(ids) != OBSERVER_COUNT or len(set(ids)) != OBSERVER_COUNT or
            len(observed) != OBSERVER_COUNT or any(link.get("type") != "tracing" for link in observed)):
            raise RuntimeError("not all three named observers attached")
        if (mode == "native") != (ready.get("struct_link_id") == 0):
            raise RuntimeError("native control unexpectedly attached a policy, or BPF control did not")
        record["observer_links"] = observed
        ready_gate_ns = time.monotonic_ns()
        record["compute_window"]["ready_observed_mono_ns"] = ready_gate_ns
        ready_sample = wait_compute_sample(
            compute_path, compute_monitor, lambda pids: pids == [target.pid],
            after_ns=ready_gate_ns)
        record["compute_window"]["ready_target_query_started_mono_ns"] = ready_sample["query_started_mono_ns"]
        record["compute_window"]["ready_target_query_finished_mono_ns"] = ready_sample["query_finished_mono_ns"]
        record["release_monitor_gate"] = {
            "alive": [process.poll() is None for process in monitors],
            "compute_query_started_mono_ns": ready_sample["query_started_mono_ns"],
            "compute_query_finished_mono_ns": ready_sample["query_finished_mono_ns"],
            "telemetry_has_sample": len([
                line for line in telemetry_path.read_text(errors="replace").splitlines() if line.strip()
            ]) >= 2,
        }
        if (loader.poll() is not None or record["release_monitor_gate"]["alive"] != [True, True, True] or
                not record["release_monitor_gate"]["telemetry_has_sample"]):
            raise RuntimeError("loader or a continuous monitor was not ready at target release")
        raise_if_interrupted()
        record["release_write_started_ns"] = time.time_ns()
        target.stdin.write("\n")
        target.stdin.flush()
        target.stdin.close()
        record["compute_window"]["released_mono_ns"] = time.monotonic_ns()
        record["released_ns"] = time.time_ns()
        post_release = wait_compute_sample(
            compute_path, compute_monitor, lambda pids: pids == [target.pid],
            after_ns=record["compute_window"]["released_mono_ns"])
        record["compute_window"]["post_release_target_query_started_mono_ns"] = (
            post_release["query_started_mono_ns"])
        record["compute_window"]["post_release_target_query_finished_mono_ns"] = (
            post_release["query_finished_mono_ns"])
        target.wait(timeout=60)
        raise_if_interrupted()
        record["compute_window"]["target_exit_mono_ns"] = time.monotonic_ns()
        stop_owned(target)
        if target.returncode != 0 or loader.poll() is not None:
            raise RuntimeError("target failed, or loader exited during execution")
        record["target"] = control.validate_workload_result(json.loads((directory / "target.json").read_text()))
        stop_owned(loader)
        record["compute_window"]["loader_stopped_mono_ns"] = time.monotonic_ns()
        if loader.returncode != 0:
            raise RuntimeError(f"fixture failed: exit {loader.returncode}")
        metrics = control.latest_event((directory / "loader.jsonl").read_text(), "final_metrics")
        record["metrics"] = metrics
        validate_metrics(mode, metrics)
        post_exit = wait_compute_sample(
            compute_path, compute_monitor, lambda pids: pids == [],
            after_ns=record["compute_window"]["loader_stopped_mono_ns"])
        record["compute_window"]["post_exit_empty_query_started_mono_ns"] = (
            post_exit["query_started_mono_ns"])
        record["compute_window"]["post_exit_empty_query_finished_mono_ns"] = (
            post_exit["query_finished_mono_ns"])
        raise_if_interrupted()
        record["complete"] = True
    except BaseException as error:
        record["complete"] = False
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_errors = []

        def cleanup_error(stage, error):
            cleanup_errors.append({"stage": stage, "error": f"{type(error).__name__}: {error}"})

        target_stopped = False
        loader_stopped = False
        try:
            stop_owned(target)
            target_stopped = True
        except BaseException as error:
            cleanup_error("target_group", error)
            record["cleanup_failure"] = (
                f"target survived; retain loader {getattr(loader, 'pid', None)}: {error}")
        if target_stopped:
            try:
                stop_owned(loader)
                loader_stopped = True
            except BaseException as error:
                cleanup_error("loader_group", error)
                record["cleanup_failure"] = f"loader group {getattr(loader, 'pid', None)} survived: {error}"
        else:
            record["link_cleanup"] = {
                "status": "not_attempted", "reason": "target survived; loader intentionally retained"}

        # Once attachment IDs are known and the loader is gone, this check is
        # independent of monitor, stream, and safety-snapshot cleanup failures.
        if record.get("ready") and loader_stopped:
            try:
                remaining = json.loads(checked(["sudo", "-n", "bpftool", "link", "show", "-j"]))
                owned_ids = set(record["ready"]["observer_link_ids"])
                if record["ready"].get("struct_link_id"):
                    owned_ids.add(record["ready"]["struct_link_id"])
                survivors = sorted(link.get("id") for link in remaining if link.get("id") in owned_ids)
                if survivors:
                    raise RuntimeError(f"owned observer or policy links remain: {survivors}")
                record["link_cleanup"] = {"status": "passed", "owned_link_ids": sorted(owned_ids)}
            except BaseException as error:
                cleanup_error("owned_links", error)
                record["link_cleanup"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        elif record.get("ready") and target_stopped:
            record["link_cleanup"] = {"status": "not_attempted", "reason": "loader group survived"}

        record["monitors_alive"] = [process.poll() is None for process in monitors]
        try:
            stop_monitors(monitors, record)
        except BaseException as error:
            cleanup_error("monitors", error)
        for stream in streams:
            try:
                stream.close()
            except BaseException as error:
                cleanup_error("stream_close", error)

        if before is not None:
            try:
                record["safety_after"] = safety.wait_for_post_server_safety(before, timeout=60)
                if record["safety_after"]["gpu"]["driver"] != "575.57.08":
                    raise RuntimeError("driver changed during the cell")
            except BaseException as error:
                cleanup_error("post_safety", error)
        if record.get("files"):
            try:
                record["files_after"] = runtime_files()
                if record["files_after"] != record["files"]:
                    raise RuntimeError("source/binary paths, sizes, or mtimes changed during the cell")
            except BaseException as error:
                cleanup_error("runtime_files", error)
        if monitors:
            try:
                if record["monitors_alive"] != [True, True, True]:
                    raise RuntimeError("continuous safety monitor exited early")
                record["telemetry"] = safety.validate_gpu_telemetry(
                    telemetry_path, allow_fixed_power_cap=True)
            except BaseException as error:
                cleanup_error("telemetry", error)
            try:
                record["kernel_abnormal"] = safety.filtered_kernel_records(
                    (directory / "kernel-follow.log").read_text())
                if record["kernel_abnormal"]:
                    raise RuntimeError("new kernel abnormality")
            except BaseException as error:
                cleanup_error("kernel_log", error)
            try:
                if compute_monitor is None or compute_monitor.returncode != 0:
                    raise RuntimeError(
                        f"compute-process monitor exited {getattr(compute_monitor, 'returncode', None)}")
                record["compute_monitor"] = validate_compute_monitor(
                    compute_path, record["target_pid"], record["compute_window"])
            except BaseException as error:
                cleanup_error("compute_monitor", error)
        try:
            if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != record["boot_id"]:
                raise RuntimeError("boot changed during the cell")
        except BaseException as error:
            cleanup_error("boot", error)

        interrupted = list(INTERRUPTED_SIGNALS)
        if interrupted:
            record["complete"] = False
            record["interrupt_signals"] = interrupted
            record.setdefault("error", f"InterruptedError: signal {interrupted[0]}")
        if cleanup_errors:
            record["complete"] = False
            record["cleanup_errors"] = cleanup_errors
            record["safety_error"] = "; ".join(
                f"{item['stage']}: {item['error']}" for item in cleanup_errors)
        record["finished_ns"] = time.time_ns()
        try:
            safety.atomic_write_json(directory / "execution.json", record)
        finally:
            safety.run_checked = old_checked
        if cleanup_errors:
            raise RuntimeError(f"Q2 cleanup/safety gates failed: {record['safety_error']}")
        if interrupted:
            raise InterruptedError(f"signal {interrupted[0]}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists():
        raise RuntimeError("refusing to reuse an output directory")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    INTERRUPTED_SIGNALS.clear()
    for sig in previous:
        signal.signal(sig, note_interrupt)
    lease = None
    try:
        lease = owned.Leases()
        raise_if_interrupted()
        output.mkdir(parents=True, exist_ok=False)
        for mode in MODES:
            raise_if_interrupted()
            run_cell(mode, output / mode)
        raise_if_interrupted()
        safety.atomic_write_json(output / "summary.json", {"complete": True, "modes": list(MODES)})
    finally:
        if lease is not None:
            lease.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
