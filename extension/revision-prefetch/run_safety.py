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
LOADER = HERE / "build/prefetch_safety"
HOOKS = ("compute_prefetch_mask", "uvm_bpf_call_gpu_page_prefetch",
         "uvm_perf_prefetch_bitmap_tree_iter_get_range", "uvm_bpf_call_gpu_page_prefetch_iter")


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
                 HERE / "fixture.h", HERE / "loader.c", Path(__file__),
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
    masks, decisions = count("mask_enter"), count("wrapper_enter")
    if not masks or not decisions or masks != count("mask_exit") or decisions != count("wrapper_exit"):
        raise RuntimeError("unmatched/non-engaged entries and exits")
    if decisions != count("decisions_complete") or masks != count("empty_masks") + count("nonempty_masks"):
        raise RuntimeError("decision/mask accounting does not reconcile")
    returned = {"native": "returned_default", "bypass": "returned_bypass", "invalid99": "returned_invalid99"}
    if count(returned[mode]) != decisions or sum(count(key) for key in returned.values()) != decisions:
        raise RuntimeError("actual wrapper-return values disagree with the selected control")
    expected_policy_calls = 0 if mode == "native" else decisions
    if count("policy_calls") != expected_policy_calls or count("setter_ok") != expected_policy_calls:
        raise RuntimeError("actual policy/setter calls differ from matched wrappers")
    if mode == "bypass":
        if count("bypass_decisions") != decisions or count("native_decisions") or count("range_calls") or count("nonempty_masks"):
            raise RuntimeError("legal empty BYPASS traversed or returned a nonempty mask")
    elif count("native_decisions") != decisions or count("bypass_decisions") or count("range_calls") < decisions:
        raise RuntimeError("native traversal was not observed after every returned decision")
    for key in ("map_errors", "nesting_errors", "missing_frame", "identity_errors", "order_errors",
                "read_errors", "request_errors", "action_errors", "traversal_errors",
                "iterator_calls", "mask_bounds_errors"):
        if count(key):
            raise RuntimeError(f"observer failure: {key}")
    expected = {"mask_enter", "mask_exit", "wrapper_enter", "wrapper_exit",
                "range_enter", "iterator_enter", "gpu_page_prefetch"}
    programs = row.get("programs", [])
    if (len(programs) != 7 or {p.get("name") for p in programs} != expected or
        len({p.get("id") for p in programs}) != 7):
        raise RuntimeError("missing actual program statistics")
    for p in programs:
        if type(p.get("id")) is not int or p["id"] <= 0 or type(p.get("recursion_misses")) is not int or p["recursion_misses"] != 0:
            raise RuntimeError("invalid program identity or missed recursion events")
        runs = p.get("run_count")
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < 0:
            raise RuntimeError("missing actual program run count")
        name = p["name"]
        if name in {"mask_enter", "mask_exit", "wrapper_enter", "wrapper_exit"} and runs != count(name):
            raise RuntimeError(f"BPF run count disagrees with observed {name}")
        if name == "gpu_page_prefetch" and runs != expected_policy_calls:
            raise RuntimeError("actual struct_ops runs disagree with callback counter")
        if name == "iterator_enter" and runs != 0:
            raise RuntimeError("unexpected iterator-wrapper execution")
        if name == "range_enter" and runs < count("range_calls"):
            raise RuntimeError("actual range observer ran less often than its counter")
    samples = row.get("mask_samples", [])
    if not samples or len({sample.get("cpu") for sample in samples}) != len(samples):
        raise RuntimeError("missing/duplicate final-mask samples")
    for sample in samples:
        first, outer, words = sample.get("first"), sample.get("outer"), sample.get("bitmap")
        if (type(sample.get("cpu")) is not int or sample["cpu"] < 0 or type(first) is not int or
            type(outer) is not int or not 0 <= first < outer <= 512 or not isinstance(words, list) or len(words) != 8):
            raise RuntimeError("invalid actual-mask sample")
        if any(isinstance(word, bool) or not isinstance(word, int) or not 0 <= word < 2**64 for word in words):
            raise RuntimeError("invalid actual-mask word")
        bitmap = sum(word << (64 * index) for index, word in enumerate(words))
        allowed = ((1 << outer) - 1) ^ ((1 << first) - 1)
        if bitmap & ~allowed or (mode == "bypass" and bitmap):
            raise RuntimeError("actual-mask sample violates bounds/control")


def run_cell(mode, directory):
    directory.mkdir(parents=True, exist_ok=False)
    record = {"mode": mode, "complete": False, "started_ns": time.time_ns(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "worker_cpu_affinity": list(range(8, 16)), "telemetry_cpu": 16}
    streams, monitors = [], []
    target = loader = None
    before = None
    old_checked = safety.run_checked
    safety.run_checked = checked
    try:
        if not set(range(8, 17)) <= set(os.sched_getaffinity(0)):
            raise RuntimeError("coordinator must retain CPUs 8-16; target alone is pinned to 8-15")
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
        validate_layout(raw_btf, "uvm_page_mask_t", 64, [("bitmap", 0)])
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
        target_log = (directory / "target.log").open("x")
        loader_log = (directory / "loader.jsonl").open("x")
        streams.extend((target_log, loader_log))
        argv = ["taskset", "-c", "8-15", str(control.WORKLOAD), "--gib", "8",
                "--region-kib", "64", "--wait-for-monitor", "--output", str(directory / "target.json")]
        record["target_argv"] = argv
        target = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=target_log,
                                  stderr=subprocess.STDOUT, text=True, start_new_session=True)
        record["target_pid"] = target.pid
        deadline = time.monotonic() + 60
        while f"MONITOR_PID: {target.pid}\n" not in (directory / "target.log").read_text():
            if target.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("target failed to reach its post-CPU-initialization pause")
            time.sleep(0.1)
        argv = ["sudo", "-n", "taskset", "-c", "17", str(LOADER), mode]
        record["loader_argv"] = argv
        loader = subprocess.Popen(argv, stdout=loader_log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        record["loader_group_pid"] = loader.pid
        ready = safety.wait_event(loader, directory / "loader.jsonl", "ready", 30)
        record["ready"] = ready
        if (ready.get("mode") != mode or ready.get("action") != MODES[mode] or
            ready.get("pid") not in safety.descendants(loader.pid) or os.getpgid(ready["pid"]) != loader.pid):
            raise RuntimeError("loader identity or fixed configuration mismatch")
        safety.validate_policy_ownership(ready, safety.struct_ops_inventory())
        links = json.loads(checked(["sudo", "-n", "bpftool", "link", "show", "-j"]))
        ids = ready.get("observer_link_ids", [])
        observed = [link for link in links if link.get("id") in ids]
        if len(ids) != 6 or len(set(ids)) != 6 or len(observed) != 6 or any(link.get("type") != "tracing" for link in observed):
            raise RuntimeError("not all six named observers attached")
        if (mode == "native") != (ready.get("struct_link_id") == 0):
            raise RuntimeError("native control unexpectedly attached a policy, or BPF control did not")
        record["observer_links"] = observed
        if loader.poll() is not None:
            raise RuntimeError("loader exited before release")
        target.stdin.write("\n")
        target.stdin.flush()
        target.stdin.close()
        record["released_ns"] = time.time_ns()
        target.wait(timeout=60)
        stop_owned(target)
        if target.returncode != 0 or loader.poll() is not None:
            raise RuntimeError("target failed, or loader exited during execution")
        record["target"] = control.validate_workload_result(json.loads((directory / "target.json").read_text()))
        stop_owned(loader)
        if loader.returncode != 0:
            raise RuntimeError(f"fixture failed: exit {loader.returncode}")
        metrics = control.latest_event((directory / "loader.jsonl").read_text(), "final_metrics")
        record["metrics"] = metrics
        validate_metrics(mode, metrics)
        record["complete"] = True
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            try:
                stop_owned(target)
            except BaseException as error:
                record["cleanup_failure"] = f"target survived; retain loader {getattr(loader, 'pid', None)}: {error}"
                record["complete"] = False
                raise
            try:
                stop_owned(loader)
            except BaseException as error:
                record["complete"] = False
                record["cleanup_failure"] = f"loader group {getattr(loader, 'pid', None)} survived: {error}"
                raise
        finally:
            record["monitors_alive"] = [p.poll() is None for p in monitors]
            try:
                try:
                    stop_monitors(monitors, record)
                finally:
                    for stream in streams:
                        stream.close()
                if before is not None:
                    record["safety_after"] = safety.wait_for_post_server_safety(before, timeout=60)
                    if record["safety_after"]["gpu"]["driver"] != "575.57.08":
                        raise RuntimeError("driver changed during the cell")
                if record.get("complete"):
                    record["files_after"] = runtime_files()
                    if record["files_after"] != record["files"]:
                        raise RuntimeError("source/binary paths, sizes, or mtimes changed during the cell")
                    remaining = json.loads(checked(["sudo", "-n", "bpftool", "link", "show", "-j"]))
                    owned_ids = set(record["ready"]["observer_link_ids"])
                    if any(link.get("id") in owned_ids for link in remaining):
                        raise RuntimeError("owned observer links remain after loader cleanup")
                if monitors:
                    if record["monitors_alive"] != [True, True]:
                        raise RuntimeError("continuous safety monitor exited early")
                    record["telemetry"] = safety.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True)
                    record["kernel_abnormal"] = safety.filtered_kernel_records((directory / "kernel-follow.log").read_text())
                    if record["kernel_abnormal"]:
                        raise RuntimeError("new kernel abnormality")
                if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != record["boot_id"]:
                    raise RuntimeError("boot changed during the cell")
            except BaseException as error:
                record["complete"] = False
                record["safety_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                record["finished_ns"] = time.time_ns()
                safety.atomic_write_json(directory / "execution.json", record)
                safety.run_checked = old_checked
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists():
        raise RuntimeError("refusing to reuse an output directory")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, _frame):
        # The first interrupt aborts; repeated interrupts must not break cleanup.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(f"signal {signum}")
    for sig in previous:
        signal.signal(sig, interrupted)
    lease = None
    try:
        lease = owned.Leases()
        output.mkdir(parents=True, exist_ok=False)
        for mode in MODES:
            run_cell(mode, output / mode)
        safety.atomic_write_json(output / "summary.json", {"complete": True, "modes": list(MODES)})
    finally:
        if lease is not None:
            lease.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
