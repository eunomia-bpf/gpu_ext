#!/usr/bin/env python3
"""CPU-only tests for the independent revision-RQ4 analyzer."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import analyze_revision_rq4 as audit
import run_revision_rq4 as runner
from test_offline import (
    lossless_exit_log,
    lossless_launchlate_log,
    lossless_nvbit_exit_log,
    lossless_nvbit_launchlate_log,
)


def safe_cell_record():
    def snapshot():
        return {
            "power_limit_service": "active",
            "power_limit_w": 400.0,
            "gpu": {
                "driver": audit.EXPECTED_DRIVER,
                "compute_apps": [],
                "memory_used_mib": 15,
                "utilization_gpu_percent": 0,
            },
            "uvm_refcount": 0,
            "struct_ops": {"maps": [], "links": []},
            "dmesg_abnormal": [],
            "journal_abnormal": [],
            "xids": [],
        }
    return {
        "passed": True,
        "boot_id": "boot-A",
        "before": snapshot(),
        "after": snapshot(),
        "telemetry": {
            "samples": 2, "throttled": False, "pstates": ["P0"],
            "clock_pairs_mhz": [[2385, 14001]],
        },
    }


def correctness_cell(probe=None):
    cell = {
        "valid": True,
        "returncode": 0,
        "safety": safe_cell_record(),
        "normalized_stdout": audit.EXPECTED_OUTPUT,
        "stdout_bytes": audit.EXPECTED_OUTPUT_BYTES,
    }
    if probe is not None:
        cell.update(probe=probe, matches_baseline=True)
    return cell


def timing_cell(block, throughput, probe=None):
    cell = {
        "valid": True,
        "returncode": 0,
        "safety": safe_cell_record(),
        "block": block,
        "metrics": {"pp_tokens": 32, "pp_tok_s": throughput},
    }
    if probe is not None:
        cell["probe"] = probe
    return cell


def timing_exit_probe(pp):
    layout = runner.kernelretsnoop_layout(pp, correctness=False)
    return runner.parse_gpubpf("kernelretsnoop", lossless_exit_log(
        requested=layout["thread_slots"], allocated=layout["thread_slots"],
        requested_entries=layout["entries_per_thread"], entries=layout["entries_per_thread"],
        committed=layout["events"], collected=layout["events"],
        runtime_collected=layout["events"], nonzero=layout["events"],
        launches=layout["launches"], coordinates=layout["coordinates"],
        extent_x=layout["extent_x"], extent_y=layout["extent_y"],
        extent_z=layout["extent_z"],
        multiplicity_220=0, multiplicity_44=layout["coordinates"], multiplicity_22=0,
        other_multiplicity=0, segment_mismatches=0,
        unique_coordinates=layout["coordinates"], oracle_enabled=0,
        oracle_total_events=layout["events"], oracle_passed=0,
    ))


def nvbit_exit_probe(pp, correctness=False):
    layout = runner.kernelretsnoop_layout(pp, correctness=correctness)
    if correctness:
        multiplicities = (1024, 1024, 20480, 0)
    else:
        multiplicities = (0, layout["coordinates"], 0, 0)
    return runner.parse_nvbit("kernelretsnoop", lossless_nvbit_exit_log(
        selected=layout["launches"], events=layout["events"],
        nonzero=layout["events"], launches=layout["launches"],
        coordinates=layout["coordinates"], extent_x=layout["extent_x"],
        extent_y=layout["extent_y"], extent_z=layout["extent_z"],
        multiplicity_220=multiplicities[0], multiplicity_44=multiplicities[1],
        multiplicity_22=multiplicities[2], multiplicity_other=multiplicities[3],
        unique_coordinates=layout["coordinates"],
    ))


def two_tool_state():
    tools = ("kernelretsnoop", "threadhist")
    configs = audit.selected_configs(tools)
    params = {
        "phase": "preflight",
        "tools": list(tools),
        "runs": 1,
        "pp": 32,
        "model": "/inputs/model.gguf",
        "llama_bench": "/bin/llama-bench",
        "llama_cli": "/bin/llama-cli",
        "bpftime_root": "/source/bpftime",
        "bpftime_build_dir": "/build/bpftime",
        "nvbit_root": "/deps/nvbit_release_x86_64",
        "target_symbol": "selected_kernel",
        "tg": 0,
        "n_gpu_layers": 99,
        "timeout_s": 300,
        "probe_startup_s": 3.0,
        "gpu_thread_count": 22528,
        "threadhist_gpu_thread_count": 1048576,
        "kernelretsnoop_shm_memory_mb": 1000,
        "kernelretsnoop_correctness_exact_oracle": True,
        "kernelretsnoop_timing_exact_oracle": False,
        "kernelretsnoop_correctness_thread_slots": 22528,
        "kernelretsnoop_correctness_ring_entries_per_thread": 256,
        "kernelretsnoop_timing_thread_slots": 32768,
        "kernelretsnoop_timing_ring_entries_per_thread": 44,
        "kernelretsnoop_timing_expected_launches": 44,
        "kernelretsnoop_timing_expected_coordinates": 32768,
        "kernelretsnoop_timing_expected_events": 1441792,
        "kernelretsnoop_timing_shared_bytes": 58458144,
        "uprobe_binary": "/bin/libggml-cuda.so",
        "uprobe_symbol_hint": "selected_kernel",
        "uvm": False,
        "no_warmup": False,
        "cuda_graphs_disabled": True,
        "schedule_seed": 1797,
        "bootstrap_samples": 10000,
        "expected_driver": "575.57.08",
        "worker_cpus": "8-15",
        "telemetry_cpu": 16,
        "cpu_affinity": [0, 1],
        "launch_environment": {"PATH": "/original"},
    }
    exact_exit = runner.parse_gpubpf("kernelretsnoop", lossless_exit_log())
    timed_exit = timing_exit_probe(32)
    nvbit_exit = nvbit_exit_probe(32, correctness=True)
    gpubpf_hist = {
        "sample_count": 8192,
        "nonzero_threads": 1024,
        "configured_entries": 1048576,
        "readback_entries": 1048576,
        "readback_bytes": 8388608,
        "readback_complete": 1,
    }
    nvbit_hist = {"sample_count": 8192, "nonzero_threads": 1024,
                  "selected_launches": audit.CORRECTNESS_EXIT_LAUNCHES}
    probes = {
        "gpubpf_kernelretsnoop": (exact_exit, timed_exit),
        "nvbit_kernelretsnoop": (nvbit_exit, nvbit_exit_probe(32)),
        "gpubpf_threadhist": (gpubpf_hist, gpubpf_hist),
        "nvbit_threadhist": (nvbit_hist, nvbit_hist),
    }
    state = {
        "phase": "preflight",
        "params": params,
        "provenance": {"driver": audit.EXPECTED_DRIVER},
        "schedule": audit.fixed_schedule(configs, 1),
        "correctness": {},
        "configs": {},
    }
    for index, config in enumerate(configs):
        if config == "baseline":
            correctness = correctness_cell()
            timing = timing_cell(1, 100.0)
        else:
            correctness = correctness_cell(probes[config][0])
            timing = timing_cell(1, 90.0 - index, probes[config][1])
        state["correctness"][config] = {"attempts": [correctness]}
        state["configs"][config] = {"runs": [timing]}
    return state


def with_explicit_verifier(state, level):
    state["params"]["verifier_level"] = level
    state["params"]["verifier_runtime_configuration"] = {
        "ENABLE_EBPF_VERIFIER": "ON",
        "BPFTIME_ENABLE_CUDA_ATTACH": "ON",
        "BPFTIME_LLVM_JIT": "ON",
    }
    for matrix_name, entries_name in (("correctness", "attempts"), ("configs", "runs")):
        correctness = matrix_name == "correctness"
        executable = "llama_cli" if correctness else "llama_bench"
        for config, records in state[matrix_name].items():
            if config.startswith("gpubpf_"):
                tool = config.removeprefix("gpubpf_")
                expected_map = audit.verifier_map_expectation(
                    tool, correctness=correctness
                )
                maps = ([{"fd": 16, **expected_map}] if level == "STRICT" else [])
                evidence = {
                    "level": level,
                    "required": True,
                    "passed": True,
                    "program": "cuda__retprobe",
                    "attach": "kretprobe/selected_kernel",
                    "target_pid": 4321,
                    "execution_record": f"{executable}.execution.json",
                    "execution_error": None,
                    "expected_map": expected_map,
                    "accepted_records": 1 if level == "STRICT" else 0,
                    "instruction_counts": [13] if level == "STRICT" else [],
                    "verified_map_records": len(maps),
                    "verified_maps": maps,
                    "skipped_records": 1 if level == "NO_VERIFY" else 0,
                    "rejected": False,
                    "foreign_pid_records": 0,
                    "unexpected_target_records": 0,
                    "unparsed_records": 0,
                    "logs_scanned": [f"{executable}.log"],
                    "logs_missing": [],
                    "matched_log_sources": [f"{executable}.log"],
                }
                for cell in records[entries_name]:
                    cell["verifier"] = copy.deepcopy(evidence)
    return state


def three_tool_state():
    state = two_tool_state()
    state["params"]["tools"] = list(audit.TASKS)
    configs = audit.selected_configs(audit.TASKS)
    state["schedule"] = audit.fixed_schedule(configs, 1)
    state["provenance"]["boot_id"] = "boot-A"
    state["provenance"]["supported_clock_pairs_mhz"] = [[2385, 14001]]
    probes = {
        "gpubpf_launchlate": runner.parse_gpubpf(
            "launchlate", lossless_launchlate_log()),
        "nvbit_launchlate": runner.parse_nvbit(
            "launchlate", lossless_nvbit_launchlate_log()),
    }
    for index, config in enumerate(("gpubpf_launchlate", "nvbit_launchlate"), 5):
        state["correctness"][config] = {"attempts": [correctness_cell(probes[config])]}
        state["configs"][config] = {"runs": [timing_cell(1, 90.0 - index, probes[config])]}
    return state


def launch_only_state():
    state = three_tool_state()
    configs = audit.selected_configs(("launchlate",))
    state["params"]["tools"] = ["launchlate"]
    state["schedule"] = audit.fixed_schedule(configs, 1)
    state["correctness"] = {
        config: state["correctness"][config] for config in configs
    }
    state["configs"] = {config: state["configs"][config] for config in configs}
    return state


def full_launch_only_state(preflight_path):
    state = launch_only_state()
    state["phase"] = "full"
    state["params"].update(
        phase="full", runs=10, pp=512,
        preflight_campaign=str(Path(preflight_path).resolve()),
        kernelretsnoop_timing_thread_slots=524288,
        kernelretsnoop_timing_expected_coordinates=524288,
        kernelretsnoop_timing_expected_events=23068672,
        kernelretsnoop_timing_shared_bytes=935329824,
    )
    configs = audit.selected_configs(("launchlate",))
    state["schedule"] = audit.fixed_schedule(configs, 10)
    for config in configs:
        template = state["configs"][config]["runs"][0]
        state["configs"][config]["runs"] = []
        for block in range(1, 11):
            cell = copy.deepcopy(template)
            cell["block"] = block
            cell["metrics"]["pp_tokens"] = 512
            state["configs"][config]["runs"].append(cell)
    return state


def full_two_tool_state(preflight_path):
    state = two_tool_state()
    state["phase"] = "full"
    state["params"].update(
        phase="full", runs=10, pp=512,
        preflight_campaign=str(Path(preflight_path).resolve()),
    )
    layout = runner.kernelretsnoop_layout(512, correctness=False)
    state["params"].update(
        kernelretsnoop_timing_thread_slots=layout["thread_slots"],
        kernelretsnoop_timing_expected_coordinates=layout["coordinates"],
        kernelretsnoop_timing_expected_events=layout["events"],
        kernelretsnoop_timing_shared_bytes=layout["shared_bytes"],
    )
    state["configs"]["gpubpf_kernelretsnoop"]["runs"][0]["probe"] = timing_exit_probe(512)
    state["configs"]["nvbit_kernelretsnoop"]["runs"][0]["probe"] = \
        nvbit_exit_probe(512)
    configs = audit.selected_configs(tuple(state["params"]["tools"]))
    state["schedule"] = audit.fixed_schedule(configs, 10)
    for config in configs:
        template = state["configs"][config]["runs"][0]
        runs = []
        for block in range(1, 11):
            cell = copy.deepcopy(template)
            cell["block"] = block
            cell["metrics"]["pp_tokens"] = 512
            runs.append(cell)
        state["configs"][config]["runs"] = runs
    return state


def full_three_tool_state():
    state = three_tool_state()
    state["phase"] = "full"
    state["params"].update(phase="full", runs=10, pp=512)
    layout = runner.kernelretsnoop_layout(512, correctness=False)
    state["params"].update(
        kernelretsnoop_timing_thread_slots=layout["thread_slots"],
        kernelretsnoop_timing_expected_coordinates=layout["coordinates"],
        kernelretsnoop_timing_expected_events=layout["events"],
        kernelretsnoop_timing_shared_bytes=layout["shared_bytes"],
    )
    state["configs"]["gpubpf_kernelretsnoop"]["runs"][0]["probe"] = timing_exit_probe(512)
    state["configs"]["nvbit_kernelretsnoop"]["runs"][0]["probe"] = \
        nvbit_exit_probe(512)
    configs = audit.selected_configs(audit.TASKS)
    state["schedule"] = audit.fixed_schedule(configs, 10)
    for config in configs:
        template = state["configs"][config]["runs"][0]
        state["configs"][config]["runs"] = []
        for block in range(1, 11):
            cell = copy.deepcopy(template)
            cell["block"] = block
            cell["metrics"]["pp_tokens"] = 512
            state["configs"][config]["runs"].append(cell)
    return state


def endpoint_records():
    samples = []
    widths = []
    outer_widths = []
    for index in range(audit.LAUNCH_CONTROL_SAMPLES):
        cpu_before = 1_000_000_000 + index * 10_000
        cpu_after = cpu_before + 100
        before, after = cpu_before - 100, cpu_before + 200
        gpu = cpu_before + 1_000
        samples.append({
            "record": "sample", "index": index,
            "control_transport": "direct", "correlation_command": "endpoints-v1",
            "rm_status": 0, "host_before_ns": before, "host_after_ns": after,
            "rm_cpu_before_ns": cpu_before,
            "rm_cpu_midpoint_ns": cpu_before + 50,
            "rm_cpu_after_ns": cpu_after, "rm_gpu_ptimer_ns": gpu,
            "outer_width_ns": after - before, "max_selected_gap_ns": 100,
            "cpu_lower_ns": cpu_before, "cpu_upper_ns": cpu_after,
            "offset_low_ns": gpu - cpu_after - 32,
            "offset_high_ns": gpu - cpu_before + 32,
            "bracket_width_ns": 164, "cpu_midpoint_regression": False,
            "ptimer_regression": False, "valid": True,
        })
        widths.append(164)
        outer_widths.append(300)
    samples.append({
        "record": "summary", "setup_stage": "samples",
        "control_transport": "direct", "correlation_command": "endpoints-v1",
        "setup_error": 0, "cleanup_error": 0, "cleanup_rm_status": 0,
        "output_error": 0, "requested": audit.LAUNCH_CONTROL_SAMPLES,
        "attempted": audit.LAUNCH_CONTROL_SAMPLES,
        "accepted": audit.LAUNCH_CONTROL_SAMPLES, "rejected": 0,
        "cpu_midpoint_regressions": 0, "ptimer_regressions": 0,
        "min_outer_width_ns": min(outer_widths),
        "median_outer_width_ns": audit.integer_median(outer_widths),
        "max_outer_width_ns": max(outer_widths),
        "min_bracket_width_ns": min(widths),
        "median_bracket_width_ns": audit.integer_median(widths),
        "max_bracket_width_ns": max(widths),
        "target_median_bracket_ns": audit.LAUNCH_RM_MAX_BRACKET_NS,
        "gate_pass": True,
    })
    return samples


def identity_records():
    samples = []
    for index in range(audit.LAUNCH_CONTROL_SAMPLES):
        base = 3_000_000_000 + index * 10_000
        bo, bc, ba, bx = base, base + 100, base + 200, base + 300
        ko, kx = base + 400, base + 500
        ao, ac, aa, ax = base + 600, base + 700, base + 800, base + 900
        bg, kg, ag = base + 5_000, base + 5_100, base + 5_200
        samples.append({
            "type": "identity_sample", "trial": index,
            "rm_before_outer_before_raw_ns": bo,
            "rm_before_cpu_before_raw_ns": bc,
            "rm_before_gpu_ptimer_ns": bg,
            "rm_before_cpu_after_raw_ns": ba,
            "rm_before_outer_after_raw_ns": bx,
            "rm_before_offset_low_ns": bg - ba - 32,
            "rm_before_offset_high_ns": bg - bc + 32,
            "kernel_before_raw_ns": ko, "device_globaltimer_ns": kg,
            "kernel_after_raw_ns": kx, "rm_after_outer_before_raw_ns": ao,
            "rm_after_cpu_before_raw_ns": ac,
            "rm_after_gpu_ptimer_ns": ag,
            "rm_after_cpu_after_raw_ns": aa,
            "rm_after_outer_after_raw_ns": ax,
            "rm_after_offset_low_ns": ag - aa - 32,
            "rm_after_offset_high_ns": ag - ac + 32,
            "before_bracket_width_ns": ba - bc + 64,
            "after_bracket_width_ns": aa - ac + 64,
            "contained": True, "accepted": True,
        })
    samples.append({
        "type": "identity_summary", "requested": audit.LAUNCH_CONTROL_SAMPLES,
        "attempted": audit.LAUNCH_CONTROL_SAMPLES,
        "accepted": audit.LAUNCH_CONTROL_SAMPLES, "rejected": 0,
        "containment_failures": 0, "raw_regressions": 0,
        "ptimer_regressions": 0, "cuda_errors": 0,
        "setup_complete": True, "cleanup_complete": True, "gate_passed": True,
    })
    return samples


def write_process_evidence(directory, filename, stdout, stderr, safety):
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / filename
    log.write_text(
        f"$ /fake/client\n# cwd: /fake\n\n## stdout\n{stdout}"
        f"\n## stderr\n{stderr}\n# exit: 0\n"
    )
    log.with_suffix(".execution.json").write_text(json.dumps({
        "command": ["/fake/client"], "identity": {"pid": 4321},
        "cleanup_passed": True, "returncode": 0, "timed_out": False,
    }))
    (directory / "gpu-safety.json").write_text(json.dumps(safety))
    return log


def write_clock_telemetry(directory, *, sm=2385, memory=14001, pstate="P0"):
    (directory / "gpu-telemetry.csv").write_text(
        "timestamp, memory.used [MiB], temperature.gpu, power.draw [W], "
        "clocks.current.sm [MHz], clocks.current.memory [MHz], pstate, "
        "clocks_event_reasons.sw_power_cap, clocks_event_reasons.hw_slowdown, "
        "clocks_event_reasons.hw_thermal_slowdown, "
        "clocks_event_reasons.hw_power_brake_slowdown, "
        "clocks_event_reasons.sw_thermal_slowdown\n"
        f"t0, 10 MiB, 40, 100 W, {sm} MHz, {memory} MHz, {pstate}, "
        "Not Active, Not Active, Not Active, Not Active, Not Active\n"
    )


def materialize_launch_raw(campaign, state):
    state["provenance"]["boot_id"] = "boot-A"
    for config, group in state["correctness"].items():
        for index, cell in enumerate(group.get("attempts", ()), 1):
            if cell.get("valid") is not True:
                continue
            stdout = audit.EXPECTED_OUTPUT
            stderr = (lossless_nvbit_launchlate_log()
                      if config == "nvbit_launchlate" else "")
            directory = campaign / "correctness" / config / f"attempt_{index:02d}"
            log = write_process_evidence(
                directory, "llama_cli.log", stdout, stderr, cell["safety"]
            )
            cell["log"] = str(log.relative_to(campaign))
            if config == "gpubpf_launchlate":
                (directory / "probe.log").write_text(lossless_launchlate_log())
    for config, group in state["configs"].items():
        for index, cell in enumerate(group.get("runs", ()), 1):
            if cell.get("valid") is not True:
                continue
            throughput = float(cell["metrics"]["pp_tok_s"])
            raw = [{
                "n_prompt": state["params"]["pp"], "n_gen": 0,
                "avg_ts": throughput, "stddev_ts": 0.0,
                "samples_ts": [throughput],
            }]
            cell["raw"] = copy.deepcopy(raw)
            cell["metrics"] = {
                "pp_tok_s": throughput, "pp_stddev": 0.0,
                "pp_tokens": state["params"]["pp"],
                "pp_samples_tok_s": [throughput],
            }
            stderr = (lossless_nvbit_launchlate_log()
                      if config == "nvbit_launchlate" else "")
            directory = campaign / "timing" / config / f"cell_{index:02d}"
            log = write_process_evidence(
                directory, "llama_bench.log", json.dumps(raw), stderr, cell["safety"]
            )
            write_clock_telemetry(directory)
            cell["log"] = str(log.relative_to(campaign))
            if config == "gpubpf_launchlate":
                probe = directory / "probe.log"
                probe.write_text(lossless_launchlate_log())
                cell["probe_log"] = str(probe.relative_to(campaign))

    controls = {}
    for name, records, executable in (
        ("endpoint_precision", endpoint_records(), "rm_ptimer_correlation_sanity"),
        ("globaltimer_identity", identity_records(), "rm_globaltimer_identity"),
    ):
        directory = campaign / "clock_controls" / name
        stdout = "".join(json.dumps(record, separators=(",", ":")) + "\n"
                         for record in records)
        process = write_process_evidence(
            directory, "process.log", stdout, "", safe_cell_record()
        )
        stdout_path, stderr_path = directory / "stdout.jsonl", directory / "stderr.log"
        stdout_path.write_text(stdout)
        stderr_path.write_text("")
        executable_path = campaign / "clock_control_build" / executable
        executable_path.parent.mkdir(parents=True, exist_ok=True)
        executable_path.write_text("synthetic control executable\n")
        command = [str(executable_path), "--samples", str(audit.LAUNCH_CONTROL_SAMPLES)]
        if name == "endpoint_precision":
            command += ["--control-transport", "direct",
                        "--correlation-command", "endpoints-v1"]
        controls[name] = {
            "command": command, "returncode": 0,
            "stdout": str(stdout_path.resolve()), "stderr": str(stderr_path.resolve()),
            "safety": str((directory / "gpu-safety.json").resolve()),
            "valid": True, "error": None,
        }
        self_execution = process.with_suffix(".execution.json")
        assert self_execution.is_file()
    record = {
        "role": "calibration_only", "boot_id": "boot-A",
        "driver": audit.EXPECTED_DRIVER,
        "endpoint_precision": controls["endpoint_precision"],
        "globaltimer_identity": controls["globaltimer_identity"], "passed": True,
    }
    state["clock_controls"] = record
    (campaign / "clock-controls.json").write_text(json.dumps(record))


def analyze_state(state):
    with tempfile.TemporaryDirectory() as tmp:
        campaign = Path(tmp)
        if "launchlate" in state.get("params", {}).get("tools", ()):
            materialize_launch_raw(campaign, state)
        (campaign / "result.json").write_text(json.dumps(state))
        return audit.analyze(campaign)


class AnalyzeRevisionRQ4Tests(unittest.TestCase):
    def test_launch_control_replay_rejects_missing_and_shifted_samples(self):
        endpoints = endpoint_records()
        identity = identity_records()
        self.assertTrue(audit.endpoint_control_valid(endpoints))
        self.assertTrue(audit.identity_control_valid(identity))
        self.assertTrue(runner.endpoint_control_valid(endpoints))
        self.assertTrue(runner.identity_control_valid(identity))
        self.assertFalse(audit.endpoint_control_valid(endpoints[:-1]))
        self.assertFalse(runner.endpoint_control_valid(endpoints[:-1]))
        shifted = copy.deepcopy(identity)
        shifted[100]["device_globaltimer_ns"] += 5_100_000
        self.assertFalse(audit.identity_control_valid(shifted))
        self.assertFalse(runner.identity_control_valid(shifted))

        def with_bracket(width):
            records = endpoint_records()
            gap = width - 64
            for sample in records[:-1]:
                cpu_before = sample["rm_cpu_before_ns"]
                sample["rm_cpu_after_ns"] = cpu_before + gap
                sample["rm_cpu_midpoint_ns"] = cpu_before + gap // 2
                sample["host_after_ns"] = cpu_before + gap + 100
                sample["outer_width_ns"] = sample["host_after_ns"] - sample["host_before_ns"]
                sample["max_selected_gap_ns"] = gap
                sample["cpu_upper_ns"] = cpu_before + gap
                sample["offset_low_ns"] = sample["rm_gpu_ptimer_ns"] - (cpu_before + gap) - 32
                sample["bracket_width_ns"] = width
            summary = records[-1]
            summary.update(
                min_outer_width_ns=gap + 200,
                median_outer_width_ns=gap + 200,
                max_outer_width_ns=gap + 200,
                min_bracket_width_ns=width,
                median_bracket_width_ns=width,
                max_bracket_width_ns=width,
                gate_pass=width <= audit.LAUNCH_RM_MAX_BRACKET_NS,
            )
            return records

        at_limit = with_bracket(1500)
        above_limit = with_bracket(1501)
        self.assertTrue(audit.endpoint_control_valid(at_limit))
        self.assertTrue(runner.endpoint_control_valid(at_limit))
        self.assertFalse(audit.endpoint_control_valid(above_limit))
        self.assertFalse(runner.endpoint_control_valid(above_limit))

    def test_launch_correctness_requires_exact_220_engagement(self):
        params = three_tool_state()["params"]
        gp = runner.parse_gpubpf("launchlate", lossless_launchlate_log(
            samples=219, histogram=219, host_launches=219, host_enqueued=219,
            device_entries=219, matched=219, classified=219,
        ))
        nv = runner.parse_nvbit("launchlate", lossless_nvbit_launchlate_log(
            selected=219, samples=219,
        ))
        self.assertTrue(audit.gpubpf_valid("launchlate", gp, params, False))
        self.assertTrue(audit.nvbit_valid("launchlate", nv, params, False))
        self.assertFalse(audit.gpubpf_valid("launchlate", gp, params, True))
        self.assertFalse(audit.nvbit_valid("launchlate", nv, params, True))

    def test_launch_analysis_reopens_raw_and_fails_closed_when_it_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            state = three_tool_state()
            materialize_launch_raw(campaign, state)
            (campaign / "result.json").write_text(json.dumps(state))
            self.assertTrue(audit.analyze(campaign)["complete"])
            cell = state["configs"]["gpubpf_launchlate"]["runs"][0]
            (campaign / cell["probe_log"]).unlink()
            result = audit.analyze(campaign)
            self.assertFalse(result["complete"])
            self.assertIn({"block": 1, "config": "gpubpf_launchlate"},
                          result["rejected_cells"])

        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            state = three_tool_state()
            materialize_launch_raw(campaign, state)
            (campaign / "result.json").write_text(json.dumps(state))
            (campaign / "clock-controls.json").unlink()
            result = audit.analyze(campaign)
            self.assertFalse(result["complete"])
            self.assertFalse(
                result["launch_clock_controls"]["independently_passed"]
            )

        # Retain the attempt-08 failure shape as a regression: benchmark and
        # probe records in a tool-only directory cannot borrow safety and
        # telemetry from a separate config-specific directory.
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            state = three_tool_state()
            materialize_launch_raw(campaign, state)
            cell = state["configs"]["gpubpf_launchlate"]["runs"][0]
            config_dir = (campaign / cell["log"]).parent
            split_dir = campaign / "launchlate_run_101"
            split_dir.mkdir()
            for name in ("llama_bench.log", "llama_bench.execution.json", "probe.log"):
                config_dir.joinpath(name).replace(split_dir / name)
            cell["log"] = "launchlate_run_101/llama_bench.log"
            cell["probe_log"] = "launchlate_run_101/probe.log"
            (campaign / "result.json").write_text(json.dumps(state))
            result = audit.analyze(campaign)
            self.assertFalse(result["complete"])
            self.assertIn({"block": 1, "config": "gpubpf_launchlate"},
                          result["rejected_cells"])

    def test_launch_analysis_rejects_raw_state_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            state = three_tool_state()
            materialize_launch_raw(campaign, state)
            state["configs"]["baseline"]["runs"][0]["metrics"]["pp_tok_s"] += 1
            (campaign / "result.json").write_text(json.dumps(state))
            result = audit.analyze(campaign)
            self.assertFalse(result["complete"])

    def test_launch_clock_fairness_is_independently_replayed_from_csv(self):
        replacements = (
            ("P0", "P2"),
            ("2385 MHz, 14001 MHz", "2392 MHz, 14001 MHz"),
            ("2385 MHz, 14001 MHz", "2400 MHz, 14001 MHz"),
            ("Not Active, Not Active", "Not Active, Active"),
        )
        for old, new in replacements:
            with self.subTest(replacement=(old, new)), tempfile.TemporaryDirectory() as tmp:
                campaign = Path(tmp)
                state = launch_only_state()
                materialize_launch_raw(campaign, state)
                cell = state["configs"]["gpubpf_launchlate"]["runs"][0]
                telemetry = (campaign / cell["log"]).parent / "gpu-telemetry.csv"
                telemetry.write_text(telemetry.read_text().replace(old, new))
                (campaign / "result.json").write_text(json.dumps(state))
                result = audit.analyze(campaign)
                self.assertFalse(result["complete"])
                self.assertEqual(result["valid_complete_blocks"], 0)

        state = launch_only_state()
        state["provenance"].pop("supported_clock_pairs_mhz")
        with self.assertRaisesRegex(ValueError, "clock inventory"):
            analyze_state(state)
            self.assertIn({"block": 1, "config": "baseline"}, result["rejected_cells"])

    def test_exact_two_tool_preflight_is_complete(self):
        result = analyze_state(two_tool_state())
        self.assertTrue(result["complete"])
        self.assertEqual(result["tools"], ["kernelretsnoop", "threadhist"])
        self.assertEqual(result["valid_complete_blocks"], 1)
        self.assertEqual([row["task"] for row in result["comparisons"]], result["tools"])

    def test_explicit_verifier_cells_fail_closed_in_independent_analysis(self):
        for level in ("STRICT", "NO_VERIFY"):
            with self.subTest(level=level):
                state = with_explicit_verifier(two_tool_state(), level)
                result = analyze_state(state)
                self.assertTrue(result["complete"])
                self.assertEqual(result["verifier_level"], level)
                cell = state["correctness"]["gpubpf_threadhist"]["attempts"][0]
                cell["verifier"][
                    "verified_map_records" if level == "STRICT" else "skipped_records"
                ] = 0
                broken = analyze_state(state)
                self.assertFalse(broken["complete"])
                self.assertFalse(broken["correctness"]["gpubpf_threadhist"])

        state = with_explicit_verifier(two_tool_state(), "STRICT")
        state["params"]["verifier_runtime_configuration"]["ENABLE_EBPF_VERIFIER"] = "OFF"
        with self.assertRaisesRegex(ValueError, "enabled runtime"):
            analyze_state(state)

    def test_default_three_tool_analysis_still_requires_and_accepts_all_seven_arms(self):
        result = analyze_state(three_tool_state())
        self.assertTrue(result["complete"])
        self.assertEqual(result["tools"], list(audit.TASKS))
        self.assertEqual(len(result["configs"]), 7)

        full = analyze_state(full_three_tool_state())
        self.assertTrue(full["complete"])
        self.assertEqual(full["preflight_gate"], {
            "required": False, "campaign": None, "independently_complete": None,
        })

    def test_launch_only_full_is_ten_raw_replayed_three_arm_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight_dir, full_dir = root / "preflight", root / "full"
            preflight_dir.mkdir()
            full_dir.mkdir()
            preflight = launch_only_state()
            full = full_launch_only_state(preflight_dir)
            materialize_launch_raw(preflight_dir, preflight)
            materialize_launch_raw(full_dir, full)
            (preflight_dir / "result.json").write_text(json.dumps(preflight))
            (full_dir / "result.json").write_text(json.dumps(full))
            result = audit.analyze(full_dir)
            self.assertTrue(result["complete"])
            self.assertEqual(result["configs"], [
                "baseline", "gpubpf_launchlate", "nvbit_launchlate",
            ])
            self.assertEqual(result["valid_complete_blocks"], 10)
            comparison = result["comparisons"][0]
            self.assertEqual(comparison["paired_blocks"], 10)
            self.assertEqual(len(comparison["raw_paired_triples"]), 10)
            self.assertIsNotNone(comparison["median_paired_effect"])

    def test_each_exact_engagement_gate_fails_closed(self):
        mutations = (
            ("gpubpf_kernelretsnoop", "full_drops", 1),
            ("nvbit_kernelretsnoop", "nonzero_timestamps", 1),
            ("gpubpf_threadhist", "readback_entries", 4095),
            ("nvbit_threadhist", "nonzero_threads", 0),
        )
        for config, field, value in mutations:
            with self.subTest(config=config, field=field):
                state = two_tool_state()
                state["correctness"][config]["attempts"][0]["probe"][field] = value
                result = analyze_state(state)
                self.assertFalse(result["complete"])
                self.assertFalse(result["correctness"][config])

    def test_frozen_defining_parameters_are_exactly_audited(self):
        mutations = {
            "tg": 1,
            "n_gpu_layers": 98,
            "gpu_thread_count": 22527,
            "threadhist_gpu_thread_count": 1048575,
            "kernelretsnoop_shm_memory_mb": 999,
            "kernelretsnoop_correctness_exact_oracle": False,
            "kernelretsnoop_timing_exact_oracle": True,
            "kernelretsnoop_correctness_thread_slots": 22527,
            "kernelretsnoop_correctness_ring_entries_per_thread": 255,
            "kernelretsnoop_timing_thread_slots": 32767,
            "kernelretsnoop_timing_ring_entries_per_thread": 15,
            "kernelretsnoop_timing_expected_launches": 43,
            "kernelretsnoop_timing_expected_coordinates": 32767,
            "kernelretsnoop_timing_expected_events": 1441791,
            "kernelretsnoop_timing_shared_bytes": 58458143,
            "schedule_seed": 1798,
            "bootstrap_samples": 9999,
            "expected_driver": "610.43.02",
            "uvm": True,
            "no_warmup": True,
            "cuda_graphs_disabled": False,
            "worker_cpus": "0-7",
            "telemetry_cpu": 15,
            "target_symbol": "different_kernel",
            "timeout_s": 0,
            "probe_startup_s": float("inf"),
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                state = two_tool_state()
                state["params"][key] = value
                message = (
                    "non-finite" if key == "probe_startup_s"
                    else key if key != "target_symbol" else "target_symbol"
                )
                with self.assertRaisesRegex(ValueError, message):
                    analyze_state(state)

        for key in ("model", "llama_bench", "llama_cli", "bpftime_root",
                    "bpftime_build_dir", "nvbit_root", "uprobe_binary"):
            for value in ("relative/path", 7):
                with self.subTest(key=key, value=value):
                    state = two_tool_state()
                    state["params"][key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        analyze_state(state)

        state = two_tool_state()
        state["params"]["cpu_affinity"] = [999]
        state["params"]["launch_environment"] = {"PATH": "/changed", "EXTRA": "value"}
        self.assertTrue(analyze_state(state)["complete"])

    def test_each_recorded_safety_field_is_independently_checked(self):
        mutations = (
            (("passed",), False),
            (("before", "gpu", "driver"), "610.43.02"),
            (("after", "gpu", "compute_apps"), [{"pid": 1}]),
            (("before", "uvm_refcount"), 1),
            (("after", "struct_ops", "maps"), ["map"]),
            (("before", "struct_ops", "links"), ["link"]),
            (("after", "xids"), ["Xid"]),
            (("before", "dmesg_abnormal"), ["fault"]),
            (("after", "journal_abnormal"), ["fault"]),
            (("before", "power_limit_service"), "inactive"),
            (("after", "power_limit_w"), 399.0),
            (("before", "gpu", "memory_used_mib"), 257),
            (("after", "gpu", "utilization_gpu_percent"), 1),
            (("telemetry", "samples"), 0),
            (("telemetry", "throttled"), True),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                state = two_tool_state()
                safety = state["correctness"]["baseline"]["attempts"][0]["safety"]
                target = safety
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                result = analyze_state(state)
                self.assertFalse(result["complete"])
                self.assertFalse(result["correctness"]["baseline"])

    def test_fixed_matrix_rejects_extra_or_missing_configuration(self):
        for mutation in ("extra", "schedule"):
            with self.subTest(mutation=mutation):
                state = two_tool_state()
                if mutation == "extra":
                    state["configs"]["gpubpf_launchlate"] = {"runs": []}
                else:
                    state["schedule"]["1"].pop()
                with self.assertRaisesRegex(ValueError, "matrix|schedule"):
                    analyze_state(state)

    def test_kernelret_timing_pair_mismatch_is_rejected(self):
        state = two_tool_state()
        state["configs"]["nvbit_kernelretsnoop"]["runs"][0]["probe"]["sample_count"] -= 1
        state["configs"]["nvbit_kernelretsnoop"]["runs"][0]["probe"]["nonzero_timestamps"] -= 1
        result = analyze_state(state)
        self.assertFalse(result["complete"])
        self.assertEqual(result["valid_complete_blocks"], 0)

    def test_kernelret_timing_geometry_and_ring_are_independently_exact(self):
        for field in ("requested_thread_slots", "allocated_thread_slots",
                      "requested_entries_per_thread", "entries_per_thread",
                      "sample_count", "cartesian_launches", "cartesian_coordinates",
                      "multiplicity_44"):
            with self.subTest(field=field):
                state = two_tool_state()
                probe = state["configs"]["gpubpf_kernelretsnoop"]["runs"][0]["probe"]
                probe[field] -= 1
                result = analyze_state(state)
                self.assertFalse(result["complete"])
                self.assertEqual(result["valid_complete_blocks"], 0)

    def test_selecting_three_tools_does_not_reclassify_launchlate_failure(self):
        state = three_tool_state()
        for config in ("gpubpf_launchlate", "nvbit_launchlate"):
            state["correctness"][config] = {"attempts": [{"valid": False, "returncode": 1}]}
            state["configs"][config] = {"runs": []}
        result = analyze_state(state)
        self.assertFalse(result["complete"])
        self.assertFalse(result["correctness"]["gpubpf_launchlate"])
        self.assertFalse(result["correctness"]["nvbit_launchlate"])
        self.assertIn("remain failures", result["scope_policy"])

    def test_full_analysis_reaudits_absolute_disjoint_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight_dir = root / "preflight"
            full_dir = root / "full"
            preflight_dir.mkdir()
            full_dir.mkdir()
            preflight_state = two_tool_state()
            full_state = full_two_tool_state(preflight_dir)
            (preflight_dir / "result.json").write_text(json.dumps(preflight_state))
            (full_dir / "result.json").write_text(json.dumps(full_state))

            result = audit.analyze(full_dir)
            self.assertTrue(result["complete"])
            self.assertEqual(result["preflight_gate"], {
                "required": True,
                "campaign": str(preflight_dir.resolve()),
                "independently_complete": True,
            })

            preflight_state["correctness"]["baseline"]["attempts"][0]["safety"][
                "before"]["gpu"]["driver"] = "610.43.02"
            (preflight_dir / "result.json").write_text(json.dumps(preflight_state))
            with self.assertRaisesRegex(ValueError, "not independently complete"):
                audit.analyze(full_dir)

    def test_full_analysis_rejects_relative_or_nested_preflight_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_dir = root / "full"
            full_dir.mkdir()
            for reference, message in (("relative/preflight", "not absolute"),
                                       (str(root), "mutually non-nested"),
                                       (str(full_dir / "preflight"), "mutually non-nested")):
                with self.subTest(reference=reference):
                    state = full_two_tool_state(root / "separate-preflight")
                    state["params"]["preflight_campaign"] = reference
                    (full_dir / "result.json").write_text(json.dumps(state))
                    with self.assertRaisesRegex(ValueError, message):
                        audit.analyze(full_dir)


if __name__ == "__main__":
    unittest.main()
