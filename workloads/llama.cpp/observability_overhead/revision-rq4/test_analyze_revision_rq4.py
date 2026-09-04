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
        "before": snapshot(),
        "after": snapshot(),
        "telemetry": {"samples": 2, "throttled": False},
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
    timed_exit = {
        **exact_exit,
        "oracle_enabled": 0,
        "oracle_passed": 0,
    }
    nvbit_exit = {
        "sample_count": audit.CORRECTNESS_EXIT_EVENTS,
        "nonzero_timestamps": audit.CORRECTNESS_EXIT_EVENTS,
        "selected_launches": audit.CORRECTNESS_EXIT_LAUNCHES,
    }
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
        "nvbit_kernelretsnoop": (nvbit_exit, nvbit_exit),
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


def three_tool_state():
    state = two_tool_state()
    state["params"]["tools"] = list(audit.TASKS)
    configs = audit.selected_configs(audit.TASKS)
    state["schedule"] = audit.fixed_schedule(configs, 1)
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


def full_two_tool_state(preflight_path):
    state = two_tool_state()
    state["phase"] = "full"
    state["params"].update(
        phase="full", runs=10, pp=512,
        preflight_campaign=str(Path(preflight_path).resolve()),
    )
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


def analyze_state(state):
    with tempfile.TemporaryDirectory() as tmp:
        campaign = Path(tmp)
        (campaign / "result.json").write_text(json.dumps(state))
        return audit.analyze(campaign)


class AnalyzeRevisionRQ4Tests(unittest.TestCase):
    def test_exact_two_tool_preflight_is_complete(self):
        result = analyze_state(two_tool_state())
        self.assertTrue(result["complete"])
        self.assertEqual(result["tools"], ["kernelretsnoop", "threadhist"])
        self.assertEqual(result["valid_complete_blocks"], 1)
        self.assertEqual([row["task"] for row in result["comparisons"]], result["tools"])

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
                with self.assertRaisesRegex(ValueError, key if key != "target_symbol" else "target_symbol"):
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
