#!/usr/bin/env python3
"""CPU-only fixtures for the S0 runner and independent analyzer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("s0_runner", "run_device_verifier_s0.py")
analyzer = load("s0_analyzer", "analyze_device_verifier_s0.py")
TARGET = "target_kernel"


def prefix(pid: int) -> str:
    return f"[stamp][info][{pid}] "


def admission(tool: str, treatment: str, pid: int) -> str:
    if treatment == "control":
        return ""
    if treatment == "NO_VERIFY":
        return prefix(pid) + "Skipping GPU eBPF verification for cuda__retprobe\n"
    expected = runner.expected_map(tool)
    return "".join((
        prefix(pid) + "GPU eBPF verification timing: program=cuda__retprobe verification_elapsed_ns=17\n",
        prefix(pid) + "GPU eBPF verification accepted: mode=STRICT program=cuda__retprobe "
        + f"attach=kretprobe/{TARGET} instructions=19\n",
        prefix(pid) + "GPU eBPF verified map: program=cuda__retprobe fd=0 "
        + f"type={expected['type']} key_size={expected['key_size']} "
        + f"value_size={expected['value_size']} max_entries={expected['max_entries']}\n",
    ))


def bench_json(pp: int, model: Path, throughput: float) -> str:
    row = [{"build_commit": "fixture-build", "build_number": 7102,
            "model_filename": str(model), "n_gpu_layers": 99, "n_prompt": pp,
            "n_gen": 0, "avg_ns": int(pp / throughput * 1e9), "avg_ts": throughput,
            "samples_ns": [int(pp / throughput * 1e9)], "samples_ts": [throughput]}]
    return json.dumps(row, indent=2)


def target_log(pp: int, model: Path, throughput: float, records: str) -> str:
    return ("$ llama-bench\n# cwd: fixture\n\n## stdout\n"
            + bench_json(pp, model, throughput) + "\n\n## stderr\n"
            + records + "\n# exit: 0\n")


def safety_snapshot(timestamp_ns: int) -> dict:
    return {
        "timestamp_ns": timestamp_ns, "power_limit_service": "active",
        "power_limit_w": 400.0,
        "gpu": {"driver": analyzer.EXPECTED_DRIVER, "memory_used_mib": 15,
                "utilization_gpu_percent": 0, "compute_apps": []},
        "uvm_refcount": 0, "struct_ops": {"maps": [], "links": []},
        "dmesg_abnormal": [], "journal_abnormal": [], "xids": [],
    }


def telemetry_text() -> str:
    inactive = ["Not Active"] * 5
    rows = [
        ["2026/09/04 00:00:00.000", "10 MiB", "40", "20 W", "100 MHz", "5000 MHz",
         *inactive],
        ["2026/09/04 00:00:00.200", "20 MiB", "41", "30 W", "200 MHz", "5000 MHz",
         "Active", *inactive[1:]],
    ]
    return ", ".join(analyzer.TELEMETRY_HEADERS) + "\n" + "".join(
        ", ".join(row) + "\n" for row in rows
    )


def safety_record() -> dict:
    return {
        "passed": True, "worker_cpus": analyzer.EXPECTED_WORKER_CPUS,
        "boot_id": "fixture-boot", "before": safety_snapshot(100),
        "after": safety_snapshot(200),
        "telemetry": {
            "samples": 2, "peak_memory_mib": 20.0, "peak_temperature_c": 41.0,
            "mean_power_w": 25.0, "min_sm_clock_mhz": 100.0,
            "max_sm_clock_mhz": 200.0, "throttled": False,
            "fixed_power_cap_samples": 1,
        },
    }


def kernel_probe(pp: int) -> str:
    slots = pp * 1024
    events = slots * 44
    values = (
        ("Total events collected", events), ("Nonzero timestamps", events),
        ("Requested thread slots", slots), ("Allocated thread slots", slots),
        ("Ring entries per thread", 44), ("Requested ring entries per thread", 44),
        ("Record bytes", 32), ("Committed events", events),
        ("Runtime collected events", events), ("OOB drops", 0), ("Full drops", 0),
        ("Bad-size drops", 0), ("Other drops", 0), ("Dirty slots", 0),
        ("Pending events", 0), ("Final drain events", 1), ("Second drain events", 0),
        ("Cartesian launches", 44), ("Cartesian coordinates", slots),
        ("Cartesian complete", 1), ("Coordinate extent x", slots // 256),
        ("Coordinate extent y", 256), ("Coordinate extent z", 1),
        ("Coordinate multiplicity 220", 0), ("Coordinate multiplicity 44", slots),
        ("Coordinate multiplicity 22", 0), ("Coordinate multiplicity other", 0),
        ("Coordinate segment mismatches", 0), ("Invalid launch coordinates", 0),
        ("Unique coordinates", slots), ("Multiplicity oracle enabled", 0),
        ("Multiplicity oracle total events", events), ("Multiplicity oracle passed", 0),
        ("Collector gate passed", 1),
    )
    return "".join(f"{label}: {value}\n" for label, value in values)


def thread_probe() -> str:
    return ("Configured thread entries: 1048576\nReadback entries: 1048576\n"
            "Readback bytes: 8388608\nReadback complete: 1\n"
            "Nonzero threads: 22528\nTotal exit probes: 720896\n")


class ScheduleTests(unittest.TestCase):
    def test_fixed_plan_has_one_correctness_and_ten_complete_timing_blocks(self):
        plan = runner.fixed_plan()
        correctness, timing = analyzer.fixed_schedules()
        self.assertEqual(plan["correctness_schedule"], correctness)
        self.assertEqual(plan["timing_schedule"], timing)
        self.assertEqual(len(correctness), 6)
        self.assertEqual(len(timing), 60)
        for tool in runner.TOOLS:
            self.assertEqual({cell["treatment"] for cell in correctness if cell["tool"] == tool},
                             set(runner.TREATMENTS))
            for block in range(1, 11):
                cells = [cell for cell in timing if cell["tool"] == tool and cell["block"] == block]
                self.assertEqual({cell["treatment"] for cell in cells}, set(runner.TREATMENTS))
                self.assertEqual(sorted(cell["position"] for cell in cells), [1, 2, 3])

    def test_binary_contract_requires_enabled_markers_and_fresh_dsos(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runtime"
            build = Path(temporary) / "build"
            source = root / "attach/nv_attach_impl/nv_attach_impl.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("\n".join((*runner.RUNTIME_BINARY_MARKERS, "verifier unavailable")))
            binaries = (build / "runtime/agent/libbpftime-agent.so",
                        build / "runtime/syscall-server/libbpftime-syscall-server.so")
            for binary in binaries:
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"\0".join(marker.encode() for marker in runner.RUNTIME_BINARY_MARKERS))
            os.utime(source, ns=(1_000_000_000, 1_000_000_000))
            for binary in binaries:
                os.utime(binary, ns=(2_000_000_000, 2_000_000_000))
            self.assertTrue(runner.runtime_binary_contract(root, build)["passed"])
            os.utime(source, ns=(3_000_000_000, 3_000_000_000))
            self.assertFalse(runner.runtime_binary_contract(root, build)["passed"])
            os.utime(source, ns=(1_000_000_000, 1_000_000_000))
            binaries[0].write_bytes(b"missing contract literals")
            self.assertFalse(runner.runtime_binary_contract(root, build)["passed"])

    def test_threadhist_complete_block_requires_matched_instrumented_events(self):
        cells = [
            {"tool": "threadhist", "block": 1, "treatment": "control", "valid": True},
            {"tool": "threadhist", "block": 1, "treatment": "STRICT", "valid": True,
             "probe": {"sample_count": 10}},
            {"tool": "threadhist", "block": 1, "treatment": "NO_VERIFY", "valid": True,
             "probe": {"sample_count": 11}},
        ]
        self.assertFalse(runner.reconcile_complete_block(
            cells, {"tool": "threadhist", "block": 1}))
        self.assertTrue(all(cell["valid"] is False for cell in cells))


class ParserTests(unittest.TestCase):
    def parse(self, tool: str, treatment: str, records: str, pid: int = 73):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "llama_bench.log"
            execution = root / "llama_bench.execution.json"
            log.write_text(target_log(512, root / "model", 100.0, records))
            execution.write_text(json.dumps({"identity": {"pid": pid, "start_ticks": 9001},
                                             "cleanup_passed": True,
                                             "timed_out": False, "returncode": 0}))
            return runner.parse_admission(log, execution, tool=tool, treatment=treatment,
                                          target_symbol=TARGET)

    def test_all_treatments_have_disjoint_exact_admission_contracts(self):
        for treatment in runner.TREATMENTS:
            with self.subTest(treatment=treatment):
                self.assertTrue(self.parse("kernelretsnoop", treatment,
                                           admission("kernelretsnoop", treatment, 73))["passed"])

    def test_wrong_pid_duplicate_timing_and_noverify_timing_fail(self):
        strict = admission("threadhist", "STRICT", 73)
        self.assertFalse(self.parse("threadhist", "STRICT", strict, pid=74)["passed"])
        self.assertFalse(self.parse("threadhist", "STRICT", strict + prefix(73)
            + "GPU eBPF verification timing: program=cuda__retprobe verification_elapsed_ns=9\n")["passed"])
        no_verify = admission("threadhist", "NO_VERIFY", 73) + prefix(73)
        no_verify += "GPU eBPF verification timing: program=cuda__retprobe verification_elapsed_ns=9\n"
        self.assertFalse(self.parse("threadhist", "NO_VERIFY", no_verify)["passed"])

    def test_bench_gate_cross_checks_average_sample_and_elapsed_formula(self):
        args = runner.parse_args([])
        args.pp = 512
        row = {"build_commit": "fixture-build", "build_number": 7102,
               "model_filename": str(args.model), "n_gpu_layers": 99,
               "n_prompt": 512, "n_gen": 0, "avg_ns": 5_120_000_000,
               "avg_ts": 100.0, "samples_ns": [5_120_000_000],
               "samples_ts": [100.0]}
        parsed = {"raw": [row], "metrics": {"pp_tokens": 512, "pp_tok_s": 100.0}}
        self.assertTrue(runner.validate_bench_output(parsed, args))
        row["samples_ts"] = [99.0]
        self.assertFalse(runner.validate_bench_output(parsed, args))
        row["samples_ts"] = [100.0]
        row["avg_ns"] = 4_000_000_000
        self.assertFalse(runner.validate_bench_output(parsed, args))

    def test_bench_gate_accepts_only_six_significant_digit_sample_rounding(self):
        args = runner.parse_args([])
        args.pp = 512
        row = {"build_commit": "fixture-build", "build_number": 7102,
               "model_filename": str(args.model), "n_gpu_layers": 99,
               "n_prompt": 512, "n_gen": 0, "avg_ns": 13_425_897,
               "avg_ts": 38_135.254576, "samples_ns": [13_425_897],
               "samples_ts": [38_135.3]}
        parsed = {"raw": [row], "metrics": {"pp_tokens": 512,
                                               "pp_tok_s": 38_135.254576}}
        self.assertTrue(runner.validate_bench_output(parsed, args))
        self.assertTrue(analyzer.bench_gate(
            json.dumps([row]), 512, args.model, 99)["valid"])
        row["samples_ts"] = [38_135.4]
        self.assertFalse(runner.validate_bench_output(parsed, args))
        self.assertFalse(analyzer.bench_gate(
            json.dumps([row]), 512, args.model, 99)["valid"])

    def test_log_footer_is_one_final_signed_integer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target.log"
            path.write_text("$ cmd\n\n## stdout\n[]\n## stderr\nmessage\n# exit: +0\n")
            self.assertEqual(analyzer.log_sections(path), ("[]", "message", 0))
            for bad in (
                "$ cmd\n\n## stdout\n[]\n## stderr\nmessage\n# exit: nope\n",
                "$ cmd\n\n## stdout\n[]\n## stderr\nmessage\n# exit: 0\ntrailing\n",
                "$ cmd\n\n## stdout\n[]\n## stderr\n# exit: 0\n# exit: 0\n",
            ):
                with self.subTest(log=bad):
                    path.write_text(bad)
                    with self.assertRaises(ValueError):
                        analyzer.log_sections(path)

    def test_safety_snapshot_rederives_every_fixed_gate(self):
        mutations = (
            (("timestamp_ns",), 0), (("power_limit_service",), "inactive"),
            (("power_limit_w",), 399.0), (("uvm_refcount",), 1),
            (("dmesg_abnormal",), ["fault"]), (("journal_abnormal",), ["fault"]),
            (("xids",), ["NVRM: Xid"]), (("struct_ops", "maps"), ["map"]),
            (("struct_ops", "links"), ["link"]), (("gpu", "driver"), "wrong"),
            (("gpu", "compute_apps"), ["pid"]), (("gpu", "memory_used_mib"), 257),
            (("gpu", "utilization_gpu_percent"), 1),
        )
        self.assertTrue(analyzer.safety_snapshot_valid(safety_snapshot(100)))
        for path, value in mutations:
            with self.subTest(path=path):
                snapshot = safety_snapshot(100)
                target = snapshot
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertFalse(analyzer.safety_snapshot_valid(snapshot))

    def test_telemetry_replay_is_strict_and_allows_only_fixed_power_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu-telemetry.csv"
            path.write_text(telemetry_text())
            self.assertEqual(analyzer.replay_gpu_telemetry(path), safety_record()["telemetry"])
            mutations = {
                "non_power_throttle": telemetry_text().replace(
                    ", Not Active, Not Active, Not Active, Not Active, Not Active\n",
                    ", Not Active, Active, Not Active, Not Active, Not Active\n", 1),
                "extra_field": telemetry_text().replace("10 MiB", "10 MiB, extra", 1),
                "missing_sample": ", ".join(analyzer.TELEMETRY_HEADERS) + "\n",
                "nonfinite": telemetry_text().replace("20 W", "nan W", 1),
            }
            for name, text in mutations.items():
                with self.subTest(name=name):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        analyzer.replay_gpu_telemetry(path)

    def test_safety_record_rejects_attested_errors_identity_and_summary_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "gpu-telemetry.csv").write_text(telemetry_text())
            self.assertTrue(analyzer.safety_gate(
                directory, safety_record(), boot_id="fixture-boot")["valid"])
            mutations = ("error", "cleanup_errors", "fatal_cleanup", "boot_id",
                         "worker_cpus", "timestamp_order", "summary")
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    record = safety_record()
                    if mutation in {"error", "cleanup_errors", "fatal_cleanup"}:
                        record[mutation] = "injected"
                    elif mutation == "boot_id":
                        record["boot_id"] = "other-boot"
                    elif mutation == "worker_cpus":
                        record["worker_cpus"] = "0"
                    elif mutation == "timestamp_order":
                        record["after"]["timestamp_ns"] = 99
                    else:
                        record["telemetry"]["samples"] = 3
                    self.assertFalse(analyzer.safety_gate(
                        directory, record, boot_id="fixture-boot")["valid"])


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.model = root / "model.gguf"
        self.bench = root / "llama-bench"
        self.build = root / "build"
        self.pid = 1000
        self.plan = runner.fixed_plan()
        self.state = {
            "schema": runner.SCHEMA, "status": "complete", "plan": self.plan,
            "defining_inputs": {"model": str(self.model), "llama_bench": str(self.bench),
                                "bpftime_build_dir": str(self.build), "target_symbol": TARGET,
                                "n_gpu_layers": 99, "warmup": True},
            "host": {"driver": analyzer.EXPECTED_DRIVER,
                     "expected_driver": analyzer.EXPECTED_DRIVER,
                     "boot_id": "fixture-boot"},
            "runtime": {"build_configuration": {key: "ON" for key in analyzer.BUILD_KEYS},
                        "source_contract": {"passed": True}, "binary_contract": {"passed": True},
                        "agent": {"exists": True, "bytes": 10},
                        "syscall_server": {"exists": True, "bytes": 10}},
            "objects": {tool: {"exists": True, "bytes": 10} for tool in runner.TOOLS},
            "correctness_cells": [], "timing_cells": [],
        }
        for specification in self.plan["correctness_schedule"]:
            self.state["correctness_cells"].append(self.make_cell("correctness", specification))
        for specification in self.plan["timing_schedule"]:
            self.state["timing_cells"].append(self.make_cell("timing", specification))
        self.write()

    def throughput(self, specification: dict) -> float:
        base = 100.0 + specification["block"]
        return {"control": base, "NO_VERIFY": base * 0.9,
                "STRICT": base * 0.9 * 0.99}[specification["treatment"]]

    def make_cell(self, stage: str, specification: dict) -> dict:
        self.pid += 1
        treatment = specification["treatment"]
        relative = f"{stage}/{specification['sequence']:03d}-{specification['tool']}-{treatment.lower()}"
        directory = self.root / relative
        directory.mkdir(parents=True)
        (directory / "llama_bench.log").write_text(target_log(
            specification["pp"], self.model, self.throughput(specification),
            admission(specification["tool"], treatment, self.pid)))
        identity = {"pid": self.pid, "start_ticks": self.pid * 10}
        defining = self.state["defining_inputs"]
        (directory / "llama_bench.execution.json").write_text(json.dumps({
            "identity": identity, "cleanup_passed": True, "timed_out": False,
            "returncode": 0,
            "command": analyzer.expected_command(defining, treatment, specification["pp"])}))
        (directory / "gpu-safety.json").write_text(json.dumps(safety_record()))
        (directory / "gpu-telemetry.csv").write_text(telemetry_text())
        target_environment = {}
        if treatment != "control":
            segment = f"rq4_s0_fixture_{self.pid}"
            (directory / "probe-execution.json").write_text(json.dumps({
                "private_segment": segment, "private_segment_removed": True,
                "agent_environment": {"BPFTIME_GLOBAL_SHM_NAME": segment,
                                      "BPFTIME_VERIFIER_LEVEL": treatment, "SPDLOG_LEVEL": "info",
                                      "LD_PRELOAD": str(self.build / "runtime/agent/libbpftime-agent.so")},
                "loader_environment": {"BPFTIME_GLOBAL_SHM_NAME": segment,
                                       "BPFTIME_VERIFIER_LEVEL": treatment, "SPDLOG_LEVEL": "info",
                                       "LD_PRELOAD": str(self.build / "runtime/syscall-server/libbpftime-syscall-server.so")}}))
            probe = kernel_probe(specification["pp"]) if specification["tool"] == "kernelretsnoop" else thread_probe()
            (directory / "probe.log").write_text(probe)
            target_environment = {
                "BPFTIME_GLOBAL_SHM_NAME": segment, "BPFTIME_VERIFIER_LEVEL": treatment,
                "LD_PRELOAD": str(self.build / "runtime/agent/libbpftime-agent.so"),
            }
        (directory / "target-environment.json").write_text(json.dumps(target_environment))
        artifacts = {"agent": self.state["runtime"]["agent"],
                     "syscall_server": self.state["runtime"]["syscall_server"],
                     "object": self.state["objects"][specification["tool"]]}
        return {"stage": stage, **specification, "directory": relative, "valid": True,
                "execution_identity": identity, "identity_directory_valid": True,
                "artifacts_before": artifacts, "artifacts_after": artifacts,
                "artifacts_stable": True}

    def write(self) -> None:
        (self.root / "result.json").write_text(json.dumps(self.state))


class AnalyzerTests(unittest.TestCase):
    def test_complete_fixture_reopens_raw_logs_and_computes_three_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            result = analyzer.analyze(fixture.root)
            self.assertTrue(result["complete"], result["errors"])
            self.assertFalse(result["admission_timing_used_in_throughput"])
            self.assertEqual(result["llama_bench_build"], {
                "consistent": True, "cells": 66,
                "build_commit": "fixture-build", "build_number": 7102,
            })
            for tool in runner.TOOLS:
                summary = result["summary"][tool]
                self.assertEqual(summary["complete_blocks"], 10)
                self.assertAlmostEqual(summary["strict_vs_no_verify"]["mean_percent"], -1.0)
                self.assertAlmostEqual(summary["no_verify_vs_control"]["mean_percent"], -10.0)

    def test_missing_extra_and_duplicate_cells_are_rejected(self):
        for mutation, expected_status in (("missing", "invalid"), ("extra", "invalid"),
                                          ("duplicate", "invalid")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                if mutation == "missing":
                    fixture.state["timing_cells"].pop()
                elif mutation == "extra":
                    fixture.state["timing_cells"].append({"sequence": 999})
                else:
                    fixture.state["timing_cells"].append(dict(fixture.state["timing_cells"][0]))
                fixture.write()
                result = analyzer.analyze(fixture.root)
                self.assertFalse(result["complete"])
                self.assertEqual(result["run_status"], expected_status)

    def test_explicit_invalid_prefix_overrides_short_incomplete_inventory(self):
        for status, expected in (("invalid_correctness", "invalid"),
                                 ("invalid_timing", "invalid"), ("running", "incomplete")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                fixture.state["status"] = status
                fixture.state["timing_cells"] = fixture.state["timing_cells"][:3]
                fixture.write()
                result = analyzer.analyze(fixture.root)
                self.assertEqual(result["run_status"], expected)

    def test_reused_private_shm_and_raw_throughput_tampering_fail(self):
        for mutation in ("shm", "throughput", "sample", "elapsed_formula", "event_count"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                instrumented = [cell for cell in fixture.state["timing_cells"]
                                if cell["treatment"] != "control"]
                if mutation == "shm":
                    first = fixture.root / instrumented[0]["directory"] / "probe-execution.json"
                    second = fixture.root / instrumented[1]["directory"] / "probe-execution.json"
                    one, two = json.loads(first.read_text()), json.loads(second.read_text())
                    two["private_segment"] = one["private_segment"]
                    second.write_text(json.dumps(two))
                else:
                    if mutation in ("throughput", "sample", "elapsed_formula"):
                        path = fixture.root / fixture.state["timing_cells"][0]["directory"] / "llama_bench.log"
                        text = path.read_text()
                        if mutation == "throughput":
                            text = text.replace('"avg_ts":', '"wrong_avg_ts":', 1)
                        elif mutation == "sample":
                            text = text.replace('"samples_ts": [', '"samples_ts": [999,', 1)
                        else:
                            text = re.sub(r'"avg_ns":\s*[0-9]+', '"avg_ns": 1', text, count=1)
                        path.write_text(text)
                    else:
                        cell = next(cell for cell in fixture.state["timing_cells"]
                                    if cell["tool"] == "threadhist" and cell["treatment"] == "STRICT")
                        path = fixture.root / cell["directory"] / "probe.log"
                        path.write_text(path.read_text().replace(
                            "Total exit probes: 720896", "Total exit probes: 720897"))
                self.assertFalse(analyzer.analyze(fixture.root)["complete"])

    def test_noncanonical_or_reused_directories_and_execution_identity_fail(self):
        for mutation in ("noncanonical", "reused_directory", "reused_identity"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                first, second = fixture.state["timing_cells"][:2]
                if mutation == "noncanonical":
                    old = fixture.root / first["directory"]
                    new = old.parent / "alias"
                    old.rename(new)
                    first["directory"] = str(new.relative_to(fixture.root))
                elif mutation == "reused_directory":
                    second["directory"] = first["directory"]
                else:
                    first_execution = json.loads((fixture.root / first["directory"] /
                                                  "llama_bench.execution.json").read_text())
                    second_path = fixture.root / second["directory"] / "llama_bench.execution.json"
                    second_execution = json.loads(second_path.read_text())
                    old_pid = second_execution["identity"]["pid"]
                    second_execution["identity"] = dict(first_execution["identity"])
                    second_path.write_text(json.dumps(second_execution))
                    second["execution_identity"] = dict(first_execution["identity"])
                    log_path = fixture.root / second["directory"] / "llama_bench.log"
                    log_path.write_text(log_path.read_text().replace(
                        f"][{old_pid}] ", f"][{first_execution['identity']['pid']}] "))
                fixture.write()
                self.assertFalse(analyzer.analyze(fixture.root)["complete"])

    def test_exact_command_and_environment_isolation_fail_closed(self):
        for mutation in ("control_preload", "control_bpftime", "wrong_instrumented_preload",
                         "wrong_fixed_flags"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                if mutation.startswith("control"):
                    cell = next(cell for cell in fixture.state["timing_cells"]
                                if cell["treatment"] == "control")
                else:
                    cell = next(cell for cell in fixture.state["timing_cells"]
                                if cell["treatment"] == "STRICT")
                directory = fixture.root / cell["directory"]
                execution_path = directory / "llama_bench.execution.json"
                execution = json.loads(execution_path.read_text())
                environment_path = directory / "target-environment.json"
                environment = json.loads(environment_path.read_text())
                if mutation == "control_preload":
                    execution["command"].insert(4, "LD_PRELOAD=/wrong.so")
                elif mutation == "control_bpftime":
                    environment["BPFTIME_VERIFIER_LEVEL"] = "STRICT"
                elif mutation == "wrong_instrumented_preload":
                    environment["LD_PRELOAD"] = "/wrong.so"
                else:
                    index = execution["command"].index("-r")
                    execution["command"][index + 1] = "2"
                execution_path.write_text(json.dumps(execution))
                environment_path.write_text(json.dumps(environment))
                self.assertFalse(analyzer.analyze(fixture.root)["complete"])

    def test_exit_safety_telemetry_and_build_failures_are_replayed_from_raw(self):
        mutations = ("footer_nonzero", "footer_execution_mismatch", "footer_tail",
                     "execution_error", "safety_error", "safety_snapshot",
                     "telemetry_throttle", "telemetry_summary", "build_commit",
                     "build_number")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                cell = fixture.state["timing_cells"][0]
                directory = fixture.root / cell["directory"]
                log_path = directory / "llama_bench.log"
                execution_path = directory / "llama_bench.execution.json"
                safety_path = directory / "gpu-safety.json"
                telemetry_path = directory / "gpu-telemetry.csv"
                if mutation == "footer_nonzero":
                    log_path.write_text(log_path.read_text().replace("# exit: 0\n", "# exit: -9\n"))
                elif mutation == "footer_execution_mismatch":
                    execution = json.loads(execution_path.read_text())
                    execution["returncode"] = -9
                    execution_path.write_text(json.dumps(execution))
                elif mutation == "footer_tail":
                    log_path.write_text(log_path.read_text() + "trailing\n")
                elif mutation == "execution_error":
                    execution = json.loads(execution_path.read_text())
                    execution["error"] = "injected"
                    execution_path.write_text(json.dumps(execution))
                elif mutation in {"safety_error", "safety_snapshot", "telemetry_summary"}:
                    safety = json.loads(safety_path.read_text())
                    if mutation == "safety_error":
                        safety["cleanup_errors"] = []
                    elif mutation == "safety_snapshot":
                        safety["after"]["uvm_refcount"] = 1
                    else:
                        safety["telemetry"]["mean_power_w"] = 24.0
                    safety_path.write_text(json.dumps(safety))
                elif mutation == "telemetry_throttle":
                    telemetry_path.write_text(telemetry_path.read_text().replace(
                        ", Not Active, Not Active, Not Active, Not Active, Not Active\n",
                        ", Not Active, Active, Not Active, Not Active, Not Active\n", 1))
                else:
                    old = "fixture-build" if mutation == "build_commit" else '"build_number": 7102'
                    new = "other-build" if mutation == "build_commit" else '"build_number": 7103'
                    log_path.write_text(log_path.read_text().replace(old, new, 1))
                result = analyzer.analyze(fixture.root)
                self.assertFalse(result["complete"], result)
                if mutation.startswith("build_"):
                    self.assertFalse(result["llama_bench_build"]["consistent"])


if __name__ == "__main__":
    unittest.main()
