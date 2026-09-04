#!/usr/bin/env python3
"""CPU-only fixtures for the S0 runner and independent analyzer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
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
    row = [{"model_filename": str(model), "n_gpu_layers": 99, "n_prompt": pp,
            "n_gen": 0, "avg_ns": int(pp / throughput * 1e9), "avg_ts": throughput,
            "samples_ns": [int(pp / throughput * 1e9)], "samples_ts": [throughput]}]
    return json.dumps(row, indent=2)


def target_log(pp: int, model: Path, throughput: float, records: str) -> str:
    return ("$ llama-bench\n# cwd: fixture\n\n## stdout\n"
            + bench_json(pp, model, throughput) + "\n\n## stderr\n"
            + records + "\n# exit: 0\n")


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
            execution.write_text(json.dumps({"identity": {"pid": pid}, "cleanup_passed": True,
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
            "host": {"driver": analyzer.EXPECTED_DRIVER, "expected_driver": analyzer.EXPECTED_DRIVER},
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
        (directory / "llama_bench.execution.json").write_text(json.dumps({
            "identity": {"pid": self.pid}, "cleanup_passed": True, "timed_out": False,
            "returncode": 0, "command": [str(self.bench), "-m", str(self.model),
                                          "-p", str(specification["pp"])]}))
        (directory / "gpu-safety.json").write_text(json.dumps({"passed": True}))
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
        artifacts = {"agent": self.state["runtime"]["agent"],
                     "syscall_server": self.state["runtime"]["syscall_server"],
                     "object": self.state["objects"][specification["tool"]]}
        return {"stage": stage, **specification, "directory": relative, "valid": True,
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
            for tool in runner.TOOLS:
                summary = result["summary"][tool]
                self.assertEqual(summary["complete_blocks"], 10)
                self.assertAlmostEqual(summary["strict_vs_no_verify"]["mean_percent"], -1.0)
                self.assertAlmostEqual(summary["no_verify_vs_control"]["mean_percent"], -10.0)

    def test_missing_extra_and_duplicate_cells_are_rejected(self):
        for mutation, expected_status in (("missing", "incomplete"), ("extra", "invalid"),
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

    def test_reused_private_shm_and_raw_throughput_tampering_fail(self):
        for mutation in ("shm", "throughput", "event_count"):
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
                    if mutation == "throughput":
                        path = fixture.root / fixture.state["timing_cells"][0]["directory"] / "llama_bench.log"
                        path.write_text(path.read_text().replace('"avg_ts":', '"wrong_avg_ts":', 1))
                    else:
                        cell = next(cell for cell in fixture.state["timing_cells"]
                                    if cell["tool"] == "threadhist" and cell["treatment"] == "STRICT")
                        path = fixture.root / cell["directory"] / "probe.log"
                        path.write_text(path.read_text().replace(
                            "Total exit probes: 720896", "Total exit probes: 720897"))
                self.assertFalse(analyzer.analyze(fixture.root)["complete"])


if __name__ == "__main__":
    unittest.main()
