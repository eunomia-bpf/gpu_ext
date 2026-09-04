#!/usr/bin/env python3
"""Offline tests for the A1 runner and independent analyzer; no GPU access."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("device_a1_runner", "run_device_verifier_a1.py")
analyzer = load("device_a1_analyzer", "analyze_device_verifier_a1.py")
TARGET = "target_kernel"


def prefix(pid: int) -> str:
    return f"[stamp][info][{pid}] "


def admission_lines(tool: str, mode: str, pid: int, elapsed: int) -> str:
    if mode == "NO_VERIFY":
        return prefix(pid) + "Skipping GPU eBPF verification for cuda__retprobe\n"
    expected = runner.expected_map(tool)
    return "".join(
        (
            prefix(pid)
            + f"GPU eBPF verification timing: program=cuda__retprobe verification_elapsed_ns={elapsed}\n",
            prefix(pid)
            + "GPU eBPF verification accepted: mode=STRICT program=cuda__retprobe "
            + f"attach=kretprobe/{TARGET} instructions=17\n",
            prefix(pid)
            + "GPU eBPF verified map: program=cuda__retprobe fd=0 "
            + f"type={expected['type']} key_size={expected['key_size']} "
            + f"value_size={expected['value_size']} max_entries={expected['max_entries']}\n",
        )
    )


def target_log(lines: str) -> str:
    return (
        "$ llama-cli\n# cwd: fixture\n\n## stdout\n"
        + analyzer.EXPECTED_OUTPUT
        + "\n## stderr\n"
        + lines
        + "# exit: 0\n"
    )


def kernel_probe() -> str:
    values = (
        ("Total events collected", 720896), ("Nonzero timestamps", 720896),
        ("Requested thread slots", 22528), ("Allocated thread slots", 22528),
        ("Ring entries per thread", 256), ("Requested ring entries per thread", 256),
        ("Record bytes", 32), ("Committed events", 720896),
        ("Runtime collected events", 720896), ("OOB drops", 0), ("Full drops", 0),
        ("Bad-size drops", 0), ("Other drops", 0), ("Dirty slots", 0),
        ("Pending events", 0), ("Final drain events", 1), ("Second drain events", 0),
        ("Cartesian launches", 220), ("Cartesian coordinates", 22528),
        ("Cartesian complete", 1), ("Coordinate extent x", 88),
        ("Coordinate extent y", 256), ("Coordinate extent z", 1),
        ("Coordinate multiplicity 220", 1024), ("Coordinate multiplicity 44", 1024),
        ("Coordinate multiplicity 22", 20480), ("Coordinate multiplicity other", 0),
        ("Coordinate segment mismatches", 0), ("Invalid launch coordinates", 0),
        ("Unique coordinates", 22528), ("Multiplicity oracle enabled", 1),
        ("Multiplicity oracle total events", 720896), ("Multiplicity oracle passed", 1),
        ("Collector gate passed", 1),
    )
    return "".join(f"{label}: {value}\n" for label, value in values)


def thread_probe() -> str:
    return (
        "Configured thread entries: 1048576\nReadback entries: 1048576\n"
        "Readback bytes: 8388608\nReadback complete: 1\n"
        "Nonzero threads: 22528\nTotal exit probes: 720896\n"
    )


class AdmissionParserTests(unittest.TestCase):
    def parse(self, tool: str, mode: str, lines: str, execution_pid: int = 41):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "llama_cli.log"
            execution = root / "llama_cli.execution.json"
            log.write_text(target_log(lines), encoding="utf-8")
            execution.write_text(json.dumps({
                "identity": {"pid": execution_pid}, "cleanup_passed": True,
                "timed_out": False, "returncode": 0,
            }), encoding="utf-8")
            return runner.parse_target_admission(
                log, execution, tool=tool, mode=mode, target_symbol=TARGET
            )

    def test_strict_requires_unique_target_timing_accept_and_exact_map(self):
        result = self.parse("kernelretsnoop", "STRICT", admission_lines("kernelretsnoop", "STRICT", 41, 123))
        self.assertTrue(result["passed"])
        self.assertEqual(result["verification_elapsed_ns"], 123)
        self.assertEqual(result["latency_source"], "target_llama_cli_log_runtime_marker")

        duplicate = admission_lines("kernelretsnoop", "STRICT", 41, 123)
        duplicate += prefix(41) + "GPU eBPF verification timing: program=cuda__retprobe verification_elapsed_ns=124\n"
        self.assertFalse(self.parse("kernelretsnoop", "STRICT", duplicate)["passed"])

    def test_no_verify_requires_one_skip_and_forbids_timing(self):
        lines = admission_lines("threadhist", "NO_VERIFY", 41, 0)
        self.assertTrue(self.parse("threadhist", "NO_VERIFY", lines)["passed"])
        lines += prefix(41) + "GPU eBPF verification timing: program=cuda__retprobe verification_elapsed_ns=1\n"
        self.assertFalse(self.parse("threadhist", "NO_VERIFY", lines)["passed"])

    def test_foreign_pid_wrong_attach_map_and_unparsed_marker_fail_closed(self):
        base = admission_lines("kernelretsnoop", "STRICT", 41, 5)
        foreign = base + admission_lines("kernelretsnoop", "NO_VERIFY", 99, 0)
        self.assertFalse(self.parse("kernelretsnoop", "STRICT", foreign)["passed"])
        self.assertFalse(self.parse("kernelretsnoop", "STRICT", base.replace(TARGET, "wrong"))["passed"])
        self.assertFalse(self.parse("kernelretsnoop", "STRICT", base.replace("type=1527", "type=1502"))["passed"])
        malformed = base + prefix(41) + "GPU eBPF verification timing: malformed\n"
        self.assertFalse(self.parse("kernelretsnoop", "STRICT", malformed)["passed"])


class ScheduleTests(unittest.TestCase):
    def test_fixed_schedule_is_reproducible_balanced_and_complete(self):
        first = runner.schedule(10)
        self.assertEqual(first, runner.schedule(10))
        self.assertEqual(first, analyzer.fixed_schedule(10))
        self.assertEqual(len(first), 40)
        for tool in runner.TOOLS:
            strict_first = 0
            for pair in range(1, 11):
                cells = [cell for cell in first if cell["tool"] == tool and cell["pair"] == pair]
                self.assertEqual({cell["mode"] for cell in cells}, set(runner.MODES))
                strict_first += next(c["sequence"] for c in cells if c["mode"] == "STRICT") < next(
                    c["sequence"] for c in cells if c["mode"] == "NO_VERIFY"
                )
            self.assertEqual(strict_first, 5)

    def test_pair_floor_is_fail_closed(self):
        with self.assertRaises(ValueError):
            runner.schedule(9)
        with self.assertRaises(ValueError):
            analyzer.fixed_schedule(9)


class AnalyzerFixture:
    def __init__(self, root: Path):
        self.root = root
        self.pid = 1000
        self.plan = runner.plan(types.SimpleNamespace(pairs=10))
        self.state: dict = {
            "schema": runner.SCHEMA, "status": "complete", "plan": self.plan,
            "defining_inputs": {
                "target_symbol": TARGET, "bpftime_build_dir": str(self.root / "build"),
                "llama_cli": str(self.root / "llama-cli"), "model": str(self.root / "model.gguf"),
            },
            "host": {"driver": analyzer.EXPECTED_DRIVER,
                     "expected_driver": analyzer.EXPECTED_DRIVER, "boot_id": "fixture"},
            "runtime": {
                "build_configuration": {key: "ON" for key in analyzer.BUILD_KEYS},
                "source_contract": {"passed": True},
                "agent": {"exists": True, "bytes": 100},
                "syscall_server": {"exists": True, "bytes": 100},
            },
            "objects": {tool: {"exists": True, "bytes": 100} for tool in runner.TOOLS},
            "baseline": self.make_baseline(), "a0": [], "cells": [],
        }
        for sequence, tool in enumerate(runner.TOOLS, 1):
            self.state["a0"].append(self.make_cell(
                {"stage": "a0", "sequence": sequence, "pair": None,
                 "tool": tool, "mode": "STRICT"}, elapsed=50 + sequence,
            ))
        for specification in self.plan["schedule"]:
            self.state["cells"].append(self.make_cell(
                {"stage": "a1", **specification}, elapsed=100 + specification["sequence"],
            ))
        self.write()

    def make_baseline(self) -> dict:
        directory = self.root / "correctness-baseline"
        directory.mkdir()
        self.pid += 1
        (directory / "llama_cli.log").write_text(target_log(""), encoding="utf-8")
        (directory / "llama_cli.execution.json").write_text(json.dumps({
            "identity": {"pid": self.pid}, "cleanup_passed": True,
            "timed_out": False, "returncode": 0,
            "command": [str(self.root / "llama-cli"), "-m", str(self.root / "model.gguf")],
        }), encoding="utf-8")
        (directory / "gpu-safety.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
        return {"valid": True, "directory": "correctness-baseline"}

    def make_cell(self, specification: dict, elapsed: int) -> dict:
        self.pid += 1
        stage = specification["stage"]
        relative = f"{stage}/{specification['sequence']:03d}-{specification['tool']}-{specification['mode'].lower()}"
        directory = self.root / relative
        directory.mkdir(parents=True)
        lines = admission_lines(specification["tool"], specification["mode"], self.pid, elapsed)
        (directory / "llama_cli.log").write_text(target_log(lines), encoding="utf-8")
        (directory / "llama_cli.execution.json").write_text(json.dumps({
            "identity": {"pid": self.pid}, "cleanup_passed": True,
            "timed_out": False, "returncode": 0,
            "command": [str(self.root / "llama-cli"), "-m", str(self.root / "model.gguf")],
        }), encoding="utf-8")
        (directory / "gpu-safety.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
        (directory / "probe-execution.json").write_text(json.dumps({
            "private_segment": f"rq4_fixture_{self.pid}",
            "private_segment_removed": True,
            "agent_environment": {
                "BPFTIME_GLOBAL_SHM_NAME": f"rq4_fixture_{self.pid}",
                "BPFTIME_VERIFIER_LEVEL": specification["mode"], "SPDLOG_LEVEL": "info",
                "LD_PRELOAD": str(self.root / "build/runtime/agent/libbpftime-agent.so"),
            },
            "loader_environment": {
                "BPFTIME_GLOBAL_SHM_NAME": f"rq4_fixture_{self.pid}",
                "BPFTIME_VERIFIER_LEVEL": specification["mode"], "SPDLOG_LEVEL": "info",
                "LD_PRELOAD": str(self.root / "build/runtime/syscall-server/libbpftime-syscall-server.so"),
            },
        }), encoding="utf-8")
        probe = kernel_probe() if specification["tool"] == "kernelretsnoop" else thread_probe()
        (directory / "probe.log").write_text(probe, encoding="utf-8")
        return {**specification, "directory": relative, "valid": True}

    def write(self) -> None:
        (self.root / "result.json").write_text(json.dumps(self.state), encoding="utf-8")


class AnalyzerTests(unittest.TestCase):
    def test_complete_fixture_recomputes_latency_without_application_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnalyzerFixture(Path(temporary))
            result = analyzer.analyze(fixture.root)
            self.assertTrue(result["complete"], result["errors"])
            self.assertFalse(result["application_latency_or_throughput_used"])
            for tool in runner.TOOLS:
                self.assertEqual(result["summary"][tool]["complete_pairs"], 10)
                self.assertEqual(len(result["summary"][tool]["strict_verification_elapsed_ns"]), 10)

    def test_analyzer_rejects_pid_mismatch_no_verify_timing_and_reused_shm(self):
        mutations = ("pid", "timing", "shm")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = AnalyzerFixture(Path(temporary))
                first = fixture.state["cells"][0]
                directory = fixture.root / first["directory"]
                if mutation == "pid":
                    execution = json.loads((directory / "llama_cli.execution.json").read_text())
                    execution["identity"]["pid"] += 999
                    (directory / "llama_cli.execution.json").write_text(json.dumps(execution))
                elif mutation == "timing":
                    skip = next(cell for cell in fixture.state["cells"] if cell["mode"] == "NO_VERIFY")
                    directory = fixture.root / skip["directory"]
                    text = (directory / "llama_cli.log").read_text()
                    pid = json.loads((directory / "llama_cli.execution.json").read_text())["identity"]["pid"]
                    text = text.replace("# exit: 0", prefix(pid) +
                        "GPU eBPF verification timing: program=cuda__retprobe verification_elapsed_ns=7\n# exit: 0")
                    (directory / "llama_cli.log").write_text(text)
                else:
                    second = fixture.state["cells"][1]
                    first_probe = json.loads((fixture.root / first["directory"] / "probe-execution.json").read_text())
                    path = fixture.root / second["directory"] / "probe-execution.json"
                    second_probe = json.loads(path.read_text())
                    second_probe["private_segment"] = first_probe["private_segment"]
                    path.write_text(json.dumps(second_probe))
                result = analyzer.analyze(fixture.root)
                self.assertFalse(result["complete"])
                self.assertEqual(result["run_status"], "invalid")

    def test_missing_cell_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnalyzerFixture(Path(temporary))
            fixture.state["cells"].pop()
            fixture.write()
            result = analyzer.analyze(fixture.root)
            self.assertFalse(result["complete"])
            self.assertEqual(result["run_status"], "incomplete")

    def test_extra_cell_is_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnalyzerFixture(Path(temporary))
            extra = copy.deepcopy(fixture.state["cells"][-1])
            extra["sequence"] = 41
            fixture.state["cells"].append(extra)
            fixture.write()
            result = analyzer.analyze(fixture.root)
            self.assertFalse(result["complete"])
            self.assertIn(
                "A1 cell cardinality differs from the fixed schedule",
                result["errors"],
            )


if __name__ == "__main__":
    unittest.main()
