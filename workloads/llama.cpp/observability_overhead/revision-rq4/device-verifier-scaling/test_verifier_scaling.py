#!/usr/bin/env python3
"""CPU-only tests for the verifier-scaling harness and independent replay."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import analyze_verifier_scaling as analyzer
import run_verifier_scaling as runner


REVISION = "git-test-revision"
PROBE = "/independent/build/verifier_scaling_probe"
BPFTIME_ROOT = "/independent/source/bpftime"
PROBE_SIZE = 123456
PROBE_MTIME_NS = 987654321


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def probe_record(family: str, size: int, mode: str, block: int = 0) -> dict:
    record = runner.expected_shape(family, size, mode)
    record.update(
        {
            "bpftime_source_revision": REVISION,
            "build_type": "Release",
            "accepted": None if mode == "describe" else True,
            "error": "",
            "elapsed_ns": None,
            "process_cpu_ns": None,
            "cpu_before": None,
            "cpu_after": None,
            "minor_faults": None,
            "major_faults": None,
            "voluntary_context_switches": None,
            "involuntary_context_switches": None,
        }
    )
    if mode == "timed":
        multiplier = 1000 if family == "linear" else 1500
        elapsed = size * multiplier + block
        record.update(
            {
                "elapsed_ns": elapsed,
                "process_cpu_ns": elapsed - 10,
                "cpu_before": runner.CPU,
                "cpu_after": runner.CPU,
                "minor_faults": 0,
                "major_faults": 0,
                "voluntary_context_switches": 0,
                "involuntary_context_switches": 0,
            }
        )
    return record


def raw_call(
    root: Path,
    relative: str,
    family: str,
    size: int,
    mode: str,
    block: int = 0,
) -> None:
    directory = root / relative
    directory.mkdir(parents=True)
    record = probe_record(family, size, mode, block)
    (directory / "stdout.log").write_text(json.dumps(record, sort_keys=True) + "\n")
    (directory / "stderr.log").write_text("")
    execution = {
        "argv": analyzer.expected_argv(PROBE, family, size, mode),
        "cwd": str(directory),
        "started_utc": "2026-09-04T00:00:00+00:00",
        "duration_ns": (record["elapsed_ns"] + 1000) if mode == "timed" else 1000,
        "timeout_seconds": analyzer.TIMEOUT_SECONDS,
        "timed_out": False,
        "returncode": 0,
        "environment": {"CUDA_VISIBLE_DEVICES": "", "LD_PRELOAD": None},
    }
    write_json(directory / "execution.json", execution)


def make_full_fixture(root: Path) -> None:
    cpufreq = {
        "driver": "test-driver",
        "governor": "performance",
        "energy_performance_preference": "performance",
    }
    result = {
        "schema": runner.RESULT_SCHEMA,
        "status": "complete",
        "mode": "full",
        "seed": runner.SEED,
        "cpu": runner.CPU,
        "blocks": runner.BLOCKS,
        "sizes": list(runner.SIZES),
        "families": list(runner.FAMILIES),
        "timeout_seconds": runner.TIMEOUT_SECONDS,
        "environment": {
            "runner_affinity": [runner.CPU],
            "cuda_visible_devices": "",
            "ld_preload": None,
            "cpufreq": cpufreq,
            "probe": {
                "path": PROBE,
                "size": PROBE_SIZE,
                "mtime_ns": PROBE_MTIME_NS,
                "cmake_build_type": "Release",
                "cmake_bpftime_root": BPFTIME_ROOT,
            },
            "bpftime_root": BPFTIME_ROOT,
            "bpftime_current_revision": REVISION,
            "bpftime_verifier_status": [],
        },
        "end_environment": {
            "runner_affinity": [runner.CPU],
            "cpufreq": cpufreq,
            "probe": {
                "path": PROBE,
                "size": PROBE_SIZE,
                "mtime_ns": PROBE_MTIME_NS,
            },
            "bpftime_current_revision": REVISION,
            "bpftime_verifier_status": [],
        },
        "probe_source_revision": REVISION,
        "descriptions": [],
        "warmups": [],
        "cells": [],
        "error": None,
    }
    for family, size in runner.ARMS:
        relative = f"descriptions/{family}-{size}"
        raw_call(root, relative, family, size, "describe")
        result["descriptions"].append(
            {"family": family, "instructions": size, "directory": relative}
        )
    for family, size in runner.ARMS:
        relative = f"warmups/{family}-{size}"
        raw_call(root, relative, family, size, "accept_only")
        result["warmups"].append(
            {"family": family, "instructions": size, "directory": relative}
        )
    for item in runner.frozen_schedule():
        relative = (
            f"cells/seq-{item['sequence']:03d}-block-{item['block']:02d}-"
            f"pos-{item['position']:02d}-{item['family']}-{item['instructions']}"
        )
        raw_call(
            root,
            relative,
            item["family"],
            item["instructions"],
            "timed",
            item["block"],
        )
        cell = copy.deepcopy(item)
        cell.update({"directory": relative, "valid": True})
        result["cells"].append(cell)
    write_json(root / "result.json", result)


def make_preflight_fixture(root: Path) -> None:
    cpufreq = {
        "driver": "test-driver",
        "governor": "performance",
        "energy_performance_preference": "performance",
    }
    result = {
        "schema": runner.RESULT_SCHEMA,
        "status": "complete",
        "mode": "preflight",
        "seed": runner.SEED,
        "cpu": runner.CPU,
        "blocks": 1,
        "sizes": list(runner.SIZES),
        "families": list(runner.FAMILIES),
        "timeout_seconds": runner.TIMEOUT_SECONDS,
        "environment": {
            "runner_affinity": [runner.CPU],
            "cuda_visible_devices": "",
            "ld_preload": None,
            "cpufreq": cpufreq,
            "probe": {
                "path": PROBE,
                "size": PROBE_SIZE,
                "mtime_ns": PROBE_MTIME_NS,
                "cmake_build_type": "Release",
                "cmake_bpftime_root": BPFTIME_ROOT,
            },
            "bpftime_root": BPFTIME_ROOT,
            "bpftime_current_revision": REVISION,
            "bpftime_verifier_status": [],
        },
        "end_environment": {
            "runner_affinity": [runner.CPU],
            "cpufreq": cpufreq,
            "probe": {
                "path": PROBE,
                "size": PROBE_SIZE,
                "mtime_ns": PROBE_MTIME_NS,
            },
            "bpftime_current_revision": REVISION,
            "bpftime_verifier_status": [],
        },
        "probe_source_revision": REVISION,
        "descriptions": [],
        "warmups": [],
        "cells": [],
        "error": None,
    }
    for family, size in (("linear", 16), ("diamonds", 4096)):
        relative = f"descriptions/{family}-{size}"
        raw_call(root, relative, family, size, "describe")
        result["descriptions"].append(
            {"family": family, "instructions": size, "directory": relative}
        )
    for item in runner.preflight_schedule():
        relative = (
            f"cells/seq-{item['sequence']:03d}-block-{item['block']:02d}-"
            f"pos-{item['position']:02d}-{item['family']}-{item['instructions']}"
        )
        raw_call(root, relative, item["family"], item["instructions"], "timed", 1)
        cell = copy.deepcopy(item)
        cell.update({"directory": relative, "valid": True})
        result["cells"].append(cell)
    write_json(root / "result.json", result)


class ScheduleTests(unittest.TestCase):
    def test_shape_formulas(self) -> None:
        for size, branches in zip(runner.SIZES, (6, 30, 126, 510, 2046), strict=True):
            self.assertEqual(
                runner.expected_shape("diamonds", size, "describe")[
                    "conditional_branches"
                ],
                branches,
            )
            self.assertEqual(
                runner.expected_shape("linear", size, "describe")[
                    "conditional_branches"
                ],
                0,
            )

    def test_schedule_is_complete_and_balanced(self) -> None:
        schedule = runner.frozen_schedule()
        self.assertEqual(len(schedule), 200)
        self.assertEqual(schedule, analyzer.full_schedule())
        for block in range(1, runner.BLOCKS + 1):
            arms = {
                (item["family"], item["instructions"])
                for item in schedule
                if item["block"] == block
            }
            self.assertEqual(arms, set(runner.ARMS))

    def test_nonfrozen_size_and_block_count_fail(self) -> None:
        with self.assertRaises(ValueError):
            runner.expected_shape("linear", 17, "describe")
        with self.assertRaises(ValueError):
            runner.frozen_schedule(19)


class ProbeDescribeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        value = os.environ.get("VERIFIER_SCALING_PROBE")
        cls.probe = Path(value) if value else None

    def test_live_descriptions_when_probe_is_supplied(self) -> None:
        if self.probe is None:
            self.skipTest("VERIFIER_SCALING_PROBE not supplied")
        for family, size in runner.ARMS:
            completed = subprocess.run(
                [
                    str(self.probe),
                    "--describe",
                    "--family",
                    family,
                    "--instructions",
                    str(size),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.stderr, "")
            record = runner.read_one_json_line(completed.stdout)
            runner.validate_probe_record(record, family, size, "describe")

    def test_live_probe_rejects_nonfrozen_size_when_supplied(self) -> None:
        if self.probe is None:
            self.skipTest("VERIFIER_SCALING_PROBE not supplied")
        completed = subprocess.run(
            [
                str(self.probe),
                "--describe",
                "--family",
                "linear",
                "--instructions",
                "17",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 64)
        self.assertIn("outside frozen set", completed.stderr)


class AnalyzerTests(unittest.TestCase):
    def with_fixture(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name).resolve()
        make_full_fixture(root)
        return temporary, root

    def test_complete_fixture_is_valid(self) -> None:
        temporary, root = self.with_fixture()
        self.addCleanup(temporary.cleanup)
        result = analyzer.analyze(root)
        self.assertTrue(result["complete"], result["errors"])
        self.assertEqual(result["run_status"], "valid")
        self.assertEqual(result["summary"]["tested_hypothesis"], "supported")
        self.assertAlmostEqual(
            result["summary"]["diamonds_over_linear"]["4096"]["median_ratio"],
            1.5,
            places=3,
        )

    def test_preflight_is_valid_but_dependency_only(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        make_preflight_fixture(root)
        result = analyzer.analyze(root)
        self.assertTrue(result["complete"], result["errors"])
        self.assertIsNone(result["summary"])
        self.assertEqual(result["research_value"], "dependency-only")

    def test_fail_closed_mutations(self) -> None:
        mutations = {
            "status": lambda root, result: result.update(status="invalid"),
            "dirty_source": lambda root, result: result["environment"].update(
                bpftime_verifier_status=[" M bpftime-verifier/source.cpp"]
            ),
            "cpufreq": lambda root, result: result["end_environment"][
                "cpufreq"
            ].update(governor="powersave"),
            "schedule": lambda root, result: result["cells"][0].update(position=2),
            "missing_cell": lambda root, result: result["cells"].pop(),
            "missing_warmup": lambda root, result: result["warmups"].pop(),
            "revision": lambda root, result: result.update(
                probe_source_revision="different-revision"
            ),
            "end_revision": lambda root, result: result["end_environment"].update(
                bpftime_current_revision="different-revision"
            ),
            "probe_replaced": lambda root, result: result["end_environment"][
                "probe"
            ].update(mtime_ns=PROBE_MTIME_NS + 1),
            "stderr": lambda root, result: (
                root / result["cells"][0]["directory"] / "stderr.log"
            ).write_text("unexpected\n"),
            "argv": lambda root, result: _mutate_json(
                root / result["cells"][0]["directory"] / "execution.json",
                lambda value: value.update(argv=["wrong"]),
            ),
            "timeout": lambda root, result: _mutate_json(
                root / result["cells"][0]["directory"] / "execution.json",
                lambda value: value.update(timed_out=True, returncode=None),
            ),
            "elapsed": lambda root, result: _mutate_stdout(
                root / result["cells"][0]["directory"] / "stdout.log",
                lambda value: value.update(elapsed_ns=0),
            ),
            "cpu": lambda root, result: _mutate_stdout(
                root / result["cells"][0]["directory"] / "stdout.log",
                lambda value: value.update(cpu_after=22),
            ),
            "acceptance": lambda root, result: _mutate_stdout(
                root / result["cells"][0]["directory"] / "stdout.log",
                lambda value: value.update(accepted=False, error="rejected"),
            ),
            "branch_count": lambda root, result: _mutate_stdout(
                root / result["cells"][0]["directory"] / "stdout.log",
                lambda value: value.update(conditional_branches=999),
            ),
            "boolean_structural_count": lambda root, result: _mutate_stdout(
                root / result["cells"][0]["directory"] / "stdout.log",
                lambda value: value.update(helper_calls=True),
            ),
            "boolean_elapsed": lambda root, result: _mutate_stdout(
                root / result["cells"][0]["directory"] / "stdout.log",
                lambda value: value.update(elapsed_ns=True),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                temporary, root = self.with_fixture()
                try:
                    result = json.loads((root / "result.json").read_text())
                    mutate(root, result)
                    write_json(root / "result.json", result)
                    analysis = analyzer.analyze(root)
                    self.assertFalse(analysis["complete"], analysis)
                    self.assertTrue(analysis["errors"])
                finally:
                    temporary.cleanup()

    def test_non_object_result_fails_closed(self) -> None:
        temporary, root = self.with_fixture()
        self.addCleanup(temporary.cleanup)
        write_json(root / "result.json", [])
        analysis = analyzer.analyze(root)
        self.assertFalse(analysis["complete"], analysis)
        self.assertEqual(analysis["errors"], ["result JSON is not an object"])

    def test_non_object_execution_fails_closed(self) -> None:
        temporary, root = self.with_fixture()
        self.addCleanup(temporary.cleanup)
        result = json.loads((root / "result.json").read_text())
        execution = root / result["cells"][0]["directory"] / "execution.json"
        write_json(execution, [])
        analysis = analyzer.analyze(root)
        self.assertFalse(analysis["complete"], analysis)
        self.assertTrue(
            any("execution JSON is not an object" in error for error in analysis["errors"]),
            analysis,
        )

    def test_non_object_probe_metadata_fails_closed(self) -> None:
        temporary, root = self.with_fixture()
        self.addCleanup(temporary.cleanup)
        result = json.loads((root / "result.json").read_text())
        result["environment"]["probe"] = []
        write_json(root / "result.json", result)
        analysis = analyzer.analyze(root)
        self.assertFalse(analysis["complete"], analysis)
        self.assertIn("probe metadata is not an object", analysis["errors"])

    def test_major_fault_is_noise_veto_not_row_deletion(self) -> None:
        temporary, root = self.with_fixture()
        self.addCleanup(temporary.cleanup)
        result = json.loads((root / "result.json").read_text())
        _mutate_stdout(
            root / result["cells"][0]["directory"] / "stdout.log",
            lambda value: value.update(major_faults=1),
        )
        analysis = analyzer.analyze(root)
        self.assertTrue(analysis["complete"], analysis["errors"])
        self.assertTrue(analysis["summary"]["diagnostics"]["noise_veto"])
        self.assertEqual(analysis["summary"]["tested_hypothesis"], "inconclusive")


def _mutate_json(path: Path, mutation) -> None:
    value = json.loads(path.read_text())
    mutation(value)
    write_json(path, value)


def _mutate_stdout(path: Path, mutation) -> None:
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
