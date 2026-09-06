#!/usr/bin/env python3
"""Unittest for run_table1_perf.py: seven-arm rotation schedule and dry-run only."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


table1 = load_module("run_table1_perf", "run_table1_perf.py")


class Table1RotationAndDryRunTests(unittest.TestCase):
    def test_seven_arms_rotate_every_block(self):
        arms = [
            "baseline",
            "gpubpf_kernelretsnoop",
            "nvbit_kernelretsnoop",
            "gpubpf_threadhist",
            "nvbit_threadhist",
            "gpubpf_launchlate",
            "nvbit_launchlate",
        ]
        self.assertEqual(list(table1.ARMS), arms)
        schedule = table1.build_schedule(8)
        self.assertEqual(len(schedule), 8)
        for block in range(1, 9):
            offset = (block - 1) % len(arms)
            self.assertEqual(schedule[str(block)], arms[offset:] + arms[:offset])
        for order in schedule.values():
            self.assertEqual(sorted(order), sorted(arms))
        self.assertEqual(schedule["1"], arms)
        self.assertEqual(schedule["8"], schedule["1"])
        self.assertEqual(table1.build_schedule(1)["1"], arms)
        self.assertEqual(len(table1.build_schedule(10)["10"]), 7)

    def test_dry_run_prints_schedule_and_does_no_build_or_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "table1-dry"
            forbidden = [
                mock.patch.object(
                    table1, "run_campaign",
                    side_effect=AssertionError("dry-run must not run the campaign"),
                ),
                mock.patch.object(
                    table1, "build_tools",
                    side_effect=AssertionError("dry-run must not build tools"),
                ),
                mock.patch.object(
                    table1, "run_arm_cell",
                    side_effect=AssertionError("dry-run must not run cells"),
                ),
                mock.patch.object(
                    table1, "write_records",
                    side_effect=AssertionError("dry-run must not write records"),
                ),
                mock.patch.object(
                    table1.core, "prepare_tool_source",
                    side_effect=AssertionError("dry-run must not prepare tool sources"),
                ),
                mock.patch.object(
                    table1.core, "build_tool",
                    side_effect=AssertionError("dry-run must not build gpubpf tools"),
                ),
                mock.patch.object(
                    table1.runner, "build_nvbit",
                    side_effect=AssertionError("dry-run must not build NVBit"),
                ),
                mock.patch.object(
                    table1.runner, "private_probe",
                    side_effect=AssertionError("dry-run must not start probes"),
                ),
                mock.patch.object(
                    table1.runner, "run_bench",
                    side_effect=AssertionError("dry-run must not run benches"),
                ),
                mock.patch.object(
                    table1.core, "nvidia_smi_snapshot",
                    side_effect=AssertionError("dry-run must not query the GPU"),
                ),
                mock.patch.object(
                    table1.subprocess, "run",
                    side_effect=AssertionError("dry-run must not launch processes"),
                ),
                mock.patch.object(
                    table1.subprocess, "Popen",
                    side_effect=AssertionError("dry-run must not launch processes"),
                ),
                mock.patch.object(
                    table1.shutil, "copytree",
                    side_effect=AssertionError("dry-run must not copy tool sources"),
                ),
            ]
            with contextlib.ExitStack() as stack:
                for patch in forbidden:
                    stack.enter_context(patch)
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    code = table1.main(
                        ["--dry-run", "--blocks", "3", "--output-dir", str(out_dir)]
                    )
            plan = json.loads(buffer.getvalue())
        self.assertEqual(code, 0)
        self.assertIs(plan["dry_run"], True)
        self.assertEqual(plan["blocks"], 3)
        self.assertEqual(plan["pp"], 512)
        self.assertEqual(plan["tg"], 0)
        self.assertEqual(plan["arms"], list(table1.ARMS))
        self.assertEqual(plan["cell_count"], 21)
        self.assertEqual(plan["schedule"], table1.build_schedule(3))
        self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
