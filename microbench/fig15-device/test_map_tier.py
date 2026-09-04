#!/usr/bin/env python3
"""CPU-only schedule and raw-replay tests for the device-map experiment."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import analyze_map_tier as analyzer
import run_map_tier as runner


ARM_US = {
    "native": 10.0,
    "noop": 11.0,
    "device_update": 12.0,
    "host_update": 24.0,
    "rpc_update": 1000.0,
    "device_lookup": 14.0,
    "host_lookup": 28.0,
    "rpc_lookup": 1100.0,
}


def write_fixture(root: Path, phase: str = "full") -> None:
    root.mkdir()
    schedule = analyzer.expected_schedule(phase)
    with (root / "schedule.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("block", "order", "arm", "run_id"), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(schedule)
    (root / "environment.txt").write_text(
        "gpu\tNVIDIA GeForce RTX 5090\n"
        "driver\t575.57.08\n"
        "bpftime_root\t/test/bpftime\n"
        "bpftime_revision\ttest-revision\n"
        "BPFTIME_ENABLE_CUDA_ATTACH\tON\n"
        "BPFTIME_LLVM_JIT\tON\n"
        "ENABLE_EBPF_VERIFIER\tOFF\n"
        "bpftime_status_begin\n"
        "bpftime_status_end\n"
        "nvcc_begin\n"
        "Cuda compilation tools, release 12.9\n"
        "nvcc_end\n",
        encoding="utf-8",
    )
    _blocks, warmup, launches = analyzer.phase_parameters(phase)
    for item in schedule:
        arm = str(item["arm"])
        directory = root / (
            f"block-{int(item['block']):02d}-order-{int(item['order']):02d}-{arm}"
        )
        directory.mkdir()
        jitter = 1.0 + 0.001 * ((int(item["block"]) % 3) - 1)
        elapsed_ms = ARM_US[arm] * jitter * launches / 1000.0
        attached = arm != "native"
        prefix = ""
        if attached:
            prefix = (
                "[ptxpass] kprobe_entry_stub: matched=1, in=1, out=2\n"
                "Loaded module: patched.map-bench.ptx\n"
                "Attach successfully\n"
            )
        (directory / "application.log").write_text(
            prefix
            + "FIG15_DEVICE\tNVIDIA GeForce RTX 5090\t12\t0\t32\n"
            + f"FIG15_MEASUREMENT\t{warmup}\t{launches}\t{elapsed_ms:.9g}\n"
            + "FIG15_CORRECT\t32\t0\n",
            encoding="utf-8",
        )
        if not attached:
            continue
        (directory / "agent.log").write_text(
            "Verifier mode: WARNING\n"
            "Registered shared memory with CUDA: addr=test size=256\n"
            "Global shm constructed. shm_open_type 1 for fig15_map_test\n"
            "Global shm initialized\n",
            encoding="utf-8",
        )
        lines = [f"FIG15_READY\t{arm}\t1"]
        expectation = analyzer.expected_map(arm)
        if expectation is not None:
            name, values = expectation
            lines.extend(
                f"FIG15_MAP\t{name}\t{key}\t{value}"
                for key, value in values.items()
            )
        lines.append("FIG15_DETACHED\t1")
        (directory / "loader.log").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


class FrozenScheduleTests(unittest.TestCase):
    def test_runner_and_independent_analyzer_freeze_the_same_schedule(self) -> None:
        self.assertEqual(runner.frozen_schedule("full"), analyzer.expected_schedule("full"))

    def test_every_arm_occupies_every_position_twice(self) -> None:
        counts = Counter(
            (str(item["arm"]), int(item["order"]))
            for item in runner.frozen_schedule("full")
        )
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(len(counts), len(runner.ARMS) ** 2)

    def test_phase_budget_is_fixed(self) -> None:
        self.assertEqual(runner.phase_parameters("preflight"), (1, 1, 2))
        self.assertEqual(runner.phase_parameters("full"), (16, 8, 64))
        self.assertEqual(len(runner.frozen_schedule("full")), 128)


class SharedMemoryCleanupTests(unittest.TestCase):
    def test_owned_regular_segment_identity_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "segment"
            path.write_bytes(b"created-before-ready")
            info = path.stat()
            self.assertEqual(
                runner.owned_segment_identity(path),
                (info.st_dev, info.st_ino, os.getuid()),
            )

    def test_symlink_segment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(b"not-the-segment")
            link = root / "segment"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "not an owned file"):
                runner.owned_segment_identity(link)


class IndependentReplayTests(unittest.TestCase):
    def test_complete_raw_fixture_supports_both_primary_operations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            write_fixture(root)
            (root / "summary.tsv").write_text(
                "untrusted\tpost-hoc\n", encoding="utf-8"
            )
            analysis = analyzer.analyze_campaign(root)
        self.assertEqual(analysis["run_status"], "valid")
        self.assertEqual(analysis["tested_hypothesis"], "supported")
        self.assertEqual(analysis["raw_arm_processes"], 128)
        self.assertEqual(
            analysis["effects"]["host_vs_device_update"]["pairs"], 16
        )
        self.assertGreater(
            analysis["effects"]["host_vs_device_update"]["ratio_low"], 1.0
        )
        self.assertGreater(
            analysis["effects"]["host_vs_device_lookup"]["ratio_low"], 1.0
        )

    def test_preflight_is_valid_but_not_a_paper_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            write_fixture(root, "preflight")
            analysis = analyzer.analyze_campaign(root)
        self.assertEqual(analysis["run_status"], "valid_preflight")
        self.assertEqual(analysis["tested_hypothesis"], "not_tested")
        self.assertEqual(analysis["effects"], {})

    def test_wrong_map_value_invalidates_raw_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            write_fixture(root)
            item = next(
                value for value in analyzer.expected_schedule("full")
                if value["arm"] == "host_update"
            )
            path = root / (
                f"block-{int(item['block']):02d}-order-{int(item['order']):02d}-"
                "host_update/loader.log"
            )
            text = path.read_text(encoding="utf-8")
            expected = str(analyzer.UPDATE_MAGIC)
            path.write_text(text.replace(expected, "1", 1), encoding="utf-8")
            with self.assertRaisesRegex(analyzer.AnalysisError, "map oracle"):
                analyzer.analyze_campaign(root)

    def test_missing_engagement_invalidates_raw_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            write_fixture(root)
            item = next(
                value for value in analyzer.expected_schedule("full")
                if value["arm"] == "device_lookup"
            )
            path = root / (
                f"block-{int(item['block']):02d}-order-{int(item['order']):02d}-"
                "device_lookup/application.log"
            )
            path.write_text(
                path.read_text(encoding="utf-8").replace("Attach successfully\n", ""),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(analyzer.AnalysisError, "engagement"):
                analyzer.analyze_campaign(root)

    def test_engagement_may_be_split_across_application_and_agent_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            write_fixture(root)
            item = next(
                value for value in analyzer.expected_schedule("full")
                if value["arm"] == "device_lookup"
            )
            directory = root / (
                f"block-{int(item['block']):02d}-order-{int(item['order']):02d}-"
                "device_lookup"
            )
            application = directory / "application.log"
            agent = directory / "agent.log"
            markers = (
                "[ptxpass] kprobe_entry_stub: matched=1, in=1, out=2\n"
                "Loaded module: patched.map-bench.ptx\n"
                "Attach successfully\n"
            )
            application.write_text(
                application.read_text(encoding="utf-8").replace(markers, ""),
                encoding="utf-8",
            )
            agent.write_text(
                agent.read_text(encoding="utf-8") + markers,
                encoding="utf-8",
            )
            runner.validate_engagement_logs(application, agent)
            analysis = analyzer.analyze_campaign(root)
        self.assertEqual(analysis["run_status"], "valid")

    def test_schedule_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            write_fixture(root)
            path = root / "schedule.tsv"
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("\tnative\t", "\tnoop\t", 1),
                            encoding="utf-8")
            with self.assertRaisesRegex(analyzer.AnalysisError, "schedule differs"):
                analyzer.analyze_campaign(root)


if __name__ == "__main__":
    unittest.main()
