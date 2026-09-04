#!/usr/bin/env python3
"""Offline contract tests for the STRICT uniform-map experiment."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("strict_uniform_runner", HERE / "run_strict_uniform_map.py")
analyzer = load("strict_uniform_analyzer", HERE / "analyze_strict_uniform_map.py")


class SourceContractTests(unittest.TestCase):
    def test_bpf_program_is_uniform_and_separate(self) -> None:
        text = (HERE / "uniform_probe.bpf.c").read_text(encoding="utf-8")
        self.assertNotIn("bpf_get_lane_id", text)
        self.assertEqual(text.count('SEC("kprobe/fig15_map_kernel")'), 5)
        for program in runner.PROGRAMS.values():
            self.assertIn(f"int {program}(void)", text)
            self.assertLessEqual(len(program), 15)
        self.assertIn("u32 key = 0;", text)
        self.assertIn("u64 value = UPDATE_MAGIC;", text)

    def test_strict_build_and_runtime_markers_are_required(self) -> None:
        runner_text = (HERE / "run_strict_uniform_map.py").read_text(encoding="utf-8")
        makefile = (HERE / "Makefile").read_text(encoding="utf-8")
        self.assertIn('"ENABLE_EBPF_VERIFIER": "ON"', runner_text)
        self.assertIn('"BPFTIME_VERIFIER_LEVEL": "STRICT"', runner_text)
        self.assertIn("build-table1-575-strict", makefile)
        self.assertIn("GPU eBPF verification accepted: mode=STRICT", runner_text)
        self.assertIn('"target_pid\\treturncode\\tverifier_level\\n"', runner_text)

    def test_plan_preserves_claim_boundary(self) -> None:
        text = (HERE / "plan.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for phrase in (
            "not the earlier per-lane map workload",
            "not warp-leader aggregation",
            "not invocation",
            "or verifier soundness",
        ):
            self.assertIn(phrase, normalized)


class ScheduleTests(unittest.TestCase):
    def test_runner_and_analyzer_schedules_match(self) -> None:
        self.assertEqual(
            runner.frozen_schedule("full"), analyzer.expected_schedule("full")
        )

    def test_full_schedule_is_twice_position_balanced(self) -> None:
        schedule = runner.frozen_schedule("full")
        self.assertEqual(len(schedule), 72)
        counts = Counter((row["arm"], row["order"]) for row in schedule)
        self.assertEqual(set(counts.values()), {2})
        block_counts = Counter(row["block"] for row in schedule)
        self.assertEqual(set(block_counts.values()), {6})


class AnalyzerFixtureTests(unittest.TestCase):
    @staticmethod
    def application_log(arm: str, pid: int, warmup: int, launches: int) -> str:
        base = (
            f"FIG15_DEVICE\t{analyzer.GPU_NAME}\t12\t0\t32\n"
            f"FIG15_MEASUREMENT\t{warmup}\t{launches}\t0.02\n"
            "FIG15_CORRECT\t32\t0\n"
        )
        if arm == "native":
            return base
        program = analyzer.PROGRAMS[arm]
        lines = [
            f"[date][info][{pid}] Instantiating bpf link 1 and the corresponding "
            f"program {program} is cuda program",
            "[ptxpass] kprobe_entry_stub: matched=1, in=1, out=2",
            f"[date][info][{pid}] GPU eBPF verification timing: program={program} "
            "verification_elapsed_ns=7",
            f"[date][info][{pid}] GPU eBPF verification accepted: mode=STRICT "
            f"program={program} attach=kprobe/fig15_map_kernel instructions=9",
            f"[date][info][{pid}] GPU eBPF verified map: program={program} fd=16 "
            "type=1503 key_size=4 value_size=8 max_entries=1",
            f"[date][info][{pid}] GPU eBPF verified map: program={program} fd=17 "
            "type=1513 key_size=4 value_size=8 max_entries=1",
            f"[date][info][{pid}] GPU eBPF verified map: program={program} fd=18 "
            "type=1503 key_size=4 value_size=8 max_entries=1",
            f"[date][info][{pid}] Loaded module: patched.map_bench.sm_120.ptx",
            f"[date][info][{pid}] Attach successfully",
        ]
        return "\n".join(lines) + "\n" + base

    @staticmethod
    def loader_log(arm: str) -> str:
        rows = [
            "FIG15_UNIFORM_SERVER_PRIMED\t1",
            "libbpf: loading object from object",
            f"FIG15_UNIFORM_READY\t{arm}\t1",
        ]
        for (name, key), value in analyzer.expected_map_rows(arm).items():
            rows.append(f"FIG15_UNIFORM_MAP\t{name}\t{key}\t{value}")
        rows.append("FIG15_UNIFORM_DETACHED\t1")
        return "\n".join(rows) + "\n"

    @staticmethod
    def agent_log() -> str:
        return (
            "Verifier mode: STRICT\n"
            "Registered shared memory with CUDA: x\n"
            "Global shm constructed. shm_open_type 1 for fig15_uniform_fixture\n"
            "Global shm initialized\n"
        )

    def write_preflight(self, root: Path) -> None:
        schedule = analyzer.expected_schedule("preflight")
        with (root / "schedule.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("block", "order", "arm", "run_id"), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(schedule)
        (root / "environment.txt").write_text(
            f"gpu\t{analyzer.GPU_NAME}\n"
            f"driver\t{analyzer.DRIVER}\n"
            "BPFTIME_ENABLE_CUDA_ATTACH\tON\n"
            "BPFTIME_LLVM_JIT\tON\n"
            "ENABLE_EBPF_VERIFIER\tON\n"
            "strict_binary_markers\tpresent\n"
            "bpftime_revision\trevision\n"
            "agent_bytes\t1\nserver_bytes\t1\n"
            "nvcc_begin\nversion\nnvcc_end\n",
            encoding="utf-8",
        )
        for index, item in enumerate(schedule, start=1):
            arm = str(item["arm"])
            directory = root / (
                f"block-{int(item['block']):02d}-order-{int(item['order']):02d}-{arm}"
            )
            directory.mkdir()
            pid = 1000 + index
            (directory / "application.log").write_text(
                self.application_log(arm, pid, 1, 2), encoding="utf-8"
            )
            if arm != "native":
                (directory / "loader.log").write_text(
                    self.loader_log(arm), encoding="utf-8"
                )
                (directory / "agent.log").write_text(
                    self.agent_log(), encoding="utf-8"
                )
                (directory / "execution.tsv").write_text(
                    "target_pid\treturncode\tverifier_level\n"
                    f"{pid}\t0\tSTRICT\n", encoding="utf-8"
                )

    def test_complete_preflight_replays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_preflight(root)
            result = analyzer.analyze_campaign(root)
        self.assertEqual(result["run_status"], "valid_preflight")
        self.assertEqual(result["raw_arm_processes"], 6)
        self.assertEqual(result["strict_accepted_cells"], 5)

    def test_skip_record_invalidates_strict_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_preflight(root)
            arm_dir = next(root.glob("*-noop"))
            with (arm_dir / "application.log").open("a", encoding="utf-8") as stream:
                stream.write("Skipping GPU eBPF verification for cuda__noop\n")
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.analyze_campaign(root)

    def test_wrong_map_effect_invalidates_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_preflight(root)
            arm_dir = next(root.glob("*-device_update"))
            loader = arm_dir / "loader.log"
            loader.write_text(
                loader.read_text(encoding="utf-8").replace(
                    str(analyzer.UPDATE_MAGIC), "0"
                ), encoding="utf-8"
            )
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.analyze_campaign(root)


if __name__ == "__main__":
    unittest.main()
