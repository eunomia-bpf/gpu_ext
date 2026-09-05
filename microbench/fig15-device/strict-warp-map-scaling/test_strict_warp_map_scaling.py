#!/usr/bin/env python3
"""Offline contract tests for the STRICT warp-key-map scaling experiment."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
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


runner = load("strict_warp_runner", HERE / "run_strict_warp_map_scaling.py")
analyzer = load("strict_warp_analyzer", HERE / "analyze_strict_warp_map_scaling.py")


class SourceContractTests(unittest.TestCase):
    def test_warp_bpf_program_is_keyed(self) -> None:
        text = (HERE / "warp_map_probe.bpf.c").read_text(encoding="utf-8")
        self.assertIn("bpf_get_warp_id", text)
        self.assertEqual(text.count('SEC("kprobe/fig15_warp_map_kernel")'), 3)
        for program in runner.PROGRAMS.values():
            self.assertIn(f"int {program}(void)", text)
        self.assertIn("WARP_MAGIC", text)
        self.assertIn("WARP_MAP_ENTRIES", text)

    def test_runner_and_analyzer_schedules_match(self) -> None:
        self.assertEqual(
            runner.frozen_schedule("full"), analyzer.expected_schedule("full")
        )
        self.assertEqual(
            runner.frozen_schedule("preflight"), analyzer.expected_schedule("preflight")
        )

    def test_loader_and_strict_markers_are_required(self) -> None:
        runner_text = (HERE / "run_strict_warp_map_scaling.py").read_text(encoding="utf-8")
        makefile = (HERE / "Makefile").read_text(encoding="utf-8")
        self.assertIn('"BPFTIME_VERIFIER_LEVEL": "STRICT"', runner_text)
        self.assertIn('"ENABLE_EBPF_VERIFIER": "ON"', runner_text)
        self.assertIn("build-table1-575-strict", makefile)
        self.assertIn("GPU eBPF verification accepted: mode=STRICT", runner_text)
        self.assertIn("libbpf: loading object from .+$", runner_text)
        self.assertIn("binary_contains(agent, marker)", runner_text)
        self.assertNotIn("binary_contains(server, marker)", runner_text)

    def test_loader_shutdown_precedes_detach_validation(self) -> None:
        runner_text = (HERE / "run_strict_warp_map_scaling.py").read_text(encoding="utf-8")
        attached = runner_text[runner_text.index("def run_attached("):]
        close_app = attached.index("application_stream.close()")
        signal_loader = attached.index("os.killpg(loader_process.pid, signal.SIGINT)")
        join_loader = attached.index("loader_process.wait(timeout=90)")
        validate_loader = attached.index("validate_loader_log(loader_log, shape, arm)")
        self.assertLess(close_app, signal_loader)
        self.assertLess(signal_loader, join_loader)
        self.assertLess(join_loader, validate_loader)

    def test_plan_scope_boundaries(self) -> None:
        text = (HERE / "plan.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for phrase in (
            "does not remove the 32 identical per-lane overwrites",
            "does not test a warp-leader execution optimization",
            "complements, rather than replaces",
            "No pooling across shapes",
        ):
            self.assertIn(phrase, normalized)


class ScheduleTests(unittest.TestCase):
    def test_full_schedule_position_balance(self) -> None:
        schedule = runner.frozen_schedule("full")
        self.assertEqual(len(schedule), 8 * 5 * 4)
        counts = Counter((item["shape"], item["arm"]) for item in schedule)
        # each shape has one block per arm in each block
        self.assertEqual(set(counts.values()), {8})
        block_counts = Counter((item["shape"], item["block"]) for item in schedule)
        self.assertEqual(set(block_counts.values()), {4})

    def test_preflight_schedule(self) -> None:
        schedule = runner.frozen_schedule("preflight")
        self.assertEqual(len(schedule), 5 * 4)
        blocks = Counter((item["shape"], item["order"]) for item in schedule)
        self.assertEqual(set(blocks.values()), {1})


class CampaignFixtureMixin:
    @staticmethod
    def _application_log(arm: str, pid: int, warmup: int, launches: int,
                        shape: int) -> str:
        lines = [
            f"FIG15_DEVICE\t{analyzer.GPU_NAME}\t12\t0\t32",
            f"FIG15_MEASUREMENT\t{warmup}\t{launches}\t0.01",
            f"FIG15_CORRECT\t{shape}\t0",
        ]
        if arm != "native":
            program = analyzer.PROGRAMS[arm]
            lines.extend([
                f"[date][info][{pid}] Instantiating bpf link 1 and the corresponding "
                f"program {program} is cuda program",
                "[ptxpass] kprobe_entry_stub: matched=1, in=1, out=2",
                "[date][info] [ignored]",
                f"[date][info][{pid}] GPU eBPF verification timing: program={program} "
                "verification_elapsed_ns=7",
                f"[date][info][{pid}] GPU eBPF verification accepted: mode=STRICT "
                f"program={program} attach=kprobe/fig15_warp_map_kernel instructions=9",
                f"[date][info][{pid}] GPU eBPF verified map: program={program} fd=33 "
                "type=1503 key_size=4 value_size=8 max_entries=64",
                "[date][info] Loaded module: patched.warp_map_bench.sm_120.ptx",
                "[date][info] Attach successfully",
            ])
        return "\n".join(lines) + "\n"

    @staticmethod
    def _loader_log(shape: int, arm: str) -> str:
        lines = [
            "FIG15_WARP_SERVER_PRIMED\t1",
            "libbpf: loading object from object",
            f"FIG15_WARP_READY\t{arm}\t1",
        ]
        if arm == "noop":
            lines.append("FIG15_WARP_DETACHED\t1")
            return "\n".join(lines) + "\n"
        if arm == "shared_update":
            lines.append(f"FIG15_WARP_MAP\t0\t{analyzer.WARP_MAGIC}")
        else:
            active = max(1, shape // 32)
            for warp in range(active):
                lines.append(f"FIG15_WARP_MAP\t{warp}\t{analyzer.WARP_MAGIC ^ warp}")
        lines.append("FIG15_WARP_DETACHED\t1")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _agent_log() -> str:
        return (
            "Verifier mode: STRICT\n"
            "Registered shared memory with CUDA: x\n"
            "Global shm constructed. shm_open_type 1 for fig15_warp_fixture\n"
            "Global shm initialized\n"
        )

    def write_preflight(self, root: Path) -> None:
        self.write_campaign(root, "preflight")

    def write_campaign(self, root: Path, phase: str) -> None:
        schedule = analyzer.expected_schedule(phase)
        with (root / "schedule.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("shape", "block", "order", "arm", "run_id"),
                delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(schedule)
        (root / "environment.txt").write_text(
            f"gpu\t{analyzer.GPU_NAME}\n"
            f"driver\t{analyzer.DRIVER}\n"
            "BPFTIME_ENABLE_CUDA_ATTACH\tON\n"
            "BPFTIME_LLVM_JIT\tON\n"
            "ENABLE_EBPF_VERIFIER\tON\n"
            "strict_agent_markers\tpresent\n"
            "syscall_server_binary\tpresent\n"
            "bpftime_revision\trevision\n"
            "agent_bytes\t1\nserver_bytes\t1\n"
            "nvcc_begin\nversion\nnvcc_end\n",
            encoding="utf-8",
        )
        warmup = analyzer.phase_parameters(phase)[1]
        launches = analyzer.phase_parameters(phase)[2]
        for index, item in enumerate(schedule, start=1):
            shape = int(item["shape"])
            arm = str(item["arm"])
            directory = root / (
                f"shape-{shape}-block-{int(item['block']):02d}-order-{int(item['order']):02d}-{arm}"
            )
            directory.mkdir()
            pid = 2000 + index
            (directory / "application.log").write_text(
                self._application_log(arm, pid, warmup, launches, shape), encoding="utf-8"
            )
            if arm != "native":
                (directory / "loader.log").write_text(
                    self._loader_log(shape, arm), encoding="utf-8"
                )
                (directory / "agent.log").write_text(self._agent_log(), encoding="utf-8")
                (directory / "execution.tsv").write_text(
                    "target_pid\treturncode\tverifier_level\n"
                    f"{pid}\t0\tSTRICT\n", encoding="utf-8"
                )


class AnalyzerFixtureTests(CampaignFixtureMixin, unittest.TestCase):
    def test_complete_preflight_replays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_preflight(root)
            result = analyzer.analyze_campaign(root)
        self.assertEqual(result["run_status"], "valid_preflight")
        self.assertEqual(result["raw_arm_processes"], 20)
        self.assertEqual(result["strict_accepted_cells"], 15)

    def test_strict_rejection_invalidates_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_preflight(root)
            arm_dir = next(root.glob("*-shared_update"))
            with (arm_dir / "application.log").open("a", encoding="utf-8") as stream:
                stream.write("Skipping GPU eBPF verification for cuda__noop\n")
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.analyze_campaign(root)

    def test_shared_update_map_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_preflight(root)
            arm_dir = next(
                directory for directory in root.glob("shape-*")
                if directory.name.endswith("-shared_update")
            )
            loader = arm_dir / "loader.log"
            loader.write_text(loader.read_text(encoding="utf-8").replace(
                str(analyzer.WARP_MAGIC), "42"
            ), encoding="utf-8")
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.analyze_campaign(root)


class OutputContractTests(CampaignFixtureMixin, unittest.TestCase):
    RETAINED_PREFLIGHT = HERE / "raw" / "strict-warp-scaling-preflight-575-06"

    @staticmethod
    def _read_tsv(path: Path) -> list[list[str]]:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.reader(stream, delimiter="\t"))

    def test_preflight_outputs_emit_no_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_preflight(root)
            result = analyzer.analyze_campaign(root)
            tsv = root / "analysis.tsv"
            analyzer.write_tsv(tsv, result)
            rows = self._read_tsv(tsv)
            markdown = analyzer.render_markdown(result)
        self.assertEqual(result["run_status"], "valid_preflight")
        self.assertEqual(result["effects"], {})
        self.assertEqual(result["secondary"], {})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "scope")
        self.assertIn("Preflight establishes only execution", markdown)
        self.assertIn("No paired or cross-shape effects", markdown)
        self.assertNotIn("Per-shape paired primary effects", markdown)

    def test_retained_preflight_emits_no_invented_effects(self) -> None:
        if not self.RETAINED_PREFLIGHT.is_dir():
            self.skipTest("retained raw preflight 575-06 is not present")
        with tempfile.TemporaryDirectory() as temporary:
            tsv = Path(temporary) / "analysis.tsv"
            result = analyzer.analyze_campaign(self.RETAINED_PREFLIGHT)
            analyzer.write_tsv(tsv, result)
            rows = self._read_tsv(tsv)
            markdown = analyzer.render_markdown(result)
        self.assertEqual(result["run_status"], "valid_preflight")
        self.assertEqual(result["raw_arm_processes"], 20)
        self.assertEqual(result["strict_accepted_cells"], 15)
        self.assertEqual(result["effects"], {})
        self.assertEqual(result["secondary"], {})
        self.assertEqual(len(rows), 1)
        self.assertNotIn("Per-shape paired primary effects", markdown)

    def test_preflight_with_invented_effects_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_preflight(root)
            result = analyzer.analyze_campaign(root)
            result["effects"] = {"shape-32-shared_vs_noop": {"pairs": 1}}
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.write_tsv(root / "analysis.tsv", result)

    def test_full_phase_writes_every_effect_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_campaign(root, "full")
            result = analyzer.analyze_campaign(root)
            tsv = root / "analysis.tsv"
            analyzer.write_tsv(tsv, result)
            rows = self._read_tsv(tsv)
            markdown = analyzer.render_markdown(result)
        self.assertEqual(result["raw_arm_processes"], 160)
        self.assertEqual(result["strict_accepted_cells"], 120)
        self.assertEqual(
            list(result["effects"]), list(analyzer.expected_paired_effect_keys())
        )
        self.assertEqual(len(rows), 1 + 15 + 3)
        paired = [row for row in rows[1:] if row[0] == "paired_shape"]
        cross = [row for row in rows[1:] if row[0] == "cross_shape"]
        self.assertEqual(len(paired), 15)
        self.assertEqual(len(cross), 3)
        self.assertEqual(
            [row[2] for row in paired[:3]],
            ["shared_update/noop", "warp_update/noop", "warp_update/shared_update"],
        )
        self.assertEqual([row[1] for row in cross], ["32->1024"] * 3)
        self.assertEqual([row[2] for row in cross], list(analyzer.CROSS_SCALE_NAMES))
        self.assertIn("| warp_update/shared_update | 8 |", markdown)
        self.assertNotIn("warp_vs_shared_update", markdown)

    def test_full_phase_with_missing_effect_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_campaign(root, "full")
            result = analyzer.analyze_campaign(root)
            result["effects"].pop(analyzer.paired_effect_key(256, "warp_vs_shared"))
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.write_tsv(root / "analysis.tsv", result)

    def test_unexpected_cell_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_preflight(root)
            (root / "shape-2048-block-01-order-01-noop").mkdir()
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.analyze_campaign(root)


class CrossProcessReproducibilityTests(CampaignFixtureMixin, unittest.TestCase):
    REPRINT_SCRIPT = (
        "import json, pathlib, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import analyze_strict_warp_map_scaling as analyzer\n"
        "result = analyzer.analyze_campaign(pathlib.Path(sys.argv[2]))\n"
        "print(json.dumps(result))\n"
    )

    def _analyze_in_process(self, campaign: Path, hash_seed: str) -> str:
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", self.REPRINT_SCRIPT, str(HERE), str(campaign)],
            capture_output=True, text=True, env=environment, check=True,
        )
        return completed.stdout

    def test_bootstrap_seeds_ignore_python_hash_seed(self) -> None:
        expected = [
            analyzer.paired_effect_seed(shape, index)
            for shape in analyzer.SHAPES
            for index in range(len(analyzer.COMPARISONS))
        ]
        self.assertEqual(len(set(expected)), len(expected))
        self.assertTrue(all(0 <= seed <= 0xFFFFFFFF for seed in expected))

    def test_full_analysis_is_identical_across_hash_seed_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_campaign(root, "full")
            outputs = {
                hash_seed: self._analyze_in_process(root, hash_seed)
                for hash_seed in ("0", "1", "42", "12345")
            }
        self.assertEqual(len(set(outputs.values())), 1)
        result = json.loads(next(iter(outputs.values())))
        self.assertEqual(result["run_status"], "valid")
        self.assertEqual(
            list(result["effects"]), list(analyzer.expected_paired_effect_keys())
        )


if __name__ == "__main__":
    unittest.main()
