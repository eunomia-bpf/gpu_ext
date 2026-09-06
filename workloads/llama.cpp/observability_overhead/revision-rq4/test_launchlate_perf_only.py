#!/usr/bin/env python3
"""CPU-only checks for the launchlate performance-only runner."""

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import run_revision_rq4 as gated
import run_launchlate_perf_only as perf


FAKE_SNAPSHOT = {
    "gpu": "NVIDIA GeForce RTX 5090, 575.57.08, 32640, 0, 0, 28.67 W",
    "compute_apps": "",
}


def default_args(tmp: Path | None = None, **overrides):
    argv = []
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}"])
        if isinstance(value, str) or isinstance(value, Path):
            argv.append(str(value))
        elif isinstance(value, bool):
            continue
        else:
            argv.append(str(value))
    return perf.parse_args(argv)


def zero_gpubpf_probe() -> dict:
    return gated.parse_gpubpf("launchlate", "")


def zero_nvbit_probe() -> dict:
    return gated.parse_nvbit("launchlate", "NVBIT selected_launches=0\n")


class PerfOnlyPlanTests(unittest.TestCase):
    def test_dry_run_freezes_the_three_arm_matrix(self):
        args = default_args()
        plan = perf.dry_run_plan(args)
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["kind"], "launchlate_perf_only")
        self.assertEqual(plan["configs"], list(perf.CONFIGS))
        self.assertEqual(plan["runs"], 10)
        self.assertEqual(plan["pp"], 512)
        self.assertEqual(plan["timing_cell_count"], 30)
        self.assertEqual(plan["schedule_seed"], gated.SCHEDULE_SEED)

    def test_schedule_matches_gated_runner_for_the_same_seed(self):
        args = default_args()
        plan = perf.dry_run_plan(args)
        self.assertEqual(plan["timing_schedule"], gated.fixed_schedule(args))
        for block in range(1, 11):
            order = plan["timing_schedule"][str(block)]
            self.assertEqual(sorted(order), sorted(perf.CONFIGS))

    def test_dry_run_declares_every_bypassed_gate(self):
        plan = perf.dry_run_plan(default_args())
        self.assertEqual(plan["completion_rule"]["attempts_per_cell"], 1)
        bypassed = " | ".join(plan["bypassed_gates"])
        self.assertIn("RM/PTIMER", bypassed)
        self.assertIn("1.5 us", bypassed)
        self.assertIn("source-schema", bypassed)
        self.assertIn("correctness", bypassed)
        self.assertIn("verifier", bypassed)
        self.assertIn("hook accounting", plan["completion_rule"]["hook_accounting"])
        preserved = " | ".join(plan["preserved"])
        self.assertIn("stdout/stderr", preserved)
        self.assertIn("zero", preserved)

    def test_defaults_target_the_table1_runtime(self):
        args = default_args()
        self.assertEqual(args.tools, ["launchlate"])
        self.assertEqual(
            args.bpftime_root,
            Path("/home/yunwei37/workspace/gpu/bpftime-table1-575"),
        )
        self.assertEqual(
            args.bpftime_build_dir,
            Path("/home/yunwei37/workspace/gpu/bpftime-table1-575/build-launchlate-575"),
        )
        self.assertEqual(args.gpu_thread_count, 22528)


class BenchmarkValidityTests(unittest.TestCase):
    def test_valid_baseline_run(self):
        result = {"returncode": 0, "metrics": {"pp_tokens": 512, "pp_tok_s": 123.4}}
        self.assertTrue(perf.benchmark_valid(result, 512))

    def test_zero_counters_never_invalidate(self):
        result = {
            "returncode": 0,
            "metrics": {"pp_tokens": 512, "pp_tok_s": 10.5},
            "probe": zero_gpubpf_probe(),
        }
        result["probe"]["sample_count"] = 0
        self.assertTrue(perf.benchmark_valid(result, 512))

    def test_failed_or_misshapen_benchmark_is_invalid(self):
        self.assertFalse(
            perf.benchmark_valid(
                {"returncode": 1, "metrics": {"pp_tokens": 512, "pp_tok_s": 1.0}}, 512
            )
        )
        self.assertFalse(
            perf.benchmark_valid(
                {"returncode": 0, "metrics": {"pp_tokens": 32, "pp_tok_s": 1.0}}, 512
            )
        )
        self.assertFalse(
            perf.benchmark_valid(
                {"returncode": 0, "metrics": {"pp_tokens": 512, "pp_tok_s": 0.0}}, 512
            )
        )
        self.assertFalse(
            perf.benchmark_valid(
                {"returncode": 0, "metrics": {"pp_tokens": 512, "pp_tok_s": float("nan")}},
                512,
            )
        )
        self.assertFalse(perf.benchmark_valid({"returncode": 0}, 512))


class SummarizeTests(unittest.TestCase):
    def make_state(self, tmp: Path) -> dict:
        args = default_args()
        state = perf.new_state(args, "20260905_000000", FAKE_SNAPSHOT)
        state["params"]["runs"] = 1
        state["schedule"] = {"1": list(perf.CONFIGS)}
        runs = {
            "baseline": (100.0, zero_gpubpf_probe()),
            "gpubpf_launchlate": (90.0, zero_gpubpf_probe()),
            "nvbit_launchlate": (80.0, zero_nvbit_probe()),
        }
        for config, (tok_s, probe) in runs.items():
            state["configs"][config]["runs"].append({
                "block": 1,
                "attempt": 1,
                "run": 101,
                "returncode": 0,
                "log": f"{config}_run_101/llama_bench.log",
                "metrics": {"pp_tokens": 512, "pp_tok_s": tok_s},
                "probe": probe,
                "valid": True,
            })
        return state

    def test_zero_counter_state_reports_throughput_and_overhead(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            state = self.make_state(tmp)
            summary = perf.summarize(state)
            block = summary["blocks"][0]
            self.assertTrue(block["complete"])
            self.assertEqual(block["baseline"], 100.0)
            self.assertEqual(block["gpubpf_launchlate"], 90.0)
            self.assertEqual(block["nvbit_launchlate"], 80.0)
            self.assertAlmostEqual(block["gpubpf_launchlate_overhead_pct"], 10.0)
            self.assertAlmostEqual(block["nvbit_launchlate_overhead_pct"], 20.0)
            rows = {row["config"]: row for row in summary["configs"]}
            self.assertEqual(rows["baseline"]["valid_blocks"], 1)
            self.assertAlmostEqual(rows["baseline"]["pp_tok_s_geomean"], 100.0)
            self.assertAlmostEqual(rows["gpubpf_launchlate"]["pp_tok_s_geomean"], 90.0)
            self.assertAlmostEqual(rows["gpubpf_launchlate"]["mean_overhead_pct_vs_baseline"], 10.0)
            self.assertAlmostEqual(rows["nvbit_launchlate"]["mean_overhead_pct_vs_baseline"], 20.0)
            self.assertIsNone(rows["baseline"]["mean_overhead_pct_vs_baseline"])

            perf.write_state(tmp, state)
            result = json.loads((tmp / "result.json").read_text())
            self.assertEqual(result["kind"], "launchlate_perf_only")
            self.assertEqual(result["summary"]["blocks"][0]["complete"], True)
            markdown = (tmp / "summary.md").read_text()
            self.assertIn("10.00%", markdown)
            self.assertIn("20.00%", markdown)
            self.assertIn("No cell is rejected or retried on hook accounting", markdown)
            csv_text = (tmp / "summary.csv").read_text()
            self.assertIn("mean_overhead_pct_vs_baseline", csv_text)
            self.assertIn("gpubpf_launchlate", csv_text)
        finally:
            shutil.rmtree(tmp)

    def test_incomplete_block_is_reported_not_invented(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            state = self.make_state(tmp)
            del state["configs"]["nvbit_launchlate"]["runs"][0]
            summary = perf.summarize(state)
            block = summary["blocks"][0]
            self.assertFalse(block["complete"])
            self.assertIsNone(block["nvbit_launchlate"])
            self.assertIsNone(block["nvbit_launchlate_overhead_pct"])
            rows = {row["config"]: row for row in summary["configs"]}
            self.assertEqual(rows["nvbit_launchlate"]["valid_blocks"], 0)
            self.assertIsNone(rows["nvbit_launchlate"]["pp_tok_s_geomean"])
        finally:
            shutil.rmtree(tmp)


class ResumeTests(unittest.TestCase):
    def make_artifacts(self, tmp: Path) -> dict:
        tool_dir = tmp / "gpubpf_tool_build" / "launchlate"
        tool_dir.mkdir(parents=True, exist_ok=True)
        tool_bin = tool_dir / "launchlate"
        tool_bin.write_bytes(b"ELF-test")
        nvbit_tool = tmp / "nvbit_tool_build" / "observability.so"
        nvbit_tool.parent.mkdir(parents=True, exist_ok=True)
        nvbit_tool.write_bytes(b"\x7fELF-test")
        return {
            "gpubpf_launchlate": {"path": str(tool_bin)},
            "nvbit_tool": {"path": str(nvbit_tool)},
        }

    def test_resume_rebonds_recorded_tools(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            args = default_args()
            state = perf.new_state(args, "20260905_000000", FAKE_SNAPSHOT)
            state["artifacts"] = self.make_artifacts(tmp)
            tool_dirs, nvbit_tool = perf.verify_resume(state, args)
            self.assertEqual(tool_dirs["launchlate"], tmp / "gpubpf_tool_build" / "launchlate")
            self.assertEqual(nvbit_tool, tmp / "nvbit_tool_build" / "observability.so")
        finally:
            shutil.rmtree(tmp)

    def test_resume_rejects_mismatched_params(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            args = default_args()
            state = perf.new_state(args, "20260905_000000", FAKE_SNAPSHOT)
            state["artifacts"] = self.make_artifacts(tmp)
            state["params"]["pp"] = 32
            with self.assertRaises(RuntimeError):
                perf.verify_resume(state, args)
        finally:
            shutil.rmtree(tmp)

    def test_resume_rejects_mismatched_schedule_or_artifacts(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            args = default_args()
            state = perf.new_state(args, "20260905_000000", FAKE_SNAPSHOT)
            state["artifacts"] = self.make_artifacts(tmp)
            state["schedule"] = {}
            with self.assertRaises(RuntimeError):
                perf.verify_resume(state, args)

            state["schedule"] = gated.fixed_schedule(args)
            del state["artifacts"]["nvbit_tool"]
            with self.assertRaises(RuntimeError):
                perf.verify_resume(state, args)

            state["artifacts"] = self.make_artifacts(tmp)
            Path(state["artifacts"]["nvbit_tool"]["path"]).unlink()
            with self.assertRaises(RuntimeError):
                perf.verify_resume(state, args)
        finally:
            shutil.rmtree(tmp)


class ParseLayerTests(unittest.TestCase):
    def test_zero_gpubpf_counters_are_recorded_not_fatal(self):
        probe = zero_gpubpf_probe()
        # Optional metadata: counters are 0 or None when the log has none.
        self.assertIn(probe["sample_count"], (0, None))
        self.assertIsNone(probe["matched_samples"])
        self.assertIsNone(probe["host_launches"])

    def test_zero_nvbit_counters_are_recorded_not_fatal(self):
        probe = zero_nvbit_probe()
        self.assertEqual(probe["sample_count"], 0)
        self.assertEqual(probe["selected_launches"], 0)
        self.assertEqual(probe["clock_errors"], -1)
        self.assertEqual(len(probe["histogram"]), 10)


if __name__ == "__main__":
    unittest.main()
