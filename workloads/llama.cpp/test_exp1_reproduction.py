#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run_exp1_reproduction", HERE / "run_exp1_reproduction.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Exp1ReproductionTests(unittest.TestCase):
    def test_parse_markdown_bench(self) -> None:
        parsed = MODULE.parse_markdown_bench(
            "| model | size | params | backend | ngl | pp512 | 242.35 ± 4.06 |\n"
            "| model | size | params | backend | ngl | tg128 | 16.37 ± 0.03 |\n"
            "build: 6e603612 (7101)\n"
        )
        self.assertEqual(parsed["build_commit"], "6e603612")
        self.assertEqual(parsed["build_number"], 7101)
        self.assertEqual(parsed["tests"]["pp512"]["tokens_per_second"], 242.35)
        self.assertEqual(parsed["tests"]["tg128"]["stddev_tokens_per_second"], 0.03)

    def test_benchmark_command_is_explicit(self) -> None:
        command = MODULE.benchmark_command(
            Path("/bench"), Path("/model"), MODULE.REPLAY_CONFIGS["ncmoe32"], 3
        )
        self.assertIn("--n-prompt", command)
        self.assertIn("512", command)
        self.assertIn("--n-gen", command)
        self.assertIn("128", command)
        self.assertNotIn("taskset", command)
        self.assertEqual(command[-2:], ["--n-cpu-moe", "32"])

    def test_parse_json_result_rejects_wrong_sample_count(self) -> None:
        rows = [
            {
                "n_prompt": 512, "n_gen": 0, "n_cpu_moe": 64,
                "backends": "CUDA", "build_commit": "revision", "build_number": 1,
                "gpu_info": "gpu", "avg_ts": 1, "stddev_ts": 0, "samples_ts": [1],
            },
            {
                "n_prompt": 0, "n_gen": 128, "n_cpu_moe": 64,
                "backends": "CUDA", "build_commit": "revision", "build_number": 1,
                "gpu_info": "gpu", "avg_ts": 1, "stddev_ts": 0, "samples_ts": [1],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps(rows))
            with self.assertRaises(MODULE.ReplayError):
                MODULE.parse_json_result(path, MODULE.REPLAY_CONFIGS["ncmoe64"], 2)

    def test_reference_has_all_five_reported_cells(self) -> None:
        results = MODULE.load_reference()["reported_results_tokens_per_second"]
        self.assertEqual(
            set(results),
            {"ncmoe64", "ncmoe32", "uvm_plain", "uvm_user_hint", "gpubpf_stride_lfu"},
        )
        self.assertTrue(all(set(value) == {"pp512", "tg128"} for value in results.values()))

    def test_current_replay_checks_model_build_and_gpu(self) -> None:
        rows = [{"n_prompt": p, "n_gen": g, "n_cpu_moe": 32,
                 "backends": "CUDA", "build_commit": MODULE.safety.EXPECTED_LLAMA_COMMIT[:8],
                 "build_number": 7102, "gpu_info": MODULE.safety.EXPECTED_GPU,
                 "model_filename": "/model", "avg_ts": 1, "stddev_ts": 0, "samples_ts": [1]}
                for p, g in ((512, 0), (0, 128))]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps(rows))
            MODULE.parse_json_result(path, MODULE.REPLAY_CONFIGS["ncmoe32"], 1,
                                     expected_model=Path("/model"))
            for key, wrong in (("build_commit", "not-matching"),
                               ("gpu_info", "another GPU"), ("model_filename", "/another-model")):
                changed = [dict(row) for row in rows]
                changed[0][key] = wrong
                path.write_text(json.dumps(changed))
                with self.assertRaises(MODULE.ReplayError):
                    MODULE.parse_json_result(path, MODULE.REPLAY_CONFIGS["ncmoe32"], 1,
                                             expected_model=Path("/model"))

    def test_current_admission_uses_the_575_entrypoint(self) -> None:
        self.assertEqual(MODULE.current_stack_admission.__module__, "run_575_head_to_head")


if __name__ == "__main__":
    unittest.main()
