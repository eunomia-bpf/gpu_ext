#!/usr/bin/env python3
"""CPU-only structural tests for the LMCache revision harness."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("run_lmcache_disk.py")
SPEC = importlib.util.spec_from_file_location("run_lmcache_disk", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class HarnessTests(unittest.TestCase):
    def test_all_cells_share_runtime_repair(self):
        for config in runner.CONFIGS:
            env = runner.server_environment(config, Path("/cache"))
            self.assertEqual(env["VLLM_USE_DEEP_GEMM"], "0")
            argv = runner.server_argv(config, Path("/model"), 18080)
            self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.98")

    def test_strace_output_is_absolute_across_server_cwd(self):
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root = Path(tmp).resolve().relative_to(Path.cwd())
            self.assertFalse(root.is_absolute())
            with patch.object(runner.legacy.subprocess, "Popen"):
                _, log, _, launch = runner.start_server(
                    "lmcache_disk", Path("/model"), root.resolve() / "cache",
                    18080, root / "server.log", root / "strace")
            log.close()
            trace_output = Path(launch[launch.index("-o") + 1])
            self.assertEqual(trace_output, root.resolve() / "strace/open.trace")

    def test_request_scoped_log_parser(self):
        request_id = "cmpl-lmc-p3-warm-0"
        log = "\n".join(
            [
                f"Reqid: {request_id}, Total tokens 1550, Inference Engine computed tokens: 0, LMCache hit tokens: 1536, need to load: 1536",
                f"[req_id={request_id}] Retrieved 1536 out of 1536 required tokens (from 1550 total tokens).",
                f"[req_id={request_id}] Stored 1536 out of total 1550 tokens.",
            ]
        )
        self.assertEqual(
            runner.request_log_values(log, request_id),
            {"hits": [1536], "stores": [(1536, 1550)], "retrieved": [(1536, 1536, 1550)]},
        )

    def test_retrieval_parser_preserves_denominators(self):
        request_id = "cmpl-lmc-p0-warm-0"
        log = (
            f"Reqid: {request_id}, LMCache hit tokens: 1536\n"
            f"[req_id={request_id}] Retrieved 1536 out of 512 required tokens "
            "(from 1549 total tokens)."
        )
        values = runner.request_log_values(log, request_id)
        self.assertEqual(values["retrieved"], [(1536, 512, 1549)])

    def test_odirect_requires_every_pt_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            trace = Path(tmp) / "open.trace.1"
            lines = []
            for i in range(48):
                lines.append(f'openat(AT_FDCWD, "{cache / f"{i}.pt"}", O_WRONLY|O_CREAT|O_DIRECT, 0644) = 3')
                lines.append(f'openat(AT_FDCWD, "{cache / f"{i}.pt"}", O_RDONLY|O_DIRECT) = 3')
            trace.write_text("\n".join(lines) + "\n")
            evidence = runner.validate_odirect(Path(tmp), cache)
            self.assertEqual(evidence["write_open_count"], 48)
            self.assertEqual(evidence["read_open_count"], 48)
            self.assertEqual(len(evidence["unique_write_paths"]), 48)
            self.assertEqual(evidence["unique_write_paths"], evidence["unique_read_paths"])
            trace.write_text(
                trace.read_text()
                + f'openat(AT_FDCWD, "{cache / "bad.pt"}", O_RDONLY) = 3\n'
            )
            with self.assertRaises(runner.GateError):
                runner.validate_odirect(Path(tmp), cache)

    def test_odirect_rejects_failed_or_mismatched_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            trace = Path(tmp) / "open.trace.1"
            lines = []
            for i in range(48):
                lines.append(f'openat(AT_FDCWD, "{cache / f"{i}.pt"}", O_WRONLY|O_CREAT|O_DIRECT, 0644) = 3')
                read_index = 99 if i == 47 else i
                result = "-1 EIO (Input/output error)" if i == 0 else "3"
                lines.append(f'openat(AT_FDCWD, "{cache / f"{read_index}.pt"}", O_RDONLY|O_DIRECT) = {result}')
            trace.write_text("\n".join(lines) + "\n")
            with self.assertRaises(runner.GateError):
                runner.validate_odirect(Path(tmp), cache)

    def test_warm_gate_rejects_contradictory_partial_evidence(self):
        cold_id = "cmpl-lmc-p0-cold-0"
        warm_id = "cmpl-lmc-p0-warm-0"
        log = "\n".join(
            [
                "LMCache initialized with version 0.5.4, vllm version 0.27.1+cu129",
                f"Reqid: {cold_id}, LMCache hit tokens: 0",
                f"Reqid: {warm_id}, LMCache hit tokens: 512",
                f"Reqid: {warm_id}, LMCache hit tokens: 1536",
                f"[req_id={warm_id}] Retrieved 512 out of 1536 required tokens",
                f"[req_id={warm_id}] Retrieved 1536 out of 1536 required tokens",
            ]
        )
        observations = [{"expected_hit_tokens": 1536,
                         "cold": {"engine_request_id": cold_id},
                         "warm": {"engine_request_id": warm_id}}]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(runner.GateError):
                runner.validate_log("lmcache_cpu", log, observations, Path(tmp))

    def test_frozen_prompt_expectations(self):
        prompts = runner.load_prompts(runner.PROMPTS)
        self.assertEqual(prompts["schema"], 3)
        self.assertEqual(len(prompts["prefixes"]), 8)
        self.assertTrue(all(x["expected_hit_tokens"] == 1536 for x in prompts["prefixes"]))

    def test_runner_uses_semantic_evidence_without_content_fingerprints(self):
        source = MODULE_PATH.read_text() + runner.LEGACY_PATH.read_text()
        forbidden = ("hashlib", "sha256", "checksum", "digest", "output_hash")
        self.assertFalse(any(term in source.lower() for term in forbidden))
        for forbidden_control in ("PREFLIGHT-PASS", "SMOKE-PASS", "RUN-COMPLETE", "--resume", "--approval"):
            self.assertNotIn(forbidden_control, source)

    def test_fatal_patterns_cover_lmcache_allocation_wording(self):
        log = "Memory allocation failed while staging a disk chunk"
        self.assertTrue(any(runner.legacy.re.search(pattern, log, runner.legacy.re.I)
                            for pattern in runner.legacy.FATAL_LOG_PATTERNS))

    def test_schedule_semantics_are_exact(self):
        schedule = json.loads(runner.SCHEDULE.read_text())
        runner.validate_schedule(schedule)
        first_ten = schedule["attempts"][:10]
        for config in runner.CONFIGS:
            counts = [sum(item["order"][position] == config for item in first_ten)
                      for position in range(3)]
            self.assertLessEqual(max(counts) - min(counts), 1)
        schedule["attempts"][0]["order"] = ["recompute"]
        with self.assertRaises(runner.GateError):
            runner.validate_schedule(schedule)

    def test_effect_classification_requires_established_rate_regression(self):
        self.assertEqual(runner.classify_effect([-8.0, -2.0], [-0.04, 0.01]), "beneficial")
        self.assertEqual(
            runner.classify_effect([-8.0, -2.0], [-0.12, -0.06]),
            "latency-throughput tradeoff",
        )
        self.assertEqual(runner.classify_effect([-8.0, -2.0], [-0.12, -0.02]), "inconclusive")
        self.assertEqual(runner.classify_effect([1.0, 5.0], [-0.02, 0.01]), "not beneficial")

    def test_disk_environment_disables_hot_cpu_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = runner.server_environment("lmcache_disk", Path(tmp))
        self.assertEqual(env["LMCACHE_LOCAL_CPU"], "False")
        self.assertEqual(env["LMCACHE_SAVE_UNFULL_CHUNK"], "False")
        self.assertEqual(env["LMCACHE_EXTRA_CONFIG"], '{"use_odirect":true}')
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")


if __name__ == "__main__":
    unittest.main()
