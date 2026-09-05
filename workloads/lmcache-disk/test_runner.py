#!/usr/bin/env python3
"""CPU-only structural tests for the LMCache revision harness."""

import importlib.util
from contextlib import nullcontext
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).with_name("run_lmcache_disk.py")
SPEC = importlib.util.spec_from_file_location("run_lmcache_disk", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PLACEMENT = dict(boot_id="11111111-2222-3333-4444-555555555555", worker_cpu_affinity=list(range(8, 16)), telemetry_cpu=16)


class HarnessTests(unittest.TestCase):
    def test_driver_selection_is_exact_and_explicit(self):
        runner.legacy.validate_driver("610.43.02")
        runner.legacy.validate_driver("575.57.08", "575.57.08")
        for observed, expected in (("575.57.08", "610.43.02"),
                                   ("610.43.02", "575.57.08"),
                                   ("575.99", "575.99")):
            with self.subTest(observed=observed, expected=expected):
                with self.assertRaises(runner.GateError):
                    runner.legacy.validate_driver(observed, expected)
        with self.assertRaisesRegex(runner.GateError, "driver"):
            runner._validate_recorded_environment({"gpu": {"driver": "575.57.08"}})

    def test_explicit_driver_is_passed_to_admission_and_saved_observations(self):
        observations = {"model_path": "/model", "expected_driver": "575.57.08",
                        "gpu": {"driver": "575.57.08"}}
        with (patch.object(runner.legacy, "admission", return_value=observations) as admit,
              patch.object(runner, "managed_cell", return_value=nullcontext(PLACEMENT)),
              patch.object(runner, "load_prompts", return_value={"prefixes": [{}]}),
              patch.object(runner.legacy, "run_config", return_value={}) as run):
            runner.run_cell("lmcache_disk", Path("/output"), 18080, False, 1, "575.57.08")
        admit.assert_called_once_with(18080, require_model=True, storage_path=Path("/output"),
                                      expected_driver="575.57.08")
        self.assertEqual(run.call_args.kwargs["recorded_environment"], observations)
        self.assertEqual(run.call_args.kwargs["expected_driver"], "575.57.08")

    def test_formal_rejects_smoke_and_mixed_driver_cells(self):
        schedule = json.loads(runner.SCHEDULE.read_text())["attempts"]
        groups = {row["attempt"]: {config: (position, Path(f"/{row['attempt']}/{config}"))
                                  for position, config in enumerate(row["order"])}
                  for row in schedule[:2]}
        for defect in ("prefix", "within-block driver", "across-block driver", "within-block boot", "across-block boot"):
            def checked(path):
                block, config = int(path.parent.name), path.name
                driver = ("610.43.02" if
                          (defect == "within-block driver" and config == "recompute") or
                          (defect == "across-block driver" and block == 1) else "575.57.08")
                return {"result": {"prefix_count": 1 if defect == "prefix" else 8},
                        "environment": {"gpu": {"driver": driver}, "boot_id": (
                            "other" if (defect == "within-block boot" and config == "recompute") or
                            (defect == "across-block boot" and block == 1) else PLACEMENT["boot_id"])}}
            with (self.subTest(defect=defect),
                  patch.object(runner, "_attempt_cells", return_value=groups),
                  patch.object(runner, "_validate_execution_sequence", return_value=([0, 1], [])),
                  patch.object(runner, "validate_cell", side_effect=checked),
                  patch.object(runner.legacy, "output_texts", return_value={"same": "text"}),
                  patch.object(runner.legacy, "atomic_write_json") as write):
                with self.assertRaisesRegex(runner.GateError, "eight prefixes|mix driver|mix boot"):
                    runner.analyze(Path("/unused"))
                write.assert_not_called()

    def test_response_gate_requires_captured_vllm_request_id(self):
        response = {
            "request_header": "lmc-p0-cold",
            "engine_request_id": "cmpl-lmc-p0-cold",
            "status": 200,
            "input_tokens": 1549,
            "usage": {"prompt_tokens": 1549, "completion_tokens": runner.OUTPUT_TOKENS},
            "text": "ok",
            "ttft_ms": 1.0,
            "e2e_ms": 2.0,
        }
        runner._validate_response(response, 1549, "lmc-p0-cold")
        response["engine_request_id"] = "cmpl-lmc-p0-cold-0-a1b2c3d4"
        with self.assertRaises(runner.GateError):
            runner._validate_response(response, 1549, "lmc-p0-cold")

    def test_prefix_limited_cell_slices_only_after_full_prompt_load(self):
        prompts = {"prefixes": [{"index": index} for index in range(runner.PREFIXES)]}
        with (
            patch.object(runner, "managed_cell", return_value=nullcontext(PLACEMENT)),
            patch.object(runner, "inspect_environment", return_value={"model_path": "/model"}),
            patch.object(runner, "load_prompts", return_value=prompts) as load,
            patch.object(runner.legacy, "run_config", return_value={"status": "ok"}) as run,
        ):
            self.assertEqual(
                runner.run_cell("lmcache_disk", Path("/output"), 18080, False, 1),
                {"status": "ok"},
            )
        load.assert_called_once_with(runner.PROMPTS)
        self.assertEqual(len(run.call_args.args[2]["prefixes"]), 1)
        with self.assertRaises(runner.GateError):
            runner.run_cell("lmcache_disk", Path("/output"), 18080, False, 0)

    def test_all_cells_share_runtime_repair(self):
        with patch.dict(os.environ, {"TRITON_PTXAS_BLACKWELL_PATH": "/wrong/compiler",
                                     "TRITON_PTXAS_PATH": "/also/wrong"}):
            for config in runner.CONFIGS:
                old_env = runner.server_environment(config, Path("/cache"))
                self.assertEqual(old_env, runner.server_environment(config, Path("/cache"), "610.43.02"))
                self.assertNotIn("TRITON_PTXAS_BLACKWELL_PATH", old_env)
                self.assertNotIn("TRITON_PTXAS_PATH", old_env)
                new_env = runner.server_environment(config, Path("/cache"), "575.57.08")
                self.assertEqual(new_env, {**old_env, "TRITON_PTXAS_BLACKWELL_PATH":
                                          str(runner.legacy.TRITON_PTXAS_575)})
                self.assertEqual(new_env["VLLM_USE_DEEP_GEMM"], "0")
                if config == "recompute":
                    self.assertNotIn("LMCACHE_USE_GPU_CONNECTOR_V3", new_env)
                else:
                    self.assertEqual(new_env["LMCACHE_USE_GPU_CONNECTOR_V3"], "True")
                argv = runner.server_argv(config, Path("/model"), 18080)
                self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.98")

    def test_compiler_pin_reaches_actual_server_launch_for_all_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for driver in runner.legacy.EXPERIMENT_DRIVERS:
                for config in runner.CONFIGS:
                    with (self.subTest(driver=driver, config=config),
                          patch.object(runner.legacy.subprocess, "Popen") as start):
                        _, log, _, _ = runner.start_server(
                            config, Path("/model"), root / "cache", 18080,
                            root / f"{driver}-{config}.log", expected_driver=driver)
                    log.close()
                    self.assertEqual(start.call_args.kwargs["env"],
                                     runner.server_environment(config, root / "cache", driver))

    def test_failed_start_preserves_explicit_compiler_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            observations = {"gpu": {"driver": "575.57.08"}, "expected_driver": "575.57.08"}
            def failed_start(*args, **kwargs):
                saved = json.loads((output / "environment.json").read_text())
                self.assertEqual(saved["server_environment"], runner.server_environment(
                    "recompute", output / "cache", "575.57.08"))
                self.assertEqual(kwargs["expected_driver"], "575.57.08")
                raise RuntimeError("simulated server startup failure")
            with patch.object(runner.legacy, "start_server", side_effect=failed_start):
                with self.assertRaisesRegex(RuntimeError, "simulated server startup failure"):
                    runner.legacy.run_config(
                        "recompute", output, {"prefixes": [{}]}, 18080, Path("/model"),
                        recorded_environment=observations, expected_driver="575.57.08")
            self.assertNotIn("server_environment", observations)

    def test_575_recorded_compiler_pin_and_inventory_are_required(self):
        frozen = runner.legacy.load_artifacts()
        model = Path("/model") / runner.MODEL_REVISION
        model_names = ["config.json", "model.safetensors.index.json"] + [
            f"model-{index:05d}-of-00007.safetensors" for index in range(1, 8)]
        record = {
            **PLACEMENT, "expected_driver": "575.57.08",
            "gpu": {"driver": "575.57.08", "compute_apps": [], "memory_used_mib": 0},
            "lmcache_source": {"commit": runner.legacy.LMCACHE_COMMIT, "path": str(runner.legacy.LMCACHE_REPO)},
            "runtime_imports": {
                "lmcache_version": runner.EXPECTED_LMCACHE_VERSION,
                "vllm_version": runner.EXPECTED_VLLM_VERSION,
                "modules": {name: {"path": path, "bytes": 1} for name, path in frozen["runtime_import_paths"].items()},
                "dependency_lines": (runner.HERE / frozen["environment_freeze"]["relative_path"]).read_text().splitlines(),
            },
            "storage": {"mount": {"filesystems": [{"source": runner.legacy.EXPECTED_MOUNT_SOURCE, "fstype": "ext4"}]},
                        "free_bytes": 100 * 1024**3},
            "model_path": str(model), "model_revision": runner.MODEL_REVISION,
            "model_artifacts": [{"name": name, "path": str(model / name), "bytes": 1} for name in model_names],
            "workload_artifacts": {name: {"path": str(path.absolute())} for name, path in (
                ("dataset", runner.legacy.DATASET), ("prompts", runner.PROMPTS), ("schedule", runner.SCHEDULE))},
            "triton_ptxas": {"path": str(runner.legacy.TRITON_PTXAS_575), "bytes": 1,
                             "version_output": "Cuda compilation tools, release 12.9, V12.9.86\n"},
            "server_environment": runner.server_environment("recompute", Path("/cache"), "575.57.08"),
        }
        runner._validate_recorded_environment(record)
        for defect in ("missing inventory", "wrong path", "wrong version", "empty binary", "missing pin", "wrong pin"):
            bad = json.loads(json.dumps(record))
            if defect == "missing inventory":
                del bad["triton_ptxas"]
            elif defect == "wrong path":
                bad["triton_ptxas"]["path"] = "/wrong/ptxas"
            elif defect == "wrong version":
                bad["triton_ptxas"]["version_output"] = "release 13.1, V13.1.80"
            elif defect == "empty binary":
                bad["triton_ptxas"]["bytes"] = 0
            elif defect == "missing pin":
                del bad["server_environment"]
            else:
                bad["server_environment"]["TRITON_PTXAS_BLACKWELL_PATH"] = "/wrong/ptxas"
            with self.subTest(defect=defect), self.assertRaisesRegex(runner.GateError, "compiler pin and inventory"):
                runner._validate_recorded_environment(bad)
        # Older 610 evidence need not retroactively acquire 575-only fields.
        record["gpu"]["driver"] = record["expected_driver"] = "610.43.02"
        del record["triton_ptxas"], record["server_environment"]
        runner._validate_recorded_environment(record)

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
            self.assertEqual(launch[:3], ["/usr/bin/taskset", "-c", "8-15"])

    def test_owned_diagnostics_and_exited_server_cleanup(self):
        process = Mock()
        process.communicate.side_effect = subprocess.TimeoutExpired(["/diagnostic"], 1)
        with (patch.object(runner.legacy.subprocess, "Popen", return_value=process) as start,
              patch.object(runner.shared, "stop_owned") as stop):
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.legacy.run_checked(["/diagnostic"], timeout=1)
        self.assertTrue(start.call_args.kwargs["start_new_session"])
        stop.assert_called_once_with(process)
        log = Mock()
        process.poll.return_value = 0
        with patch.object(runner.shared, "stop_owned") as stop:
            runner.legacy.stop_owned_server(process, log)
        stop.assert_called_once_with(process)
        log.close.assert_called_once()

    def test_busy_lease_prevents_inspection_and_output_creation(self):
        with (tempfile.TemporaryDirectory() as tmp,
              patch.object(runner.shared, "Leases", side_effect=BlockingIOError("busy")),
              patch.object(runner, "inspect_environment") as inspect):
            output = Path(tmp) / "cell"
            with self.assertRaises(BlockingIOError):
                runner.run_cell("recompute", output, 18080, False)
            self.assertFalse(output.exists())
        inspect.assert_not_called()

    def test_managed_cell_safety_is_independent_of_a_result_file(self):
        def snapshot():
            return dict(timestamp_ns=time.time_ns(), power_limit_service="active", power_limit_w=400.0,
                        gpu=dict(driver="575.57.08", compute_apps=[], memory_used_mib=0, utilization_gpu_percent=0),
                        uvm_refcount=0, struct_ops=dict(maps=[], links=[]), dmesg_abnormal=[], journal_abnormal=[], xids=[])
        for defect in (None, "body failure", "monitor died", "post safety", "kernel error"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "cell"
                lease, gpu_monitor, kernel_monitor, stream = Mock(), Mock(), Mock(), Mock()
                gpu_monitor.poll.return_value = 0 if defect == "monitor died" else None
                kernel_monitor.poll.return_value = None
                def start_monitor(directory):
                    path = directory / "gpu-telemetry.csv"
                    headers = "timestamp, memory.used, temperature.gpu, power.draw, clocks.current.sm, clocks.current.memory, clocks_event_reasons.sw_power_cap, clocks_event_reasons.hw_slowdown\n"
                    path.write_text(headers + "2026/09/03 10:00:00.000, 100 MiB, 40, 80 W, 1000 MHz, 1000 MHz, Not Active, Not Active\n")
                    return gpu_monitor, stream, path
                with (patch.dict(os.environ, {}, clear=True),
                      patch.object(runner.shared, "Leases", return_value=lease),
                      patch.object(runner.os, "sched_getaffinity", return_value=set(range(8, 17))),
                      patch.object(runner.safety, "safety_snapshot", side_effect=snapshot),
                      patch.object(runner.safety, "start_gpu_telemetry", side_effect=start_monitor),
                      patch.object(runner.safety, "wait_for_post_server_safety", side_effect=(RuntimeError("new Xid") if defect == "post safety" else lambda before: snapshot())),
                      patch.object(runner.subprocess, "Popen", return_value=kernel_monitor),
                      patch.object(runner.shared, "stop_owned") as stop):
                    def run():
                        with runner.managed_cell(output, "575.57.08") as execution:
                            environment = {**PLACEMENT, "boot_id": execution["boot_id"], "expected_driver": "575.57.08", "timestamp_ns": time.time_ns()}
                            (output / "result.json").write_text('{"unit_test_only":true}\n')
                            if defect == "kernel error":
                                (output / "kernel-follow.log").write_text("NVRM: Xid test fixture\n")
                            if defect == "body failure":
                                raise RuntimeError("server failed")
                        return environment
                    if defect is None:
                        environment = run()
                    else:
                        with self.assertRaises(RuntimeError):
                            run()
                lease.close.assert_called_once()
                self.assertEqual(stop.call_count, 2)
                record = json.loads((output / "execution.json").read_text())
                self.assertEqual(record["status"], "passed" if defect is None else "failed")
                if defect is None:
                    self.assertEqual(runner.validate_execution(output, environment), record)
                    for field, bad in (("status", "failed"), ("final_boot_id", "different"),
                                       ("monitors_alive_before_stop", [False, True]), ("telemetry_cpu", 8)):
                        (output / "execution.json").write_text(json.dumps({**record, field: bad}))
                        with self.assertRaises(runner.GateError):
                            runner.validate_execution(output, environment)

    def test_request_scoped_log_parser(self):
        request_id = "cmpl-lmc-p3-warm"
        runtime_id = request_id + "-0-a1b2c3d4"
        log = "\n".join(
            [
                f"Reqid: {runtime_id}, Total tokens 1550, Inference Engine computed tokens: 0, LMCache hit tokens: 1536, need to load: 1536",
                f"[req_id={runtime_id}] Retrieved 1536 out of 1536 required tokens (from 1550 total tokens).",
                f"[req_id={runtime_id}] Stored 1536 out of total 1550 tokens.",
            ]
        )
        self.assertEqual(
            runner.request_log_values(log, request_id),
            {"runtime_ids": [runtime_id], "request_totals": [1550], "hits": [1536],
             "stores": [[1536, 1550]], "retrieved": [[1536, 1536, 1550]]},
        )

    def test_retrieval_parser_preserves_denominators(self):
        request_id = "cmpl-lmc-p0-warm"
        runtime_id = request_id + "-0-a1b2c3d4"
        log = (
            f"Reqid: {runtime_id}, Total tokens 1549, LMCache hit tokens: 1536\n"
            f"[req_id={runtime_id}] Retrieved 1536 out of 512 required tokens "
            "(from 1549 total tokens)."
        )
        values = runner.request_log_values(log, request_id)
        self.assertEqual(values["request_totals"], [1549])
        self.assertEqual(values["retrieved"], [[1536, 512, 1549]])

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
        cold_id = "cmpl-lmc-p0-cold"
        warm_id = "cmpl-lmc-p0-warm"
        cold_runtime_id = cold_id + "-0-a1b2c3d4"
        warm_runtime_id = warm_id + "-0-a1b2c3d4"
        log = "\n".join(
            [
                "Creating LMCacheEngine with config: {'use_gpu_connector_v3': True}",
                "init kv cache pointers success in VLLMPagedMemGPUConnectorV3",
                "LMCache initialized with version 0.5.4, vllm version 0.27.1+cu129",
                f"Reqid: {cold_runtime_id}, Total tokens 1549, LMCache hit tokens: 0",
                f"[req_id={cold_runtime_id}] Stored 1536 out of total 1536 tokens",
                f"Reqid: {warm_runtime_id}, Total tokens 1549, LMCache hit tokens: 512",
                f"Reqid: {warm_runtime_id}, Total tokens 1549, LMCache hit tokens: 1536",
                f"[req_id={warm_runtime_id}] Retrieved 512 out of 1536 required tokens",
                f"[req_id={warm_runtime_id}] Retrieved 1536 out of 1536 required tokens",
            ]
        )
        observations = [{"expected_hit_tokens": 1536,
                         "cold": {"engine_request_id": cold_id,
                                  "usage": {"prompt_tokens": 1549}},
                         "warm": {"engine_request_id": warm_id,
                                  "usage": {"prompt_tokens": 1549}}}]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(runner.GateError):
                runner.validate_log("lmcache_cpu", log, observations, Path(tmp))

    def test_frozen_prompt_expectations(self):
        prompts = runner.load_prompts(runner.PROMPTS)
        self.assertEqual(prompts["schema"], 3)
        self.assertEqual(len(prompts["prefixes"]), 8)
        self.assertTrue(all(x["expected_hit_tokens"] == 1536 for x in prompts["prefixes"]))

    def test_runner_uses_semantic_evidence_without_content_fingerprints(self):
        source = MODULE_PATH.read_text() + runner.PRIMITIVES_PATH.read_text()
        forbidden = ("hashlib", "sha256", "checksum", "digest", "output_hash")
        self.assertFalse(any(term in source.lower() for term in forbidden))
        for forbidden_control in ("PREFLIGHT-PASS", "SMOKE-PASS", "RUN-COMPLETE", "--resume", "--approval"):
            self.assertNotIn(forbidden_control, source)

    def test_fatal_patterns_cover_lmcache_allocation_wording(self):
        for log in (
            "Memory allocation failed while staging a disk chunk",
            "allocation failed for local CPU staging",
            "3 evictions occurred",
        ):
            self.assertTrue(any(runner.legacy.re.search(pattern, log, runner.legacy.re.I)
                                for pattern in runner.legacy.FATAL_LOG_PATTERNS))
        optional_tuning = "Using default MoE config. Config file not found at optional.json"
        self.assertFalse(any(runner.legacy.re.search(pattern, optional_tuning, runner.legacy.re.I)
                             for pattern in runner.legacy.FATAL_LOG_PATTERNS))
        fatal_missing = "Required model File not found"
        self.assertTrue(any(runner.legacy.re.search(pattern, fatal_missing, runner.legacy.re.I)
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

    def test_runtime_log_accepts_public_base_versions(self):
        runtime_line = (
            "Creating LMCacheEngine with config: {'use_gpu_connector_v3': True}\n"
            "init kv cache pointers success in VLLMPagedMemGPUConnectorV3\n"
            "LMCache initialized for role worker with version 0.5.4-gsource, "
            "vllm version 0.27.1, lmcache cache_engine metadata: None"
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner.legacy, "sync_and_verify_disk",
                              return_value={"files": 0, "bytes": 0}):
                value = runner.validate_log("lmcache_cpu", runtime_line, [], Path(tmp))
        self.assertEqual(value["request_evidence"], {})
        self.assertEqual(value["gpu_connector_v3"], {
            "config_enabled": True, "connector_initialized": True,
        })

    def test_runtime_log_rejects_v2_or_uninitialized_connector(self):
        init = "LMCache initialized with version 0.5.4, vllm version 0.27.1"
        with tempfile.TemporaryDirectory() as tmp:
            for evidence in (
                "Creating LMCacheEngine with config: {'use_gpu_connector_v3': False}\n" + init,
                "Creating LMCacheEngine with config: {'use_gpu_connector_v3': True}\n" + init,
            ):
                with self.subTest(evidence=evidence):
                    with self.assertRaises(runner.GateError):
                        runner.validate_log("lmcache_cpu", evidence, [], Path(tmp))

    def test_store_state_is_recomputed_for_each_prefix(self):
        request_evidence = {
            "runtime_ids": ["cmpl-lmc-p0-cold-0-a1b2c3d4"],
            "request_totals": [1540],
            "hits": [0],
            "stores": [[1536, 1536]],
            "retrieved": [],
        }
        self.assertEqual(
            runner._expected_store_state("recompute", 3, request_evidence),
            {"files": 0, "bytes": 0, "durability": "not applicable"},
        )
        self.assertEqual(
            runner._expected_store_state("lmcache_cpu", 3, request_evidence)["request_log"],
            request_evidence,
        )
        first = runner._expected_store_state("lmcache_disk", 0, request_evidence)
        last = runner._expected_store_state("lmcache_disk", 7, request_evidence)
        self.assertEqual(first["files"], runner.CHUNKS_PER_PREFIX)
        self.assertEqual(first["bytes"], runner.CHUNKS_PER_PREFIX * runner.KV_CHUNK_BYTES)
        self.assertEqual(last["files"], 8 * runner.CHUNKS_PER_PREFIX)
        self.assertEqual(last["bytes"], 8 * runner.CHUNKS_PER_PREFIX * runner.KV_CHUNK_BYTES)
        self.assertEqual(last["per_file_bytes"], [runner.KV_CHUNK_BYTES])
        self.assertEqual(last["request_log"], request_evidence)

    def test_execution_sequence_requires_contiguous_timestamped_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "attempt-00"
            cell = attempt / "position-0-recompute"
            cell.mkdir(parents=True)
            (cell / "environment.json").write_text('{"timestamp_ns": 100}\n')
            (attempt / "failure.md").write_text("server failed before result\n")
            observed, timeline = runner._validate_execution_sequence(root, {})
            self.assertEqual(observed, [0])
            self.assertEqual(timeline, [(0, 0, 100)])
            (attempt / "failure.md").unlink()
            with self.assertRaises(runner.GateError):
                runner._validate_execution_sequence(root, {})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for attempt_number in (0, 2):
                attempt = root / f"attempt-{attempt_number:02d}"
                cell = attempt / "position-0-recompute"
                cell.mkdir(parents=True)
                (cell / "environment.json").write_text(
                    json.dumps({"timestamp_ns": 100 + attempt_number}) + "\n"
                )
                (attempt / "failure.md").write_text("recorded failure\n")
            with self.assertRaises(runner.GateError):
                runner._validate_execution_sequence(root, {})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "attempt-00"
            for position, config, timestamp in (
                (0, "recompute", 200),
                (1, "lmcache_cpu", 100),
            ):
                cell = attempt / f"position-{position}-{config}"
                cell.mkdir(parents=True)
                (cell / "environment.json").write_text(
                    json.dumps({"timestamp_ns": timestamp}) + "\n"
                )
            (attempt / "failure.md").write_text("recorded failure\n")
            with self.assertRaises(runner.GateError):
                runner._validate_execution_sequence(root, {})


if __name__ == "__main__":
    unittest.main()
