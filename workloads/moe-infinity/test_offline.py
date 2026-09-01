from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "deps" / "MoE-Infinity"
REPOSITORY = ROOT.parents[1]
EXTENSION = REPOSITORY / "extension"
RUNNER_PATH = ROOT / "run_moe_head_to_head.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_moe_head_to_head", RUNNER_PATH)
assert RUNNER_SPEC and RUNNER_SPEC.loader
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


class InstrumentationTests(unittest.TestCase):
    def test_patch_exactly_matches_worktree(self) -> None:
        subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "--check",
                "--reverse",
                str(ROOT / "instrumentation.patch"),
            ],
            cwd=UPSTREAM,
            check=True,
        )

    def test_native_getter_body_is_load_only(self) -> None:
        source = (UPSTREAM / "core/parallel/expert_dispatcher.cpp").read_text()
        match = re.search(
            r"ExpertDispatcher::GetCacheCounts\(\) const \{(?P<body>.*?)\n\}",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertEqual(body.count(".load(std::memory_order_relaxed)"), 2)
        self.assertNotIn(".store(", body)
        self.assertNotIn("GetNodeVisitCounts", body)
        self.assertNotIn("IsTensorOffloaded", body)

    def test_native_binding_repeated_read_is_stable_without_cuda(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch  # noqa: F401
        from moe_infinity import _store

        dispatcher = _store.expert_dispatcher(1, 1, 0, 6, 1)
        first = dispatcher.get_cache_counts()
        second = dispatcher.get_cache_counts()
        self.assertEqual(first, (0, 0))
        self.assertEqual(second, first)

    def test_revision_stats_schema_and_derived_misses(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from moe_infinity.entrypoints.openai import revision_server

        class Dispatcher:
            @staticmethod
            def get_cache_counts() -> tuple[int, int]:
                return (13, 5)

            @staticmethod
            def get_hit_rate() -> float:
                raise AssertionError("mutating getter must not be called")

            @staticmethod
            def clear_expert_cache_counts() -> None:
                raise AssertionError("reset must not be called")

        class Offload:
            expert_dispatcher = Dispatcher()

            @staticmethod
            def get_exposed_fetch_seconds() -> float:
                return 1.25

        class Runtime:
            engine = Offload()

            @staticmethod
            def get_stats() -> dict[str, int]:
                return {
                    "total_generated_tokens": 512,
                    "num_steps": 64,
                    "kv_cache_num_blocks": 128,
                }

        old_engine = revision_server.server.engine
        revision_server.server.engine = Runtime()
        try:
            response = asyncio.run(revision_server.revision_stats())
        finally:
            revision_server.server.engine = old_engine

        payload = json.loads(response.body)
        self.assertEqual(
            payload,
            {
                "engine_generated_tokens": 512,
                "engine_steps": 64,
                "expert_cache_accesses": 13,
                "expert_cache_hits": 5,
                "expert_cache_misses": 8,
                "exposed_fetch_seconds_total": 1.25,
                "kv_cache_num_blocks": 128,
            },
        )

    def test_wrapper_avoids_forbidden_native_queries(self) -> None:
        source = (
            UPSTREAM
            / "moe_infinity/entrypoints/openai/revision_server.py"
        ).read_text()
        self.assertNotIn("get_hit_rate(", source)
        self.assertNotIn("clear_expert_cache_counts(", source)
        self.assertNotIn("is_tensor_offloaded(", source)
        self.assertNotIn("num_offloaded_experts", source)


class RowChunkingRepairTests(unittest.TestCase):
    def test_patch_exactly_matches_worktree(self) -> None:
        subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "--check",
                "--reverse",
                str(ROOT / "row-chunking.patch"),
            ],
            cwd=UPSTREAM,
            check=True,
        )

    def test_native_forward_chunks_without_changing_row_order(self) -> None:
        source = (UPSTREAM / "core/parallel/expert_module.cpp").read_text()
        start = source.index("torch::Tensor MoEMLP::forward(")
        end = source.index("void MoEMLP::ForwardHelper", start)
        body = source[start:end]

        self.assertNotIn("batch_size > kMaxTokens", body)
        self.assertIn("if (batch_size <= kMaxTokens)", body)
        self.assertIn("auto output = output_.clone()", body)
        self.assertIn("row_begin += kMaxTokens", body)
        self.assertIn(
            "std::min(kMaxTokens, batch_size - row_begin)", body
        )
        self.assertIn(
            "hidden_contiguous.narrow(0, row_begin, chunk_rows)", body
        )
        self.assertIn(
            "output.narrow(0, row_begin, chunk_rows).copy_(output_)", body
        )
        self.assertIn("resize_buffers(kMaxTokens)", body)

    def test_required_row_boundaries_are_covered(self) -> None:
        def chunks(rows: int) -> list[tuple[int, int]]:
            return [
                (begin, min(256, rows - begin))
                for begin in range(0, rows, 256)
            ]

        self.assertEqual(chunks(1), [(0, 1)])
        self.assertEqual(chunks(256), [(0, 256)])
        self.assertEqual(chunks(257), [(0, 256), (256, 1)])
        self.assertEqual(chunks(353), [(0, 256), (256, 97)])

    def test_gpu_numerical_gate_executes_repaired_moe_mlp(self) -> None:
        header = (UPSTREAM / "core/parallel/expert_module.h").read_text()
        source = (UPSTREAM / "core/parallel/expert_module.cpp").read_text()
        binding = (UPSTREAM / "core/python/py_archer_prefetch.cpp").read_text()
        script = (ROOT / "numerical_row_chunking_check.py").read_text()
        runner_source = RUNNER_PATH.read_text()
        self.assertIn("NumericalRowChunkingCheck", header)
        self.assertIn("module.forward(hidden, stream)", source)
        self.assertIn("row_begin += kMaxTokens", source)
        self.assertIn("torch::allclose(actual, reference, rtol, atol)", source)
        self.assertIn("row_chunking_numerical_check", binding)
        self.assertIn("ROWS = (1, 256, 257, 353)", script)
        self.assertIn("RTOL = 1.0e-2", script)
        self.assertIn("ATOL = 1.0e-2", script)
        self.assertIn("run_row_chunking_numerical_gate()", runner_source)


class DeterministicAccumulationRepairTests(unittest.TestCase):
    def test_patch_exactly_matches_worktree(self) -> None:
        subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "--check",
                "--reverse",
                str(ROOT / "deterministic-accumulation.patch"),
            ],
            cwd=UPSTREAM,
            check=True,
        )

    def test_worker_results_reduce_in_expert_order_after_stream_completion(self) -> None:
        source = (UPSTREAM / "core/parallel/expert_dispatcher.cpp").read_text()
        header = (UPSTREAM / "core/parallel/expert_dispatcher.h").read_text()
        binding = (UPSTREAM / "core/python/py_archer_prefetch.cpp").read_text()
        script = (ROOT / "numerical_row_chunking_check.py").read_text()
        guard = source.index("c10::cuda::CUDAStreamGuard guard(torch_stream);")
        gathered_input = source.index("auto token_mask = router_mask_.index")
        self.assertLess(guard, gathered_input)
        self.assertIn("CUDA_CHECK(\n      cudaStreamSynchronize", source)
        self.assertIn("pending_accumulations_.emplace_back", source)
        self.assertIn("std::sort(pending.begin(), pending.end()", source)
        self.assertIn("AccumulateInExpertOrder(final_hidden_states_", source)
        self.assertIn("worker_error_ = message.str()", source)
        self.assertIn("TORCH_CHECK(worker_error.empty(), worker_error)", source)
        self.assertIn("pending_accumulations_", header)
        self.assertIn("worker_error_", header)
        self.assertIn("deterministic_accumulation_check", binding)
        self.assertIn("arrival_orders", script)
        self.assertIn('accumulation.get("exact") is not True', RUNNER_PATH.read_text())


class FrozenWorkloadTests(unittest.TestCase):
    def test_manifest_names_generated_artifacts(self) -> None:
        manifest = json.loads((ROOT / "workload-manifest.json").read_text())
        for key in ("prompts", "schedule", "bootstrap"):
            path = ROOT / manifest[key]["path"]
            self.assertTrue(path.is_file())

    def test_prompt_roles_and_canonical_lengths(self) -> None:
        prompts = json.loads((ROOT / "prompts.json").read_text())
        records = prompts["records"]
        self.assertEqual(len(records), 9)
        self.assertEqual(records[0]["role"], "warmup")
        self.assertTrue(all(r["role"] == "measured" for r in records[1:]))
        self.assertTrue(all(len(r["prompt_token_ids"]) == 512 for r in records))
        self.assertTrue(all(r["prompt_token_count"] == 512 for r in records))

    def test_schedule_is_permutation_complete(self) -> None:
        schedule = json.loads((ROOT / "schedule.json").read_text())
        expected_configs = {
            "llama_ncmoe32",
            "llama_uvm",
            "gpubpf_host_stride_lfu",
            "moe_infinity_075",
        }
        self.assertEqual(len(schedule["attempts"]), 8)
        for number, attempt in enumerate(schedule["attempts"], start=1):
            self.assertEqual(attempt["attempt"], number)
            self.assertEqual(set(attempt["configuration_order"]), expected_configs)
            self.assertEqual(set(attempt["prompt_order"]), set(range(1, 9)))

    def test_bootstrap_indices_match_frozen_rng_api(self) -> None:
        actual = np.load(ROOT / "bootstrap-indices.npy", allow_pickle=False)
        expected = np.random.default_rng(1797).integers(
            0,
            5,
            size=(10000, 5),
            endpoint=False,
            dtype=np.int64,
        )
        self.assertTrue(np.array_equal(actual, expected))


class NoChecksumWorkflowTests(unittest.TestCase):
    def test_active_workflow_has_no_content_digest_logic(self) -> None:
        for path in (ROOT / "run_moe_head_to_head.py", ROOT / "freeze_workload.py"):
            source = path.read_text().lower()
            for forbidden in ("hashlib", "sha256", "sha-256", "checksum"):
                self.assertNotIn(forbidden, source)

        def visit(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    lowered = key.lower()
                    self.assertNotIn("hash", lowered)
                    self.assertNotIn("digest", lowered)
                    self.assertNotIn("checksum", lowered)
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for name in (
            "artifacts-current.json",
            "commands.json",
            "prompts.json",
            "workload-manifest.json",
        ):
            visit(json.loads((ROOT / name).read_text()))


class CombinedPolicyTests(unittest.TestCase):
    def test_single_struct_ops_object_has_all_six_hooks(self) -> None:
        source = (EXTENSION / "prefetch_stride_lfu.bpf.c").read_text()
        self.assertEqual(source.count('SEC(".struct_ops")'), 1)
        for hook in (
            "gpu_test_trigger",
            "gpu_page_prefetch",
            "gpu_page_prefetch_iter",
            "gpu_block_activate",
            "gpu_block_access",
            "gpu_evict_prepare",
        ):
            self.assertRegex(source, rf"\.{hook}\s*=\s*\(void \*\){hook}")

    def test_engagement_counters_are_incremented(self) -> None:
        source = (EXTENSION / "prefetch_stride_lfu.bpf.c").read_text()
        for counter in (
            "page_fault_calls",
            "stride_detections",
            "prefetches_issued",
            "lfu_activations",
            "lfu_accesses",
            "eviction_prepares",
        ):
            self.assertIn(f"&stats->{counter}", source)

    def test_loader_never_cleans_or_signals_unknown_state(self) -> None:
        source = (EXTENSION / "prefetch_stride_lfu.c").read_text()
        for forbidden in (
            "cleanup_struct_ops",
            "cleanup_old_struct_ops",
            "pkill",
            "killall",
            "system(",
            "bpf_map_delete_elem",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("bpf_link__destroy(struct_link)", source)
        self.assertIn("prefetch_stride_lfu_bpf__destroy(skel)", source)

    def test_loader_emits_owned_ids_and_final_snapshot(self) -> None:
        source = (EXTENSION / "prefetch_stride_lfu.c").read_text()
        for field in (
            "struct_link_id",
            "kprobe_link_id",
            "struct_map_id",
            "engagement_map_id",
            "config_map_id",
            "program_ids",
            "final_engagement",
        ):
            self.assertIn(field, source)

    def test_combined_policy_offline_build_exists(self) -> None:
        binary = EXTENSION / "prefetch_stride_lfu"
        bpf_object = EXTENSION / ".output/prefetch_stride_lfu.bpf.o"
        self.assertTrue(binary.is_file())
        self.assertTrue(os.access(binary, os.X_OK))
        self.assertTrue(bpf_object.is_file())


class RunnerTests(unittest.TestCase):
    def test_policy_ownership_accepts_kernel_without_link_enumeration(self) -> None:
        ready = {"pid": 42, "struct_map_id": 7, "struct_link_id": 7}
        inventory = {
            "maps": [{"id": 7, "type": "struct_ops", "pids": [{"pid": 42}]}],
            "links": [],
        }
        observed = runner.validate_policy_ownership(ready, inventory)
        self.assertEqual(observed["struct_map_id"], 7)
        self.assertEqual(observed["owner_pid"], 42)
        self.assertFalse(observed["link_enumerated"])

    def test_policy_ownership_rejects_wrong_map_owner(self) -> None:
        ready = {"pid": 42, "struct_map_id": 7, "struct_link_id": 7}
        inventory = {
            "maps": [{"id": 7, "type": "struct_ops", "pids": [{"pid": 99}]}],
            "links": [],
        }
        with self.assertRaisesRegex(runner.GateError, "PID ownership mismatch"):
            runner.validate_policy_ownership(ready, inventory)

    def test_runtime_continuity_rejects_replacement(self) -> None:
        expected = {"_store": {"path": "/tmp/store.so", "size": 1, "inode": 2}}
        runner.require_runtime_continuity(expected, expected.copy())
        changed = {"_store": {"path": "/tmp/store.so", "size": 2, "inode": 2}}
        with self.assertRaisesRegex(runner.GateError, "runtime files changed"):
            runner.require_runtime_continuity(expected, changed)

    def test_moe_revalidation_runtime_subset_excludes_unexecuted_cells(self) -> None:
        self.assertEqual(
            runner.MOE_REVALIDATION_RUNTIME_KEYS,
            {
                "python", "moe_engine", "moe_kv_cache", "moe_marlin",
                "moe_paged_attn", "moe_store", "moe_v4_fp4",
                "revision_server", "numerical_check", "sgl_common_ops",
            },
        )
        inventory = {key: {"path": key} for key in runner.MOE_REVALIDATION_RUNTIME_KEYS}
        inventory["llama_server"] = {"path": "new-llama"}
        selected = runner.select_runtime_files(
            inventory, runner.MOE_REVALIDATION_RUNTIME_KEYS
        )
        self.assertNotIn("llama_server", selected)
        self.assertEqual(set(selected), runner.MOE_REVALIDATION_RUNTIME_KEYS)

    def test_repaired_preflight_attempt_budget_is_fail_closed(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(runner, "REPAIRED_PREFLIGHT_ROOT", root):
                self.assertEqual(
                    runner.authorize_repaired_preflight_attempt(1),
                    root / "attempt-01",
                )
                first = root / "attempt-01"
                first.mkdir()
                (first / "preflight-result.json").write_text(
                    json.dumps(
                        {
                            "protocol": runner.PROTOCOL_ID,
                            "status": "failed",
                            "retry_allowed": False,
                        }
                    )
                )
                with self.assertRaisesRegex(runner.GateError, "deterministic"):
                    runner.authorize_repaired_preflight_attempt(2)
                (first / "preflight-result.json").write_text(
                    json.dumps({"status": "failed", "retry_allowed": True})
                )
                self.assertEqual(
                    runner.authorize_repaired_preflight_attempt(2),
                    root / "attempt-02",
                )
                with self.assertRaisesRegex(runner.GateError, "must be 1, 2, or 3"):
                    runner.authorize_repaired_preflight_attempt(4)

    def test_reviewed_protocol_change_can_use_next_attempt(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "attempt-01"
            first.mkdir()
            (first / "preflight-result.json").write_text(
                json.dumps(
                    {
                        "protocol": "proposal-3-revision-2",
                        "status": "failed",
                        "retry_allowed": False,
                    }
                )
            )
            with mock.patch.object(runner, "REPAIRED_PREFLIGHT_ROOT", root):
                self.assertEqual(
                    runner.authorize_repaired_preflight_attempt(2),
                    root / "attempt-02",
                )
                (first / "preflight-result.json").write_text(
                    json.dumps(
                        {
                            "protocol": "unreviewed-revision",
                            "status": "failed",
                            "retry_allowed": False,
                        }
                    )
                )
                with self.assertRaisesRegex(runner.GateError, "unchanged protocol"):
                    runner.authorize_repaired_preflight_attempt(2)

    def test_moe_trace_wrapper_inherits_frozen_taskset(self) -> None:
        argv = ["taskset", "-c", "0-7", "/venv/python", "-m", "server"]
        wrapped = runner.traced_moe_argv(argv, Path("/trace"))
        self.assertEqual(wrapped[:3], ["taskset", "-c", "0-7"])
        self.assertEqual(wrapped[3], "/usr/bin/strace")
        self.assertEqual(wrapped[-3:], ["/venv/python", "-m", "server"])
        with self.assertRaisesRegex(runner.GateError, "frozen CPU"):
            runner.traced_moe_argv(argv[3:], Path("/trace"))

    def test_timing_accepts_only_combined_revision5_preflight(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = {"moe_store": {"path": "/tmp/store.so", "size": 1}}
            valid = root / "completion-after-attempt-03"
            with mock.patch.object(runner, "PREFLIGHT_COMPLETION", valid):
                valid.mkdir()
                (valid / "admission.json").write_text(
                    json.dumps({"admitted": True, "runtime_files": runtime})
                )
                result = {
                    "protocol": runner.REVALIDATION_PROTOCOL_ID,
                    "status": "passed",
                    "row_chunking_numerical_gate": {"status": "passed"},
                    "runtime_files": runtime,
                    "configuration_order": list(runner.FROZEN_CORRECTNESS_ORDER),
                    "results": {name: {} for name in runner.CONFIGS},
                }
                (valid / "combined-preflight-result.json").write_text(json.dumps(result))
                loaded, expected = runner.load_repaired_preflight(valid)
                self.assertEqual(loaded["protocol"], runner.REVALIDATION_PROTOCOL_ID)
                self.assertEqual(expected, runtime)

                foreign = root / "foo"
                foreign.mkdir()
                with self.assertRaisesRegex(runner.GateError, "exactly the reviewed"):
                    runner.load_repaired_preflight(foreign)

                result["configuration_order"] = list(reversed(runner.FROZEN_CORRECTNESS_ORDER))
                (valid / "combined-preflight-result.json").write_text(json.dumps(result))
                with self.assertRaisesRegex(runner.GateError, "missing or inconsistent"):
                    runner.load_repaired_preflight(valid)

    def test_revision5_continuation_order_is_exact(self) -> None:
        schedule = json.loads(runner.SCHEDULE.read_text())
        self.assertEqual(
            tuple(schedule["attempts"][0]["configuration_order"]),
            runner.FROZEN_CORRECTNESS_ORDER,
        )
        self.assertEqual(
            runner.FROZEN_CORRECTNESS_ORDER[1:],
            ("gpubpf_host_stride_lfu", "llama_uvm", "llama_ncmoe32"),
        )

    def test_store_trace_classifies_construction_and_buffered_hydration(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_dir = root / "trace"
            offload = root / "offload"
            trace_dir.mkdir()
            offload.mkdir()
            lines = []
            for partition in range(7):
                path = offload / f"archer_param_{partition}"
                lines.extend((
                    f'openat(AT_FDCWD, "{path}", O_RDWR|O_CREAT|O_DIRECT, 0660) = 10',
                    f'openat(AT_FDCWD, "{path}", O_RDONLY) = 11',
                ))
            lines.extend((
                f'openat(AT_FDCWD, "{offload / "archer_index"}", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 12',
                f'openat(AT_FDCWD, "{offload / "name_id_map.json"}", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 13',
                f'openat(AT_FDCWD, "{offload / "tmpztlei0uk.tmp"}", O_RDWR|O_CREAT|O_EXCL, 0600) = 14',
            ))
            (trace_dir / "open.trace.1").write_text("\n".join(lines) + "\n")
            observed = runner.classify_moe_store_opens(trace_dir, offload)
            self.assertFalse(observed["steady_state_direct_read_claim"])
            self.assertEqual(set(observed["partitions"]), set(map(str, range(7))))
            self.assertEqual(observed["metadata_open_counts"]["name_id_map.json"], 1)

            with (trace_dir / "open.trace.1").open("a") as stream:
                stream.write(
                    f'openat(AT_FDCWD, "{offload / "unknown.bin"}", O_RDONLY) = 15\n'
                )
            with self.assertRaisesRegex(runner.GateError, "unclassified"):
                runner.classify_moe_store_opens(trace_dir, offload)

    def test_loaded_uvm_gate_accepts_exact_port_interface(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = root / "version"
            btf = root / "nvidia_uvm"
            version.write_text(runner.EXPECTED_DRIVER + "\n")
            btf.touch()
            members = "".join(
                f"\t'{name}' type_id=1 bits_offset={index * 64}\n"
                for index, name in enumerate((
                    "gpu_test_trigger", "gpu_page_prefetch",
                    "gpu_page_prefetch_iter", "gpu_block_activate",
                    "gpu_block_access", "gpu_evict_prepare",
                ))
            )
            dump = "[1] STRUCT 'gpu_mem_ops' size=48 vlen=6\n" + members + "".join(
                f"[2] FUNC '{name}' type_id=1 linkage=static\n"
                for name in (
                    "bpf_gpu_request_reorder",
                    "bpf_gpu_set_prefetch_region",
                )
            )
            with mock.patch.object(runner, "LOADED_UVM_VERSION", version), \
                 mock.patch.object(runner, "LOADED_UVM_BTF", btf), \
                 mock.patch.object(runner, "run_checked", return_value=dump):
                observed = runner.verify_loaded_uvm_interface()
        self.assertEqual(observed["version"], runner.EXPECTED_DRIVER)
        self.assertEqual(len(observed["gpu_mem_ops_members"]), 6)
        self.assertEqual(len(observed["required_kfuncs"]), 2)

    def test_loaded_uvm_gate_rejects_stock_or_incomplete_module(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = root / "version"
            btf = root / "nvidia_uvm"
            version.write_text(runner.EXPECTED_DRIVER + "\n")
            btf.touch()
            with mock.patch.object(runner, "LOADED_UVM_VERSION", version), \
                 mock.patch.object(runner, "LOADED_UVM_BTF", btf), \
                 mock.patch.object(runner, "run_checked", return_value=""):
                with self.assertRaisesRegex(runner.GateError, "six-member"):
                    runner.verify_loaded_uvm_interface()

    def test_uvm_and_gpubpf_server_commands_are_byte_identical(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            uvm_argv, uvm_cwd = runner.server_command("llama_uvm", 18080, attempt)
            bpf_argv, bpf_cwd = runner.server_command(
                "gpubpf_host_stride_lfu", 18080, attempt
            )
        self.assertEqual(uvm_argv, bpf_argv)
        self.assertEqual(uvm_cwd, bpf_cwd)
        self.assertEqual(
            runner.controlled_environment("llama_uvm"),
            runner.controlled_environment("gpubpf_host_stride_lfu"),
        )

    def test_controlled_environments_do_not_inherit_runtime_hooks(self) -> None:
        for config in runner.CONFIGS:
            env = runner.controlled_environment(config)
            for forbidden in (
                "PYTHONPATH",
                "LD_PRELOAD",
                "VLLM_USE_V1",
                "CUDA_LAUNCH_BLOCKING",
            ):
                self.assertNotIn(forbidden, env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_moe_command_preserves_relative_model_argument(self) -> None:
        argv, cwd = runner.server_command("moe_infinity_075", 18080, ROOT / "raw/x")
        self.assertEqual(argv[argv.index("--model") + 1], runner.HF_REVISION)
        self.assertEqual(cwd, runner.MODEL_VIEW_PARENT)
        self.assertEqual(argv[argv.index("--device-memory-ratio") + 1], "0.75")
        self.assertEqual(argv[argv.index("--kv-cache-ratio") + 1], "0")

    def test_frozen_schedule_drives_command_manifest(self) -> None:
        manifest = runner.frozen_commands(1, 18080, ROOT / "raw")
        schedule = json.loads((ROOT / "schedule.json").read_text())
        self.assertEqual(
            manifest["configuration_order"],
            schedule["attempts"][0]["configuration_order"],
        )
        self.assertEqual(set(manifest["configurations"]), set(runner.CONFIGS))
        self.assertIsNotNone(
            manifest["configurations"]["gpubpf_host_stride_lfu"]["policy_argv"]
        )
        for config in set(runner.CONFIGS) - {"gpubpf_host_stride_lfu"}:
            self.assertIsNone(manifest["configurations"][config]["policy_argv"])

    def test_model_view_is_exact_and_excludes_unrelated_serializations(self) -> None:
        observed = runner.verify_model_artifacts()
        members = set(observed["view_members"])
        self.assertEqual(len([x for x in members if x.endswith(".safetensors")]), 15)
        self.assertNotIn("original", members)
        self.assertNotIn("metal", members)

    def test_uvm_monitor_uses_v2_eviction_queue_not_prepare_proxy(self) -> None:
        source = (ROOT / "uvm_eviction_monitor.c").read_text()
        self.assertIn("UVM_EVENT_TYPE_EVICTION 14", source)
        self.assertIn("UVM_TOOLS_INIT_EVENT_TRACKER_V2 76", source)
        self.assertIn("UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS 58", source)
        self.assertNotIn("gpu_evict_prepare", source)
        self.assertTrue((ROOT / "uvm_eviction_monitor").is_file())

    def test_request_payloads_share_frozen_common_fields(self) -> None:
        token_ids = list(range(512))
        llama = runner.completion_payload("llama_uvm", token_ids, True)
        moe = runner.completion_payload("moe_infinity_075", token_ids, True)
        for key, value in {
            "model": "gpt-oss-120b",
            "prompt": token_ids,
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": [],
            "stream": True,
        }.items():
            self.assertEqual(llama[key], value)
            self.assertEqual(moe[key], value)
        self.assertEqual(
            {key: llama[key] for key in ("cache_prompt", "return_tokens")},
            {"cache_prompt": False, "return_tokens": True},
        )
        self.assertEqual(
            {key: moe[key] for key in ("n", "best_of", "echo")},
            {"n": 1, "best_of": 1, "echo": False},
        )

    def test_completion_gate_requires_exact_512_plus_64_accounting(self) -> None:
        response = {
            "choices": [{"text": "answer", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 512, "completion_tokens": 64},
        }
        self.assertEqual(
            runner.validate_completion_response(response, 512)["finish_reason"],
            "length",
        )
        for field, bad in (("prompt_tokens", 511), ("completion_tokens", 63)):
            changed = json.loads(json.dumps(response))
            changed["usage"][field] = bad
            with self.assertRaises(runner.GateError):
                runner.validate_completion_response(changed, 512)

    def test_counter_delta_rejects_non_monotonic_totals(self) -> None:
        self.assertEqual(
            runner.counter_delta({"x": 2, "y": 9}, {"x": 5, "y": 9}, ("x", "y")),
            {"x": 3, "y": 0},
        )
        with self.assertRaises(runner.GateError):
            runner.counter_delta({"x": 2}, {"x": 1}, ("x",))

    def test_runtime_cleanup_has_no_generic_process_or_policy_kill(self) -> None:
        source = RUNNER_PATH.read_text()
        for forbidden in ("pkill", "killall", "cleanup_old_struct_ops"):
            self.assertNotIn(forbidden, source)
        self.assertIn("os.kill(process.pid, signal.SIGINT)", source)
        self.assertIn("os.killpg(pgid, signal.SIGKILL)", source)
        self.assertIn("ready[\"struct_link_id\"]", source)

    def test_frozen_block_bootstrap_estimator(self) -> None:
        blocks = []
        for _ in range(5):
            results = {}
            for config, throughput, ttft in (
                ("llama_ncmoe32", 1.1, 11.0),
                ("llama_uvm", 1.2, 10.5),
                ("gpubpf_host_stride_lfu", 2.0, 10.0),
                ("moe_infinity_075", 1.0, 12.0),
            ):
                results[config] = {
                    "output_throughput_tokens_per_s": throughput,
                    "requests": [{"ttft_ms": ttft} for _ in range(8)],
                }
            blocks.append({"results": results})
        analysis = runner.analyze_valid_blocks(blocks)
        ratio = analysis["ratios"]["gpubpf_host_stride_lfu"]
        self.assertAlmostEqual(ratio["geometric_mean_ratio_vs_moe"], 2.0)
        self.assertTrue(all(abs(value - 2.0) < 1e-12 for value in ratio["ci95"]))
        self.assertEqual(analysis["outcome"], "higher output-token throughput")
        self.assertAlmostEqual(
            analysis["ttft_gpubpf_minus_moe_ms"]["mean_of_block_medians"], -2.0
        )

    def test_gpu_telemetry_rejects_active_throttle_reason(self) -> None:
        import tempfile
        header = (
            "timestamp, memory.used [MiB], temperature.gpu, power.draw [W], "
            "clocks.current.sm [MHz], clocks.current.memory [MHz], "
            "clocks_event_reasons.sw_power_cap, clocks_event_reasons.hw_slowdown\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.csv"
            path.write_text(header + "t0, 10, 40, 100, 2000, 1000, Not Active, Not Active\n")
            self.assertFalse(runner.validate_gpu_telemetry(path)["throttled"])
            path.write_text(header + "t0, 10, 40, 100, 2000, 1000, Active, Not Active\n")
            with self.assertRaises(runner.GateError):
                runner.validate_gpu_telemetry(path)


if __name__ == "__main__":
    unittest.main()
