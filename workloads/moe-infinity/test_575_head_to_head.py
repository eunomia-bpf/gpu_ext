"""CPU-only coverage for the current 575 continuation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_575_head_to_head as current
import run_moe_head_to_head as base
import audit_575_results as audit


class CurrentStackTests(unittest.TestCase):
    def test_lease_uses_existing_read_only_coordinator_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = tuple(Path(temporary) / name for name in ("gpu.lock", "ops.lock"))
            for path in paths:
                path.write_text("")
                path.chmod(0o444)
            try:
                with patch.object(base, "LEASE_PATHS", paths):
                    lease = base.LeaseSet.acquire()
                    try:
                        self.assertEqual(len(lease.files), 2)
                        self.assertTrue(all(stream.readable() for stream in lease.files))
                        self.assertTrue(all(not stream.writable() for stream in lease.files))
                    finally:
                        lease.close()
            finally:
                for path in paths:
                    path.chmod(0o600)

    def test_raw_sse_audit_rejects_saved_text_and_usage_drift(self):
        payload = {"choices": [{"text": "test", "finish_reason": "length"}],
                   "usage": {"prompt_tokens": 512, "completion_tokens": 64}}
        raw = b"data: " + json.dumps(payload).encode() + b"\n\ndata: [DONE]\n\n"
        request = {"text": "test", "usage": payload["usage"], "raw_sse_bytes": len(raw),
                   "frames": [{}, {}], "start_ns": 0, "first_text_ns": 1,
                   "done_ns": 2, "eof_ns": 3, "ttft_ms": 0.000001, "e2e_ms": 0.000003}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.sse"
            path.write_bytes(raw)
            audit.audit_stream(path, request, "test", "llama_uvm")
            with self.assertRaises(base.GateError):
                audit.audit_stream(path, request, "altered", "llama_uvm")
            path.write_bytes(raw.replace(b'"completion_tokens": 64', b'"completion_tokens": 63'))
            with self.assertRaises(base.GateError):
                audit.audit_stream(path, request, "test", "llama_uvm")

    def test_correctness_resume_rechecks_raw_responses_and_saved_result(self):
        response = {"choices": [{"text": "test", "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 512, "completion_tokens": 64}}
        item = base.validate_completion_response(response, 512)
        result = {"passes": [[item] * 8, [item] * 8], "goldens": ["test"] * 8}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            base.atomic_write_json(path / "result.json", result)
            base.atomic_write_json(path / "safety.json", {"passed": True, "after": {}})
            base.atomic_write_json(path / "warmup.json", response)
            for part in (1, 2):
                for prompt in range(1, 9):
                    base.atomic_write_json(path / f"smoke-pass{part}-prompt{prompt}.json", response)
            with patch.object(base, "validate_pre_server_safety") as safety:
                current.validate_saved_correctness(path, result)
                safety.assert_called_once_with({})
                altered = copy.deepcopy(response)
                altered["usage"]["completion_tokens"] = 63
                base.atomic_write_json(path / "smoke-pass2-prompt2.json", altered)
                with self.assertRaises(base.GateError):
                    current.validate_saved_correctness(path, result)

    def test_current_moe_pins_blackwell_jit_compiler_and_cache(self):
        env = base.controlled_environment("moe_infinity_075", cuda129_triton=True)
        self.assertEqual(env["TRITON_PTXAS_BLACKWELL_PATH"], "/usr/local/cuda-12.9/bin/ptxas")
        self.assertEqual(env["TRITON_PTXAS_PATH"], "/usr/local/cuda-12.9/bin/ptxas")
        self.assertTrue(env["TRITON_CACHE_DIR"].endswith("deps/triton-cache-cuda129"))
        for config in base.CONFIGS[:-1]:
            self.assertEqual(base.controlled_environment(config, cuda129_triton=True),
                             base.controlled_environment(config))

    def test_moe_cpu_offload_does_not_claim_steady_state_disk_reads(self):
        revision = dict(engine_generated_tokens=0, engine_steps=0, expert_cache_accesses=0,
                        expert_cache_hits=0, expert_cache_misses=0, kv_cache_num_blocks=128)
        before = {"process_io": {"read_bytes": 0, "cpu_time_s": 0},
                  "moe": {"revision": revision,
                          "metrics": {"moe_tokens_generated_total": 0,
                                      "moe_engine_steps_total": 0}}}
        after = copy.deepcopy(before)
        after["moe"]["revision"].update(engine_generated_tokens=512, engine_steps=512,
                                        expert_cache_accesses=100, expert_cache_hits=75,
                                        expert_cache_misses=25)
        after["moe"]["metrics"].update(moe_tokens_generated_total=512, moe_engine_steps_total=512)
        result = base.validate_measured_engagement("moe_infinity_075", before, after,
                                                   current_deployment=True)
        self.assertFalse(result["steady_state_direct_io_claimed"])
        self.assertEqual(result["read_bytes"], 0)
        after["moe"]["revision"]["engine_generated_tokens"] = 511
        with self.assertRaises(base.GateError):
            base.validate_measured_engagement("moe_infinity_075", before, after,
                                               current_deployment=True)

    def test_moe_engagement_accepts_explicit_short_cell_token_count(self):
        revision = dict(engine_generated_tokens=0, engine_steps=0, expert_cache_accesses=0,
                        expert_cache_hits=0, expert_cache_misses=0, kv_cache_num_blocks=128)
        before = {"process_io": {"read_bytes": 0, "cpu_time_s": 0},
                  "moe": {"revision": revision,
                          "metrics": {"moe_tokens_generated_total": 0,
                                      "moe_engine_steps_total": 0}}}
        after = copy.deepcopy(before)
        after["moe"]["revision"].update(
            engine_generated_tokens=64, engine_steps=64,
            expert_cache_accesses=12, expert_cache_hits=10, expert_cache_misses=2)
        after["moe"]["metrics"].update(
            moe_tokens_generated_total=64, moe_engine_steps_total=64)
        result = base.validate_measured_engagement(
            "moe_infinity_075", before, after, current_deployment=True,
            expected_generated_tokens=64)
        self.assertEqual(result["metrics_delta"], {"tokens": 64, "steps": 64})
        with self.assertRaises(base.GateError):
            base.validate_measured_engagement(
                "moe_infinity_075", before, after, current_deployment=True,
                expected_generated_tokens=384)

    def test_hook_engagement_is_not_completed_eviction_evidence(self):
        values = dict(page_fault_calls=100, stride_detections=10, prefetches_issued=10,
                      lfu_activations=10, lfu_accesses=25600, lfu_sampled_updates=100,
                      lfu_reorder_requests=100, eviction_prepares=10)
        before = {"process_io": {"read_bytes": 0, "cpu_time_s": 0},
                  "policy": dict.fromkeys(values, 0), "evictions": None}
        after = copy.deepcopy(before)
        after["policy"] = values
        result = base.validate_measured_engagement("gpubpf_host_stride_lfu", before, after,
                                                   current_deployment=True)
        self.assertFalse(result["completed_evictions_claimed"])
        self.assertIsNone(result["eviction_delta"])
        after["policy"]["lfu_reorder_requests"] = 0
        with self.assertRaises(base.GateError):
            base.validate_measured_engagement("gpubpf_host_stride_lfu", before, after,
                                               current_deployment=True)

    def test_fixed_400w_cap_is_recorded_but_thermal_slowdown_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.csv"
            header = ("timestamp, memory, temp, power, sm, mem, "
                      "clocks_event_reasons.sw_power_cap, clocks_event_reasons.hw_thermal_slowdown\n")
            path.write_text(header + "t, 12, 50, 400, 2100, 1000, Active, Not Active\n")
            result = base.validate_gpu_telemetry(path, allow_fixed_power_cap=True)
            self.assertEqual(result["fixed_power_cap_samples"], 1)
            with self.assertRaises(base.GateError):
                base.validate_gpu_telemetry(path)
            path.write_text(header + "t, 12, 80, 400, 2100, 1000, Active, Active\n")
            with self.assertRaises(base.GateError):
                base.validate_gpu_telemetry(path, allow_fixed_power_cap=True)

    def test_shared_store_does_not_change_llama_command(self):
        for config in base.CONFIGS:
            command, _ = base.server_command(config, 18080, Path("/tmp/attempt"),
                                              Path("/tmp/current-expert-store"))
            if config == "moe_infinity_075":
                self.assertEqual(command[command.index("--offload-dir") + 1],
                                 "/tmp/current-expert-store")
            else:
                self.assertEqual(command, base.server_command(config, 18080, Path("/tmp/attempt"))[0])

    def test_historical_and_partial_preflight_are_not_current_correctness(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            for protocol in (base.REPAIR_PROTOCOL_ID, current.PROTOCOL):
                (path / "preflight-result.json").write_text(json.dumps({
                    "protocol": protocol, "status": "passed", "results": {},
                    "driver": current.DRIVER, "kernel": current.KERNEL,
                }))
                with self.assertRaises(base.GateError):
                    current.load_preflight(path)

    def test_small_block_count_remains_inconclusive(self):
        self.assertEqual(base.analyze_valid_blocks([{}, {}]),
                         {"outcome": "inconclusive", "valid_blocks": 2})

    def test_descriptive_checkpoint_uses_only_complete_paired_blocks(self):
        blocks = [{"results": {c: {"output_throughput_tokens_per_s": float(i + 1)}
                                for c in base.CONFIGS}} for i in range(2)]
        values = current.descriptive_summary(blocks)
        for config in base.CONFIGS:
            self.assertTrue(values[config]["preliminary"])
            self.assertEqual(values[config]["block_output_throughput_tokens_per_s"], [1.0, 2.0])
            self.assertEqual(values[config]["paired_geometric_mean_ratio_vs_moe"], 1.0)


if __name__ == "__main__":
    unittest.main()
