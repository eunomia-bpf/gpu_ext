"""CPU-only coverage for the current 575 continuation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import run_575_head_to_head as current
import run_moe_head_to_head as base


class CurrentStackTests(unittest.TestCase):
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
