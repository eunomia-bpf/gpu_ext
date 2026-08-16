import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


audit = load("stage4_audit", ROOT / "analysis" / "audit_eviction_policies.py")
summary = load("stage4_summary", ROOT / "analysis" / "summarize_stage4.py")


class Stage4AuditTests(unittest.TestCase):
    def test_requested_candidates_are_conservative(self):
        data = audit.audit(ROOT.parents[1] / "extension")
        items = {item["policy"]: item for item in data["policies"]}
        self.assertFalse(items["eviction_fifo"]["suitable_for_initial_pressure_test"])
        self.assertTrue(items["prefetch_always_max_cycle_moe"]["suitable_for_initial_pressure_test"])
        self.assertFalse(items["prefetch_cooperative"]["suitable_for_initial_pressure_test"])
        self.assertTrue(items["prefetch_cooperative"]["uses_bpf_wq"])
        self.assertTrue(items["prefetch_cooperative"]["calls_bpf_gpu_migrate_range"])

    def test_move_head_activate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.bpf.c"
            path.write_text(
                'SEC("struct_ops/gpu_block_activate")\n'
                'int BPF_PROG(gpu_block_activate, void *p, void *c, void *l) {'
                ' bpf_gpu_block_move_head(c, l); return 1; }\n'
                'SEC(".struct_ops") struct gpu_mem_ops x = {};\n'
            )
            item = audit.audit_file(path)
            self.assertIsNotNone(item)
            self.assertFalse(item["suitable_for_initial_pressure_test"])
            self.assertTrue(any("move_head" in reason for reason in item["rejection_reasons"]))


class Stage4SummaryTests(unittest.TestCase):
    def test_reduced_capacity_grouping_and_trace_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "reduced_capacity" / "prefetch_always_max" / "1.10" / "run"
            root.mkdir(parents=True)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "evidence_class": "GPU_EXT_STAGE4_RUN",
                        "experiment": "reduced_capacity",
                        "policy": "prefetch_always_max",
                        "ratio": "1.10",
                        "run_kind": "trace",
                        "correct": True,
                        "struct_ops_detached": True,
                        "xid_delta": 0,
                    }
                )
            )
            rows = [
                {
                    "phase": "capacity_manifest",
                    "evidence_class": "PHYSICALLY_RESERVED_GUARD_MODEL",
                    "effective_gpu_capacity_bytes": 8 << 30,
                    "managed_working_set_bytes": int(8 * 1.1 * (1 << 30)),
                    "actual_working_set_ratio": 1.1,
                    "main_reserve_allocated_bytes": 15 << 30,
                    "guard_allocated_bytes": 1 << 30,
                    "gpu_free_initial": 24 << 30,
                    "gpu_free_after_main_reserve": 9 << 30,
                    "gpu_free_after_guard": 8 << 30,
                    "capacity_target_relative_error": 0.0,
                    "working_set_ratio_error": 0.0,
                    "region_a_bytes": int(4.4 * (1 << 30)),
                    "region_b_bytes": int(4.4 * (1 << 30)),
                },
                *({"phase": phase, "elapsed_ms": index + 1, "correct": True}
                  for index, phase in enumerate(summary.PHASES)),
            ]
            (root / "program.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with (root / "prefetch_decision_trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["action_name", "final_pages"])
                writer.writeheader()
                writer.writerow({"action_name": "BYPASS", "final_pages": "512"})
            (root / "chunk_trace.csv").write_text(
                "hook_type,timestamp_ns,va_start\nEVICTION_SELECTED,2,0x1000\n"
            )
            result = summary.summarize(summary.collect(Path(tmp)))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["capacity_model"], "PHYSICALLY_RESERVED_GUARD_MODEL")
            self.assertEqual(result[0]["guard_allocated_bytes_mean"], float(1 << 30))
            self.assertEqual(result[0]["actual_working_set_ratio_mean"], 1.1)
            self.assertEqual(result[0]["action_bypass_count"], 1)
            self.assertEqual(result[0]["final_pages_mean"], 512.0)
            self.assertEqual(result[0]["selected_eviction_count"], 1)

    def test_legacy_mathematical_headroom_is_relabelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "reduced_capacity" / "custom_no_policy" / "1.10" / "run"
            root.mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({
                "evidence_class": "GPU_EXT_STAGE4_RUN",
                "experiment": "reduced_capacity",
                "policy": "custom_no_policy",
                "ratio": "1.10",
                "run_kind": "timing",
                "correct": True,
                "struct_ops_detached": True,
                "xid_delta": 0,
            }))
            (root / "program.jsonl").write_text(json.dumps({
                "phase": "capacity_manifest",
                "evidence_class": "REDUCED_EFFECTIVE_GPU_CAPACITY",
                "effective_gpu_capacity_bytes": 8 << 30,
                "managed_working_set_bytes": int(8.8 * (1 << 30)),
            }) + "\n")
            result = summary.summarize(summary.collect(Path(tmp)))
            self.assertEqual(result[0]["capacity_model"],
                             "LEGACY_MATHEMATICAL_HEADROOM_MODEL")

    def test_trace_overhead_uses_vector_kernel_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind, values in (("timing", (100.0, 102.0)), ("trace", (103.0, 105.0))):
                for index, elapsed in enumerate(values):
                    root = Path(tmp) / "trace_overhead" / kind / str(index)
                    root.mkdir(parents=True)
                    (root / "manifest.json").write_text(json.dumps({
                        "evidence_class": "GPU_EXT_STAGE4_TRACE_OVERHEAD",
                        "experiment": "trace_overhead",
                        "policy": "custom_no_policy",
                        "ratio": "256M",
                        "run_kind": kind,
                        "correct": True,
                        "struct_ops_detached": True,
                        "xid_delta": 0,
                    }))
                    (root / "program.jsonl").write_text(json.dumps({
                        "phase": "kernel_1_demand", "elapsed_ms": elapsed,
                    }) + "\n")
            rows = summary.summarize_trace_overhead(summary.collect(Path(tmp)))
            by_kind = {row["run_kind"]: row for row in rows}
            self.assertEqual(by_kind["timing"]["kernel_1_demand_count"], 2)
            self.assertEqual(by_kind["timing"]["kernel_1_demand_mean"], 101.0)
            self.assertAlmostEqual(
                by_kind["trace"]["trace_attached_kernel_1_overhead_percent"],
                (104.0 / 101.0 - 1.0) * 100.0,
            )


if __name__ == "__main__":
    unittest.main()
