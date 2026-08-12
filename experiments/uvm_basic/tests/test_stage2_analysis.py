import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "analysis" / "summarize_stage2.py"
SPEC = importlib.util.spec_from_file_location("summarize_stage2", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class Stage2AnalysisTests(unittest.TestCase):
    def test_stats(self):
        result = MODULE.stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["p95"], 4.0)

    def test_trace_pid_attribution_and_unavailable_policy_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "prefetch_trace.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, ["time_ms", "fault_pid", "owner_tgid", "page_index", "max_first", "max_outer"])
                writer.writeheader()
                writer.writerow({"time_ms": 1, "fault_pid": 10, "owner_tgid": 42, "page_index": 5, "max_first": 0, "max_outer": 16})
                writer.writerow({"time_ms": 2, "fault_pid": 11, "owner_tgid": 99, "page_index": 7, "max_first": 0, "max_outer": 32})
            with (root / "chunk_trace.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, ["time_ms", "hook_type", "pid", "owner_pid"])
                writer.writeheader()
                writer.writerow({"time_ms": 1, "hook_type": "ACTIVATE", "pid": 10, "owner_pid": 42})
                writer.writerow({"time_ms": 2, "hook_type": "EVICTION_PREPARE", "pid": 11, "owner_pid": 99})
            result = MODULE.trace_summary(root, {"policy": "prefetch_none", "size": "256M", "run_kind": "trace", "workload_pid": 42})
            self.assertEqual(result["prefetch_callback_count"], 1)
            self.assertEqual(result["max_region_pages_mean"], 16)
            self.assertEqual(result["chunk_activate_count"], 1)
            self.assertEqual(result["eviction_prepare_count"], 0)
            self.assertEqual(result["selected_prefetch_bytes"], "UNAVAILABLE")

    def test_failed_trace_attach_is_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prefetch_trace.stderr").write_text("Failed to attach BPF skeleton: -2\n")
            (root / "chunk_trace.stderr").write_text("Failed to attach BPF skeleton: -2\n")
            result = MODULE.trace_summary(
                root,
                {"policy": "custom_no_policy", "size": "256M", "run_kind": "trace", "workload_pid": 42},
            )
            self.assertEqual(result["prefetch_callback_count"], "UNAVAILABLE")
            self.assertEqual(result["chunk_activate_count"], "UNAVAILABLE")
            self.assertEqual(result["eviction_prepare_count"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
