#!/usr/bin/env python3
"""CPU-only tests for the performance-only runner (no GPU, no server launch)."""

import contextlib
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "run_perf_only.py"
SPEC = importlib.util.spec_from_file_location("run_perf_only", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
ops = runner.ops

SCHEDULE = json.loads((HERE / "schedule.json").read_text(encoding="utf-8"))
PROMPTS = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
PREFIXES = PROMPTS["prefixes"]
EXPECTED_DRIVER = "610.43.02"


class FakeProc:
    def __init__(self):
        self.pid = 0
        self.returncode = None

    def poll(self):
        return None


class FakeLogFile:
    def close(self):
        pass


def fake_start_server(config, model_path, cache_dir, port, log_path,
                      trace_dir=None, expected_driver=ops.EXPECTED_DRIVER):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in PREFIXES:
        rid = f"cmpl-lmc-p{item['index']}-cold-0-r{item['index']}"
        lines.append(f"Reqid: {rid}, Total tokens {len(item['cold_token_ids'])}")
        if config != "recompute":
            stored = int(item["expected_store_tokens"])
            lines.append(f"[req_id={rid}] Stored {stored} out of total {stored} tokens")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (FakeProc(), FakeLogFile(),
            ["fake-vllm", "serve", str(model_path)],
            ["/usr/bin/taskset", "-c", "8-15", "fake-vllm", "serve", str(model_path)])


def make_fake_completion():
    state = {"calls": 0}

    def fake_completion(port, token_ids, request_id):
        state["calls"] += 1
        phase = "cold" if request_id.endswith("-cold") else "warm"
        index = int(request_id.split("-p")[1].split("-")[0])
        ttft_ms = (10.0 + index) if phase == "cold" else (5.0 + 0.5 * index)
        return {"request_header": request_id, "engine_request_id": f"cmpl-{request_id}",
                "input_tokens": len(token_ids), "status": 200,
                "ttft_ms": ttft_ms, "e2e_ms": ttft_ms + 50.0,
                "usage": {"completion_tokens": ops.OUTPUT_TOKENS,
                          "prompt_tokens": len(token_ids)},
                "text": "generated text",
                "generated_token_ids": [7] * ops.OUTPUT_TOKENS}

    return fake_completion, state


def fake_stop(proc, log_file):
    proc.returncode = 0
    log_file.close()


def fake_stop_137(proc, log_file):
    proc.returncode = 137
    log_file.close()


def bad_wait_ready(proc, port, log_path):
    raise ops.GateError("server readiness timeout")


def exploding_warm_completion(port, token_ids, request_id):
    if request_id.endswith("-warm"):
        raise RuntimeError("warm request exploded")
    index = int(request_id.split("-p")[1].split("-")[0])
    return {"request_header": request_id, "engine_request_id": f"cmpl-{request_id}",
            "input_tokens": len(token_ids), "status": 200,
            "ttft_ms": 10.0 + index, "e2e_ms": 60.0 + index,
            "usage": {"completion_tokens": ops.OUTPUT_TOKENS,
                      "prompt_tokens": len(token_ids)},
            "text": "generated text",
            "generated_token_ids": [7] * ops.OUTPUT_TOKENS}


def refusing_start_server(config, model_path, cache_dir, port, log_path,
                          trace_dir=None, expected_driver=ops.EXPECTED_DRIVER):
    raise FileExistsError(f"{log_path} exists")


@contextlib.contextmanager
def faked_ops(**overrides):
    completion, state = make_fake_completion()
    defaults = {
        "start_server": fake_start_server,
        "wait_ready": lambda proc, port, log_path: None,
        "streamed_completion": completion,
        "stop_owned_server": fake_stop,
        "resolve_model": lambda local_only=True: Path("/frozen-model"),
    }
    defaults.update(overrides)
    with contextlib.ExitStack() as stack:
        for name, fn in defaults.items():
            stack.enter_context(mock.patch.object(runner.ops, name, fn))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield state


def store_log(directory: Path, rid_prefix: str = "cmpl-lmc-p0-cold",
              total: int = 1549, stored: int = 1536, stored_total: int = 1536) -> Path:
    rid = f"{rid_prefix}-0-r0"
    log = directory / "server.log"
    log.write_text(f"Reqid: {rid}, Total tokens {total}\n"
                   f"[req_id={rid}] Stored {stored} out of total {stored_total} tokens\n",
                   encoding="utf-8")
    return log


class RunnerShapeTests(unittest.TestCase):
    def test_module_constants(self):
        self.assertEqual(runner.KIND, "lmcache_perf_only")
        self.assertEqual(runner.CONFIGS, ("recompute", "lmcache_cpu", "lmcache_disk"))
        self.assertEqual(runner.DEFAULT_BLOCKS, 5)
        self.assertEqual(runner.DEFAULT_PORT, 18080)
        self.assertEqual(runner.DEFAULT_STORE_BARRIER_TIMEOUT_S, 120.0)
        self.assertEqual(runner.REQUEST_LABELS, ("cold", "warm"))

    def test_parse_args_defaults(self):
        args = runner.parse_args([])
        self.assertEqual(args.blocks, 5)
        self.assertEqual(args.port, 18080)
        self.assertEqual(args.store_barrier_timeout_s, 120.0)
        self.assertEqual(args.expected_driver, EXPECTED_DRIVER)
        self.assertIs(args.dry_run, False)
        self.assertIsNone(args.output)

    def test_removed_mechanisms_absent_from_source(self):
        src = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("wait_gpu_idle", src)
        self.assertNotIn("file_identity", src)
        self.assertNotIn("Leases", src)
        self.assertNotIn("shared.", src)

    def test_prompt_artifact_loads_pinned_content(self):
        prompts = runner.load_fixed_prompts()
        self.assertEqual(prompts, PROMPTS)
        self.assertEqual(len(prompts["prefixes"]), 8)
        for item in prompts["prefixes"]:
            self.assertEqual(item["expected_store_tokens"], ops.PREFIX_TOKENS)
            self.assertEqual(len(item["cold_token_ids"]), item["cold_tokens"])
            self.assertEqual(len(item["warm_token_ids"]), item["warm_tokens"])
            self.assertTrue(all(isinstance(t, int) for t in item["cold_token_ids"]))


class RotationTests(unittest.TestCase):
    def test_one_block_matches_schedule_attempt_zero(self):
        self.assertEqual(runner.rotation_orders(1),
                         [SCHEDULE["attempts"][0]["order"]])

    def test_five_blocks_match_schedule_attempts_zero_to_four(self):
        expected = [row["order"] for row in SCHEDULE["attempts"][:5]]
        self.assertEqual(runner.rotation_orders(5), expected)

    def test_five_block_orders_are_exact(self):
        self.assertEqual(runner.rotation_orders(5), [
            ["lmcache_cpu", "recompute", "lmcache_disk"],
            ["recompute", "lmcache_disk", "lmcache_cpu"],
            ["lmcache_disk", "lmcache_cpu", "recompute"],
            ["recompute", "lmcache_disk", "lmcache_cpu"],
            ["lmcache_cpu", "recompute", "lmcache_disk"],
        ])

    def test_every_cell_once_and_arms_balanced(self):
        for blocks in (1, 5):
            orders = runner.rotation_orders(blocks)
            cells = [(block, config) for block, order in enumerate(orders)
                     for config in order]
            self.assertEqual(len(cells), 3 * blocks)
            self.assertEqual(len(set(cells)), len(cells), "a cell ran more than once")
            for config in runner.CONFIGS:
                self.assertEqual(sum(1 for _, c in cells if c == config), blocks)

    def test_rotation_is_deterministic(self):
        self.assertEqual(runner.rotation_orders(5), runner.rotation_orders(5))

    def test_blocks_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            runner.rotation_orders(0)
        with self.assertRaises(ValueError):
            runner.rotation_orders(16)


class BarrierTests(unittest.TestCase):
    def test_recompute_barrier_not_applicable(self):
        result = runner.store_barrier("recompute", Path("/nonexistent/log"),
                                      "cmpl-x", 1536, 1549, 120.0, 1.0)
        self.assertEqual(result, {"applicable": False, "satisfied": None, "waited_s": 0.0})

    def test_disabled_when_timeout_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = store_log(Path(tmp))
            result = runner.store_barrier("lmcache_cpu", log, "cmpl-lmc-p0-cold",
                                          1536, 1549, 0.0, 1.0)
            self.assertEqual(result, {"applicable": True, "satisfied": False,
                                      "waited_s": 0.0, "disabled": True})

    def test_satisfied_barrier_keeps_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = store_log(Path(tmp))
            result = runner.store_barrier("lmcache_disk", log, "cmpl-lmc-p0-cold",
                                          1536, 1549, 10.0, 0.01)
            self.assertIs(result["applicable"], True)
            self.assertIs(result["satisfied"], True)
            self.assertGreaterEqual(result["waited_s"], 0.0)
            self.assertEqual(result["log_values"]["runtime_ids"],
                             ["cmpl-lmc-p0-cold-0-r0"])
            self.assertEqual(result["log_values"]["request_totals"], [1549])
            self.assertEqual(result["log_values"]["stores"], [[1536, 1536]])

    def test_timeout_is_recorded_not_fatal_and_observations_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = store_log(Path(tmp), stored=1024)
            result = runner.store_barrier("lmcache_cpu", log, "cmpl-lmc-p0-cold",
                                          1536, 1549, 0.2, 0.05)
            self.assertIs(result["applicable"], True)
            self.assertIs(result["satisfied"], False)
            self.assertGreater(result["waited_s"], 0.05)
            self.assertEqual(result["log_values"]["stores"], [[1024, 1536]])
            self.assertEqual(result["log_values"]["request_totals"], [1549])

    def test_missing_log_read_error_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.store_barrier("lmcache_disk", Path(tmp) / "absent.log",
                                          "cmpl-lmc-p0-cold", 1536, 1549, 0.2, 0.05)
            self.assertIs(result["satisfied"], False)
            self.assertIn("read_error", result["log_values"])
            self.assertIn("FileNotFoundError", result["log_values"]["read_error"])

    def test_log_parse_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = store_log(Path(tmp)).read_text(encoding="utf-8")
            values = ops.request_log_values(log, "cmpl-lmc-p0-cold")
            self.assertEqual(values["runtime_ids"], ["cmpl-lmc-p0-cold-0-r0"])
            self.assertEqual(values["request_totals"], [1549])
            self.assertEqual(values["stores"], [[1536, 1536]])
            self.assertEqual(values["hits"], [])
            self.assertEqual(values["retrieved"], [])


class RecordAggregationTests(unittest.TestCase):
    def test_request_entry_minimal(self):
        entry = runner.request_entry("cold", 3)
        self.assertEqual(entry, {"phase": "cold", "prefix_index": 3,
                                 "request_id": "lmc-p3-cold", "attempted": True})

    def test_request_entry_error_and_reason(self):
        entry = runner.request_entry("warm", 1, error="TimeoutError: boom",
                                     attempted=False, reason="server never became ready")
        self.assertEqual(entry["error"], "TimeoutError: boom")
        self.assertIs(entry["attempted"], False)
        self.assertEqual(entry["reason"], "server never became ready")
        self.assertNotIn("ttft_ms", entry)

    def test_request_entry_merges_response_numbers(self):
        response = {"engine_request_id": "cmpl-x", "ttft_ms": 1.5, "e2e_ms": 2.0,
                    "status": 200, "usage": {"completion_tokens": 16}}
        entry = runner.request_entry("warm", 2, response=response)
        self.assertEqual(entry["engine_request_id"], "cmpl-x")
        self.assertEqual(entry["ttft_ms"], 1.5)
        self.assertEqual(entry["usage"], {"completion_tokens": 16})

    def test_warm_aggregates_none_without_successes(self):
        record = {"requests": [{"phase": "warm", "prefix_index": 0,
                                "attempted": True, "error": "RuntimeError: x"}]}
        self.assertIsNone(runner.warm_aggregates(record, 0, 1_000_000_000))

    def test_warm_aggregates_values(self):
        record = {"requests": [
            {"phase": "cold", "prefix_index": 0, "attempted": True, "ttft_ms": 999.0},
            {"phase": "warm", "prefix_index": 0, "attempted": True, "ttft_ms": 100.0,
             "usage": {"completion_tokens": 16}},
            {"phase": "warm", "prefix_index": 1, "attempted": True, "ttft_ms": 200.0,
             "usage": {"completion_tokens": 16}},
            {"phase": "warm", "prefix_index": 2, "attempted": True, "ttft_ms": 300.0,
             "usage": {"completion_tokens": 16}},
            {"phase": "warm", "prefix_index": 3, "attempted": True,
             "error": "RuntimeError: x"},
        ]}
        out = runner.warm_aggregates(record, 1_000_000_000, 11_000_000_000)
        self.assertEqual(out["requests"], 3)
        self.assertEqual(out["attempts"], 4)
        self.assertEqual(out["failures"], 1)
        self.assertEqual(out["output_tokens"], 48)
        self.assertEqual(out["elapsed_s"], 10.0)
        self.assertEqual(out["requests_per_s"], 0.3)
        self.assertEqual(out["output_tokens_per_s"], 4.8)
        self.assertEqual(out["warm_ttft_values_ms"], [100.0, 200.0, 300.0])
        self.assertEqual(out["warm_ttft_median_ms"], 200.0)
        self.assertEqual(out["warm_ttft_max_ms"], 300.0)
        self.assertIsInstance(out["warm_ttft_p95_ms"], float)
        self.assertGreaterEqual(out["warm_ttft_p95_ms"], 200.0)
        self.assertIn("cold population", out["excludes"])

    def test_cell_metrics_extracts_warm_phase(self):
        record = {"ready": True, "server_returncode": 0, "warm_phase": {
            "warm_ttft_median_ms": 7.5, "requests_per_s": 1.25,
            "output_tokens_per_s": 20.0, "requests": 8, "failures": 0}}
        metrics = runner.cell_metrics(record)
        self.assertEqual(metrics["warm_ttft_median_ms"], 7.5)
        self.assertEqual(metrics["warm_requests_per_s"], 1.25)
        self.assertEqual(metrics["warm_requests_ok"], 8)
        self.assertIs(metrics["ready"], True)
        self.assertEqual(metrics["server_returncode"], 0)

    def test_cell_metrics_without_warm_phase(self):
        metrics = runner.cell_metrics({"ready": False, "server_returncode": 137})
        self.assertIsNone(metrics["warm_ttft_median_ms"])
        self.assertEqual(metrics["warm_requests_ok"], 0)
        self.assertEqual(metrics["server_returncode"], 137)

    def test_arm_summary_groups_every_config(self):
        cells = [{"config": "recompute", "warm": {"warm_ttft_median_ms": 1.0,
                                                  "warm_requests_per_s": 2.0,
                                                  "warm_output_tokens_per_s": 3.0}},
                 {"config": "lmcache_cpu", "warm": {"warm_ttft_median_ms": 2.0,
                                                    "warm_requests_per_s": 4.0,
                                                    "warm_output_tokens_per_s": 5.0}}]
        summary = runner.arm_summary(cells)
        self.assertEqual(set(summary), set(runner.CONFIGS))
        self.assertEqual(summary["recompute"]["cell_count"], 1)
        self.assertEqual(summary["lmcache_cpu"]["cell_count"], 1)
        self.assertEqual(summary["lmcache_disk"]["cell_count"], 0)
        self.assertEqual(summary["recompute"]["warm_ttft_median_ms"], [1.0])

    def test_output_root_default_under_raw_and_explicit_path(self):
        default = runner.output_root(None)
        self.assertTrue(str(default).startswith(str(HERE / "raw") + "/lmcache_perf_only-"))
        self.assertEqual(runner.output_root(Path("/tmp/explicit-root")),
                         Path("/tmp/explicit-root").resolve())


class CellTests(unittest.TestCase):
    def run_fake_cell(self, config, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "block-00" / f"position-0-{config}"
            with faked_ops(**overrides):
                record = runner.run_cell(config, 0, 0, run_dir, 18080,
                                         Path("/frozen-model"), PREFIXES,
                                         EXPECTED_DRIVER, 120.0)
            result_path = run_dir / "result.json"
            self.assertTrue(result_path.is_file())
            return record, json.loads(result_path.read_text(encoding="utf-8"))

    def test_successful_cell_records_every_number_and_code(self):
        record, written = self.run_fake_cell("lmcache_cpu")
        self.assertIs(record["ready"], True)
        self.assertIsNone(record["ready_error"])
        self.assertIsNone(record["error"])
        cold = [r for r in record["requests"] if r["phase"] == "cold"]
        warm = [r for r in record["requests"] if r["phase"] == "warm"]
        self.assertEqual(len(cold), 8)
        self.assertEqual(len(warm), 8)
        for item in PREFIXES:
            i = item["index"]
            c = next(r for r in cold if r["prefix_index"] == i)
            w = next(r for r in warm if r["prefix_index"] == i)
            self.assertEqual(c["engine_request_id"], f"cmpl-lmc-p{i}-cold")
            self.assertIn("ttft_ms", c)
            self.assertIn("e2e_ms", c)
            self.assertIn("generated_token_ids", c)
            self.assertEqual(w["engine_request_id"], f"cmpl-lmc-p{i}-warm")
            self.assertIn("ttft_ms", w)
        self.assertEqual(len(record["barriers"]), 8)
        for barrier in record["barriers"]:
            self.assertIs(barrier["applicable"], True)
            self.assertIs(barrier["satisfied"], True)
        phase = record["warm_phase"]
        self.assertEqual(phase["requests"], 8)
        self.assertEqual(phase["failures"], 0)
        self.assertEqual(phase["warm_ttft_values_ms"], [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
        self.assertEqual(phase["warm_ttft_median_ms"], 6.75)
        self.assertEqual(phase["warm_ttft_max_ms"], 8.5)
        self.assertGreater(phase["requests_per_s"], 0.0)
        self.assertEqual(record["server_returncode"], 0)
        self.assertEqual(record["cleanup_errors"], [])
        self.assertEqual(record["command"], ["fake-vllm", "serve", "/frozen-model"])
        self.assertIn("/usr/bin/taskset", record["launch_command"][0])
        self.assertEqual(record["environment"]["LMCACHE_LOCAL_CPU"], "True")
        self.assertNotIn("server_log_identity", record)
        self.assertEqual(written, record)

    def test_recompute_cell_barriers_not_applicable(self):
        record, _ = self.run_fake_cell("recompute")
        self.assertIs(record["ready"], True)
        self.assertEqual(len(record["barriers"]), 8)
        for barrier in record["barriers"]:
            self.assertEqual(barrier, {"applicable": False, "satisfied": None,
                                       "waited_s": 0.0})
        self.assertIsNotNone(record["warm_phase"])

    def test_readiness_failure_preserved_without_requests(self):
        record, _ = self.run_fake_cell("lmcache_disk", wait_ready=bad_wait_ready)
        self.assertIs(record["ready"], False)
        self.assertIn("readiness", record["ready_error"])
        self.assertEqual(len(record["requests"]), 16)
        for entry in record["requests"]:
            self.assertIs(entry["attempted"], False)
            self.assertEqual(entry["reason"], "server never became ready")
        self.assertIsNone(record.get("warm_phase"))
        self.assertEqual(record["barriers"], [])
        self.assertEqual(record["server_returncode"], 0)

    def test_warm_failure_preserved_without_retry_or_filter(self):
        record, _ = self.run_fake_cell("lmcache_cpu",
                                       streamed_completion=exploding_warm_completion)
        warm = [r for r in record["requests"] if r["phase"] == "warm"]
        self.assertEqual(len(warm), 8)
        for entry in warm:
            self.assertEqual(entry["error"], "RuntimeError: warm request exploded")
            self.assertNotIn("ttft_ms", entry)
        self.assertEqual(len([r for r in record["requests"] if r["phase"] == "cold"]), 8)
        self.assertIsNone(record["warm_phase"])
        self.assertIs(record["ready"], True)
        self.assertEqual(record["server_returncode"], 0)

    def test_launch_refusal_recorded_and_cell_continues(self):
        record, written = self.run_fake_cell("lmcache_disk",
                                             start_server=refusing_start_server)
        self.assertTrue(record["error"].startswith("server log already exists"))
        self.assertIs(record["ready"], False)
        self.assertEqual(record["requests"], [])
        self.assertEqual(record["barriers"], [])
        self.assertIsNone(record["server_returncode"])
        self.assertNotIn("command", record)
        self.assertEqual(written, record)


class CampaignTests(unittest.TestCase):
    def test_one_block_campaign_runs_every_cell_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            args = runner.parse_args(["--blocks", "1", "--output", str(out)])
            with faked_ops():
                rc = runner.run_campaign(args)
            self.assertEqual(rc, 0)
            campaign = json.loads((out / "campaign.json").read_text(encoding="utf-8"))
            self.assertEqual(campaign["kind"], "lmcache_perf_only")
            self.assertEqual(campaign["params"]["attempts_per_cell"], 1)
            self.assertIs(campaign["params"]["retry"], False)
            self.assertIs(campaign["params"]["result_filtering"], False)
            self.assertIs(campaign["params"]["gpu_idle_wait"], False)
            self.assertNotIn("inode", campaign["params"]["prompts"])
            self.assertNotIn("mtime_ns", campaign["params"]["prompts"])
            self.assertEqual(campaign["block_orders"], [SCHEDULE["attempts"][0]["order"]])
            self.assertEqual(len(campaign["cells"]), 3)
            self.assertEqual(campaign["complete_cells"], 3)
            for config in runner.CONFIGS:
                self.assertEqual(campaign["arm_summary"][config]["cell_count"], 1)
            for cell in campaign["cells"]:
                run_dir = Path(cell["run_dir"])
                record = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(len(record["requests"]), 16)
                self.assertEqual(record["server_returncode"], 0)
                self.assertTrue((run_dir / "server.log").is_file())
            summary = (out / "summary.md").read_text(encoding="utf-8")
            for config in runner.CONFIGS:
                self.assertIn(config, summary)

    def test_campaign_keeps_nonzero_returncode_and_retries_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            args = runner.parse_args(["--blocks", "1", "--output", str(out)])
            with faked_ops(stop_owned_server=fake_stop_137,
                            streamed_completion=exploding_warm_completion):
                rc = runner.run_campaign(args)
            self.assertEqual(rc, 2)
            campaign = json.loads((out / "campaign.json").read_text(encoding="utf-8"))
            self.assertEqual(campaign["complete_cells"], 0)
            for cell in campaign["cells"]:
                record = json.loads((Path(cell["run_dir"]) / "result.json")
                                    .read_text(encoding="utf-8"))
                self.assertEqual(record["server_returncode"], 137)
                warm = [r for r in record["requests"] if r["phase"] == "warm"]
                self.assertEqual(len(warm), 8, "warm requests were retried or dropped")
                for entry in warm:
                    self.assertEqual(entry["error"], "RuntimeError: warm request exploded")
            self.assertTrue((out / "summary.md").is_file())


class DryRunTests(unittest.TestCase):
    def capture_main(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = runner.main(argv)
        return rc, json.loads(buf.getvalue())

    def test_main_dry_run_one_block(self):
        rc, plan = self.capture_main(["--dry-run", "--blocks", "1"])
        self.assertEqual(rc, 0)
        self.assertIs(plan["dry_run"], True)
        self.assertEqual(plan["blocks"], 1)
        self.assertEqual(plan["block_orders"], [SCHEDULE["attempts"][0]["order"]])
        self.assertEqual(plan["configs"], list(runner.CONFIGS))
        self.assertEqual(plan["prompts"]["prefix_count"], 8)
        self.assertEqual(plan["prompts"]["prefix_tokens"], ops.PREFIX_TOKENS)
        self.assertEqual(plan["store_barrier"]["timeout_s"], 120.0)
        self.assertNotIn("wait_gpu_idle", " ".join(plan["reuse"]))
        self.assertEqual(len(plan["removed_mechanisms"]), 3)

    def test_main_dry_run_five_blocks_default_driver(self):
        rc, plan = self.capture_main(["--dry-run", "--blocks", "5"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["block_orders"],
                         [row["order"] for row in SCHEDULE["attempts"][:5]])
        self.assertEqual(plan["expected_driver_parameter"], EXPECTED_DRIVER)
        self.assertEqual(plan["port"], 18080)

    def test_main_dry_run_defaults_to_five_blocks(self):
        rc, plan = self.capture_main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["blocks"], 5)


if __name__ == "__main__":
    unittest.main()
