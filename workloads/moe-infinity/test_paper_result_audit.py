"""CPU-only full raw-fixture audit and corruption/replay regression tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import paper_result_audit as audit
import run_paper_comparison as runner


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def safety():
    return {"power_limit_service": "active", "power_limit_w": 400,
            "dmesg_abnormal": [], "journal_abnormal": [], "xids": [],
            "uvm_refcount": 0, "struct_ops": {"maps": [], "links": []},
            "gpu": {"driver": "575.57.08", "compute_apps": [],
                    "memory_used_mib": 2, "utilization_gpu_percent": 0}}


def activation(mode, phase):
    completed = {"cold": 0, "before": 1, "after": 9}[phase]
    measured = phase == "after" and mode != "native-off"
    bpf = mode == "paper-bpf"
    controller = {} if mode == "native-off" else {
        "completed_requests": completed, "eamc_entries": completed * 2,
        "aborted_requests": 0, "active_request_traces": 0,
        "matched_predictions": 16 if measured else 0,
        "prefetch_candidates_selected": 100 if measured else 0,
        "rank_calls": 20 if bpf and measured else 0,
        "bpf_match_calls": 16 if bpf and measured else 0,
        "rank_mismatches": 0, "match_mismatches": 0}
    dispatcher = {"mode": audit.paper.MODES.index(mode),
        "prefetch_submitted": 100 if measured else 0,
        "prefetch_completed": 12 if measured else 0,
        "prefetch_bytes": 12000 if measured else 0,
        "eviction_selections": 20 if measured else 0,
        "bpf_eviction_calls": 20 if measured and bpf else 0,
        "prefetch_hits": 9 if measured else 0,
        "prefetch_wasted": 2 if measured else 0,
        "prefetch_wasted_bytes": 2000 if measured else 0,
        "prefetch_unused_resident": 1 if measured else 0,
        "eviction_mismatches": 0,
        **dict.fromkeys(audit.paper.PREFETCH_PROTECTION_COUNTERS, 0)}
    if mode != "native-off":
        dispatcher.update(prefetch_prediction_epoch=completed * 100,
                          prefetch_protected_resident_skips=50 if measured else 0,
                          prefetch_copy_started=12 if measured else 0)
    return {"mode": mode, "algorithm": "arxiv-2401.14361v3-reimplementation",
            "features": "shared-float64-EAMC-cosine-and-probability",
            "controller": controller, "dispatcher": dispatcher}


def engagement(after=False):
    count = 576 if after else 64
    return {"process_io": {"read_bytes": 100 if after else 0,
                           "cpu_time_s": 20 if after else 1,
                           "members": [{"pid": 123, "affinity": list(range(8))}]},
            "moe": {"revision": {"engine_generated_tokens": count, "engine_steps": count,
                "expert_cache_accesses": count * 10, "expert_cache_hits": count * 8,
                "expert_cache_misses": count * 2, "kv_cache_num_blocks": 128,
                "exposed_fetch_seconds_total": .25},
                "metrics": {"moe_tokens_generated_total": float(count),
                    "moe_engine_steps_total": float(count), "moe_kv_cache_total_blocks": 128.}}}


def fixture(directory):
    root = Path(directory)
    attempt = root / "block-01-attempt-01"
    attempt.mkdir()
    source = root / "source.py"
    source.write_text("frozen fixture source\n")
    module = root / "nvidia-uvm.ko"
    module.write_bytes(b"ordinary metadata fixture, not a kernel module")
    model_view = root / "model-view"
    model_view.mkdir()
    (model_view / "config.json").write_text("{}")
    prompts_path, goldens_path = root / "prompts.json", root / "goldens.json"
    prompts = {"records": [{"prompt_token_ids": [number + 1] * 512} for number in range(9)]}
    goldens = {"warmup": {"text": "warmup"}, "goldens": [str(number) * 64 for number in range(1, 9)]}
    write_json(prompts_path, prompts)
    write_json(goldens_path, goldens)
    runtime = {"files": [audit.base.file_metadata(source)], "source": {"commit": "fixture-source", "path": str(root)},
        "models": {"model_view": str(model_view), "hf_snapshot": str(model_view),
                   "all_sizes": {"config.json": 2}, "view_members": ["config.json"]},
        "driver_stage": str(root), "driver_stage_module": audit.base.file_metadata(module),
        "prompts": audit.base.file_metadata(prompts_path), "goldens": audit.base.file_metadata(goldens_path)}
    item = {"block": 1, "modes": list(audit.paper.MODES), "prompts": [8, 2, 5, 3, 1, 7, 4, 6]}
    block = {**item, "passed": True, "cells": []}
    for index, mode in enumerate(item["modes"]):
        cell = attempt / mode
        cell.mkdir()
        admission = {"runtime_inventory": runtime, "files": runtime["files"],
                     "source": runtime["source"], "models": runtime["models"],
                     "driver_stage_declared_by_coordinator": runtime["driver_stage"],
                     "driver_stage_module": runtime["driver_stage_module"], "safety": safety()}
        write_json(cell / "admission.json", admission)
        # Exercise the real launch writer, replacing only actual process start.
        with mock.patch.object(audit.paper.subprocess, "Popen", return_value=object()):
            _, log = audit.paper.launch(mode, cell, 18230, verify=False)
            log.close()
        log_text = "Model loaded and cleanly stopped\n"
        if mode == "paper-bpf":
            for kind, calls in (("paper_rank", 20), ("paper_match", 16), ("paper_scored", 20)):
                log_text += f"moe_expert_policy_ready: backend=ubpf-jit kind={kind} abi=1 instructions=61\n"
                log_text += (f"moe_expert_policy_stats: backend=ubpf-jit kind={kind} calls={calls} "
                             "candidates=1000 selected=50 no_victim=0 errors=0\n")
        (cell / "server.log").write_text(log_text)
        telemetry = cell / "gpu-telemetry.csv"
        telemetry.write_text("timestamp, memory.used, temperature.gpu, power.draw, clocks.sm, clocks.mem, "
                             "clocks_event_reasons.sw_power_cap, clocks_event_reasons.hw_thermal_slowdown\n"
                             "now, 20000, 60, 350, 2000, 14000, Active, Not Active\n")
        start = 1000000000 + index * 100000000000
        warm_raw = {"choices": [{"text": "warmup", "finish_reason": "length", "index": 0}],
                    "usage": {"prompt_tokens": 512, "completion_tokens": 64, "total_tokens": 576}}
        write_json(cell / "warmup.json", warm_raw)
        warm = {**audit.base.validate_completion_response(warm_raw, 512),
                "start_ns": start - 100000000, "end_ns": start - 10000000, "e2e_ms": 90.}
        requests = []
        for seq, number in enumerate(item["prompts"], 1):
            req_start = start + seq * 1000000000
            frames, lines = [], []
            for token in range(64):
                value = {"choices": [{"index": 0, "text": str(number),
                          "finish_reason": "length" if token == 63 else None}]}
                data = json.dumps(value).encode()
                lines.extend((b"data: " + data + b"\n", b"\n"))
                frames.append({"timestamp_ns": req_start + (token + 1) * 1000000, "payload_bytes": len(data)})
            done, eof = req_start + 65000000, req_start + 66000000
            frames.append({"timestamp_ns": done, "done": True})
            raw = b"".join(lines) + b"data: [DONE]\n\n"
            request = {"passed": True, "http_status": 200, "frames": frames,
                "request_payload": audit.base.completion_payload(runner.CONFIG,
                                      prompts["records"][number]["prompt_token_ids"], True),
                "start_ns": req_start, "first_text_ns": req_start + 1000000,
                "done_ns": done, "eof_ns": eof, "raw_sse_bytes": len(raw),
                "text": str(number) * 64, "finish_reason": "length", "ttft_ms": 1., "e2e_ms": 66.}
            path = cell / f"request-{seq:02d}-prompt-{number}.sse"
            path.write_bytes(raw)
            write_json(path.with_suffix(".json"), request)
            requests.append(request)
        ended = requests[-1]["eof_ns"] + 10000000
        duration = (ended - start) / 1e9
        before, after = engagement(), engagement(True)
        act_before, act_after = activation(mode, "before"), activation(mode, "after")
        result = {"passed": True, "protocol": runner.PROTOCOL, "mode": mode,
            "prompt_order": item["prompts"], "shadow_verification": False,
            "execution_domain": "host-ubpf-jit" if mode == "paper-bpf" else "native",
            "identity": {"models": {"data": [{"id": "gpt-oss-120b"}]}}, "warmup": warm,
            "requests": requests, "block_start_ns": start, "block_end_ns": ended,
            "duration_s": duration, "final_drain_s": .01, "verified_output_tokens": 512,
            "output_throughput_tokens_per_s": 512 / duration,
            "request_only_tokens_per_s": 512000 / sum(r["e2e_ms"] for r in requests),
            "first_text_ttft_median_ms": 1., "first_text_ttft_max_ms": 1.,
            "e2e_median_ms": 66., "e2e_max_ms": 66.,
            "activation_cold": activation(mode, "cold"), "activation_before": act_before,
            "activation_after": act_after, "activation_delta": runner.activation_delta(mode, act_before, act_after),
            "engagement_before": before, "engagement_after": after,
            "engagement_delta": audit.base.validate_measured_engagement(runner.CONFIG, before, after, current_deployment=True),
            "gpu_telemetry": audit.base.validate_gpu_telemetry(telemetry, allow_fixed_power_cap=True),
            "cleanup_errors": [], "server_exit_code": 0, "safety_after": safety(),
            "interrupt_warnings_before": [], "interrupt_warnings_after": []}
        write_json(cell / "result.json", result)
        block["cells"].append(result)
    write_json(attempt / "result.json", block)
    return attempt, item, runtime


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.attempt, self.item, self.runtime = fixture(self.temp.name)

    def run_audit(self):
        # The entire auditor must remain offline, including its safety checks.
        with mock.patch.object(audit.base, "run_checked", side_effect=AssertionError("subprocess forbidden")), \
             mock.patch.object(audit.base, "http_json", side_effect=AssertionError("HTTP forbidden")):
            return audit.audit_block(self.attempt, self.item, self.runtime)

    def mutate_cell(self, mutate, mode="paper-bpf"):
        path = self.attempt / mode / "result.json"
        value = json.loads(path.read_text())
        mutate(value)
        write_json(path, value)
        block_path = self.attempt / "result.json"
        block = json.loads(block_path.read_text())
        block["cells"][self.item["modes"].index(mode)] = value
        write_json(block_path, block)

    def rejects(self, pattern=None):
        with self.assertRaisesRegex(audit.AuditError, pattern or "resume audit refused"):
            self.run_audit()

    def test_full_three_mode_evidence_passes_without_live_operations(self):
        self.assertEqual(self.run_audit(), json.loads((self.attempt / "result.json").read_text()))

    def test_does_not_write_any_evidence(self):
        before = {str(p): audit.base.file_metadata(p) for p in self.attempt.rglob("*") if p.is_file()}
        self.run_audit()
        after = {str(p): audit.base.file_metadata(p) for p in self.attempt.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_embedded_cell_must_match_disk(self):
        path = self.attempt / "paper-bpf/result.json"
        value = json.loads(path.read_text())
        value["passed"] = False
        write_json(path, value)
        self.rejects("block/cell")

    def test_exact_schedule_and_three_cells_required(self):
        for mutate in (lambda b: b["cells"].pop(), lambda b: b["modes"].reverse(),
                       lambda b: b.update(error="old failure")):
            path = self.attempt / "result.json"
            original = path.read_text()
            value = json.loads(original)
            mutate(value)
            write_json(path, value)
            self.rejects()
            path.write_text(original)

    def test_launch_rejects_shadow_inherited_injection_and_wrong_policy(self):
        path = self.attempt / "paper-bpf/launch.json"
        original = path.read_text()
        for key, value in (("MOE_REVISION_VERIFY", "1"), ("LD_PRELOAD", "injected.so"),
                           ("MOE_REVISION_POLICY", "paper-native")):
            obj = json.loads(original)
            obj["env"][key] = value
            write_json(path, obj)
            self.rejects("launch differs")
        path.write_text(original)

    def test_runtime_file_actual_change_rejected_even_with_equal_cell_copies(self):
        Path(self.runtime["files"][0]["path"]).write_text("changed\n")
        self.rejects("metadata changed")

    def test_golden_change_rejected(self):
        Path(self.runtime["goldens"]["path"]).write_text("{}")
        self.rejects("metadata changed")

    def test_missing_artifact_and_alias_refused(self):
        path = self.attempt / "paper-bpf/request-01-prompt-8.sse"
        raw = path.read_bytes()
        path.unlink()
        self.rejects()
        alternate = Path(self.temp.name) / "outside.sse"
        alternate.write_bytes(raw)
        path.symlink_to(alternate)
        self.rejects("nonregular")

    def test_raw_same_size_text_corruption_rejected(self):
        path = self.attempt / "paper-bpf/request-01-prompt-8.sse"
        path.write_bytes(path.read_bytes().replace(b'"text": "8"', b'"text": "9"', 1))
        self.rejects("golden text differs")

    def test_raw_truncation_and_extra_data_not_accepted(self):
        path = self.attempt / "paper-bpf/request-01-prompt-8.sse"
        raw = path.read_bytes()
        for changed in (raw[:-15], raw + b"data: [DONE]\n\n"):
            path.write_bytes(changed)
            self.rejects()

    def test_payload_and_timestamp_corruption_survive_embedded_sync(self):
        mode = self.attempt / "paper-bpf"
        path = mode / "request-01-prompt-8.json"
        original = path.read_text()
        for mutate in (lambda r: r["request_payload"].update(prompt=[17] * 512),
                       lambda r: r.update(first_text_ns=r["first_text_ns"] + 1),
                       lambda r: r["frames"][1].update(timestamp_ns=1),
                       lambda r: r.pop("request_payload")):
            obj = json.loads(original)
            mutate(obj)
            write_json(path, obj)
            self.mutate_cell(lambda c: c["requests"].__setitem__(0, obj))
            self.rejects()

    def test_rejects_engine_511_even_when_producer_delta_claims512(self):
        self.mutate_cell(lambda c: c["engagement_after"]["moe"]["revision"].update(engine_generated_tokens=575))
        self.rejects("token/step gate")

    def test_rejects_fractional_counter_instead_of_truncating(self):
        self.mutate_cell(lambda c: c["engagement_after"]["moe"]["metrics"].update(moe_tokens_generated_total=576.5))
        self.rejects("noninteger")

    def test_activation_requires_every_jit_program_in_measured_window(self):
        self.mutate_cell(lambda c: c["activation_after"]["controller"].update(bpf_match_calls=0))
        self.rejects("programs engaged")

    def test_prefetch_conservation_not_just_nonzero(self):
        self.mutate_cell(lambda c: c["activation_after"]["dispatcher"].update(prefetch_hits=10))
        self.rejects("conservation")

    def test_timing_numbers_recomputed(self):
        self.mutate_cell(lambda c: c.update(output_throughput_tokens_per_s=9999))
        self.rejects("recomputed")

    def test_failed_cleanup_never_hidden_by_passed(self):
        self.mutate_cell(lambda c: c["safety_after"].update(uvm_refcount=1))
        self.rejects("reference count")

    def test_shutdown_failure_is_rejected(self):
        self.mutate_cell(lambda c: c.update(server_exit_code=-9))
        self.rejects("teardown")

    def test_zero_before_after_delta_cannot_hide_baseline_policy_state(self):
        self.mutate_cell(lambda c: c["activation_cold"]["dispatcher"].update(prefetch_submitted=1), mode="native-off")
        self.rejects("baseline contains")

    def test_telemetry_partial_row_not_silently_discarded(self):
        path = self.attempt / "paper-bpf/gpu-telemetry.csv"
        path.write_text(path.read_text() + "truncated, 30000\n")
        self.rejects("incomplete GPU telemetry")

    def test_jit_log_counters_must_agree_with_activation(self):
        path = self.attempt / "paper-bpf/server.log"
        path.write_text(path.read_text().replace("calls=16", "calls=15"))
        self.rejects("shutdown counters")

    def test_log_fatal_and_missing_jit_ready_rejected(self):
        path = self.attempt / "paper-bpf/server.log"
        original = path.read_text()
        path.write_text(original + "CUDA error: broken\n")
        self.rejects()
        path.write_text(original.replace("moe_expert_policy_ready", "missing_ready"))
        self.rejects("programs not ready")

    def test_duplicate_json_keys_and_nonfinite_numbers_rejected(self):
        path = self.attempt / "result.json"
        original = path.read_text()
        path.write_text(original[:-1] + ', "passed": true}')
        self.rejects("duplicate JSON key")
        path.write_text(original[:-1] + ', "extra": NaN}')
        self.rejects("nonfinite")


if __name__ == "__main__":
    unittest.main()
