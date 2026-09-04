#!/usr/bin/env python3
"""CPU-only tests for the fixed-work profile and raw-evidence analyzer."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import analyze_fixed_work as analyzer
import run_fixed_work as fixed


runner = fixed.runner


def measurement(cell: dict, elapsed_ms: float) -> dict:
    phase = runner.phase_parameters("full")
    return {
        "event": "measurement",
        **analyzer.expected_measurement(cell, phase),
        "elapsed_ms": elapsed_ms,
    }


def safety_snapshot(timestamp_ns: int) -> dict:
    return {
        "timestamp_ns": timestamp_ns,
        "power_limit_service": "active",
        "power_limit_w": 400.0,
        "gpu": {
            "index": 0,
            "name": runner.EXPECTED_GPU,
            "driver": runner.EXPECTED_DRIVER,
            "memory_used_mib": 16,
            "memory_total_mib": 32607,
            "temperature_c": 40,
            "sm_clock_mhz": 22,
            "memory_clock_mhz": 405,
            "power_w": 15.0,
            "utilization_gpu_percent": 0,
            "compute_apps": [],
        },
        "uvm_refcount": 0,
        "struct_ops": {"maps": [], "links": []},
        "dmesg_abnormal": [],
        "journal_abnormal": [],
        "xids": [],
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def write_json_lines(path: Path, records: list[dict], prefix: str = "") -> None:
    text = prefix + "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(text)


def application_events(cells: list[dict], values: list[dict], run_id: int) -> list[dict]:
    return [
        {
            "event": "device", "name": runner.EXPECTED_GPU,
            "major": 12, "minor": 0, "warp_size": 32,
            "max_threads_per_block": 1024, "max_grid_x": 2_147_483_647,
        },
        {"event": "marker", "threads": 32, "mismatches": 0},
        *values,
        {"event": "complete", "cells": len(cells), "run_id": run_id},
    ]


def loader_events(mode: str) -> list[dict]:
    records = [{
        "event": "ready", "mode": mode, "programs": 2,
        "gpu_threads": runner.MAX_THREADS, "target_map": mode == "counter",
        "attach_order": ["cuda__scale_target", "cuda__scale_marker"],
    }]
    segments = {
        ("marker_count", 0): [
            {"begin": 0, "end": 32, "value": 1},
            {"begin": 32, "end": runner.MAX_THREADS, "value": 0},
        ]
    }
    if mode == "counter":
        phase = runner.phase_parameters("full")
        increment = (
            (phase["warmup"] + phase["launches"]) * phase["hook_repeats"]
        )
        for cell in runner.CELLS:
            segments[("target_count", cell["counter_key"])] = [
                {"begin": 0, "end": cell["active_threads"], "value": increment},
                {"begin": cell["active_threads"], "end": runner.MAX_THREADS,
                 "value": 0},
            ]
    for (name, key), values in segments.items():
        records.extend(
            {"event": "counter_segment", "map": name, "key": key, **value}
            for value in values
        )
    records.append({"event": "detached", "links": 2})
    return records


TRANSFORM_LOG = """Recorded pass /runtime/ptxpass_kprobe_entry.so for func trampoline_scale_kernel
[ptxpass] kprobe_entry_stub: matched=1, in=1, out=2
Recorded pass /runtime/ptxpass_kprobe_entry.so for func trampoline_marker_kernel
[ptxpass] kprobe_entry: matched=1, in=1, out=2
Loaded module: fixed-work.ptx
Attach successfully
"""


TELEMETRY = """timestamp, memory.used [MiB], temperature.gpu, power.draw [W], clocks.current.sm [MHz], clocks.current.memory [MHz], clocks_event_reasons.sw_power_cap, clocks_event_reasons.hw_slowdown, clocks_event_reasons.hw_thermal_slowdown, clocks_event_reasons.hw_power_brake_slowdown, clocks_event_reasons.sw_thermal_slowdown
2026/09/04 00:00:00.000, 16 MiB, 40, 40 W, 2400 MHz, 14001 MHz, Not Active, Not Active, Not Active, Not Active, Not Active
"""


def result_fixture(
    output: Path, endpoint_effect_pct: float = 0.0,
    middle_extra_ms: float = 0.0,
) -> tuple[dict, Path]:
    phase = runner.phase_parameters("full")
    schedule = runner.frozen_schedule("full")
    by_id = {cell["id"]: cell for cell in runner.CELLS}
    output.mkdir()
    write_json(output / "safety-before.json", safety_snapshot(1_000_000))
    write_json(output / "safety-final.json", safety_snapshot(2_000_000))
    records = []
    for sequence, item in enumerate(schedule):
        arm = item["arm"]
        cells = [by_id[cell_id] for cell_id in item["cell_ids"]]
        values = []
        for cell in cells:
            cell_id = cell["id"]
            native = 10.0 + cell_id / 1000.0 + item["block"] / 10_000.0
            delta = 0.0
            if arm == "noop":
                delta = 0.1
                if cell_id == analyzer.HIGH_BLOCK_CELL:
                    low_native = 10.0 + analyzer.LOW_BLOCK_CELL / 1000.0
                    high_native = 10.0 + analyzer.HIGH_BLOCK_CELL / 1000.0
                    delta += endpoint_effect_pct * (low_native + high_native) / 200.0
                if cell_id == 2:
                    delta += middle_extra_ms
            elif arm == "counter":
                delta = 0.2
            values.append(measurement(cell, native + delta))

        name = f"block-{item['block'] + 1:02d}-order-{item['order'] + 1}-{arm}"
        run_dir = output / name
        run_dir.mkdir()
        write_json_lines(
            run_dir / "application.log",
            application_events(cells, values, item["run_id"]),
            prefix=TRANSFORM_LOG if arm != "baseline" else "",
        )
        telemetry_dir = output / f"telemetry-{sequence + 1:02d}"
        telemetry_dir.mkdir()
        (telemetry_dir / "gpu-telemetry.csv").write_text(TELEMETRY)
        before = safety_snapshot(3_000_000 + sequence * 2)
        after = safety_snapshot(3_000_001 + sequence * 2)
        write_json(run_dir / "safety-before.json", before)
        write_json(run_dir / "safety-after.json", after)

        command = runner.application_command(
            tuple(item["cell_ids"]), phase["warmup"], phase["launches"],
            phase["hook_repeats"], item["run_id"],
        )
        lifecycle = {
            "schema": runner.RAW_EVIDENCE_SCHEMA,
            "experiment_kind": runner.EXPERIMENT_KIND,
            "block": item["block"], "order": item["order"], "arm": arm,
            "run_id": item["run_id"], "cell_ids": list(item["cell_ids"]),
            "application_command": command,
            "application_returncode": 0,
            "application_log": "application.log",
            "owned_group_survivors": {},
            "telemetry_log": f"../telemetry-{sequence + 1:02d}/gpu-telemetry.csv",
            "safety_before": "safety-before.json",
            "safety_after": "safety-after.json",
        }
        if arm == "baseline":
            lifecycle.update(
                loader_command=None, loader_returncode=None, loader_log=None,
                agent_log=None, private_segment=None, private_segment_removed=None,
            )
        else:
            segment = f"trampoline_scaling_999_{sequence + 1}"
            object_path = runner.HERE / ".output" / f"{runner.BPF_OBJECT_PREFIX}-{arm}.bpf.o"
            lifecycle.update(
                loader_command=[
                    str(runner.LOADER_BINARY), str(object_path), arm,
                    str(runner.MAX_THREADS), "300",
                ],
                loader_returncode=0,
                loader_log="loader.log",
                agent_log="agent.log",
                private_segment=segment,
                private_segment_removed=True,
            )
            write_json_lines(run_dir / "loader.log", loader_events(arm))
            (run_dir / "agent.log").write_text(
                "Verifier mode: WARNING\n"
                "Registered shared memory with CUDA: addr=0x1 size=1\n"
                f"Global shm constructed. shm_open_type 1 for {segment}\n"
                "Global shm initialized\n"
            )
        write_json(run_dir / "lifecycle.json", lifecycle)
        records.append({
            "valid": True, "block": item["block"], "order": item["order"],
            "arm": arm, "directory": str(run_dir),
            "measurements": copy.deepcopy(values),
            "telemetry": {"summary": {"samples": 1}},
            "safety_after": copy.deepcopy(after),
            "engagement": {"untrusted": True},
            "agent_gate": {"untrusted": True},
        })
    result = {
        "kind": runner.EXPERIMENT_KIND,
        "status": "complete",
        "failures": [{"untrusted": "ignored"}],
        "params": {
            "kind": runner.EXPERIMENT_KIND,
            "phase": "full", "blocks": phase["blocks"],
            "cell_ids": list(phase["cell_ids"]),
            "warmup": phase["warmup"], "launches": phase["launches"],
            "hook_repeats": phase["hook_repeats"],
            "schedule_seed": runner.SEED,
            "expected_driver": runner.EXPECTED_DRIVER,
            "expected_gpu": runner.EXPECTED_GPU,
            "matrix": [dict(cell) for cell in runner.CELLS],
            "randomize_cell_order": True,
            "balance_arm_order": True,
            "independent_raw_evidence": True,
        },
        "schedule": schedule,
        "records": records,
        "safety_after": {"untrusted": True},
    }
    result_path = output / "result.json"
    write_json(result_path, result)
    return result, result_path


def find_run_dir(result: dict, block: int, arm: str) -> Path:
    record = next(
        item for item in result["records"]
        if item["block"] == block and item["arm"] == arm
    )
    return Path(record["directory"])


def mutate_json_event(path: Path, event: str, field: str, value: object) -> None:
    lines = path.read_text().splitlines()
    output = []
    changed = False
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            output.append(line)
            continue
        if not changed and record.get("event") == event:
            record[field] = value
            line = json.dumps(record, sort_keys=True)
            changed = True
        output.append(line)
    if not changed:
        raise AssertionError(f"event not found: {event}")
    path.write_text("\n".join(output) + "\n")


class FixedWorkProfileTests(unittest.TestCase):
    def test_matrix_holds_total_work_and_dynamic_warps_fixed(self) -> None:
        self.assertEqual([cell["blocks"] for cell in runner.CELLS],
                         [128, 256, 1024, 2048, 4096])
        self.assertEqual(
            {cell["blocks"] * cell["threads_per_block"] for cell in runner.CELLS},
            {131_072},
        )
        self.assertEqual({cell["active_threads"] for cell in runner.CELLS}, {131_072})
        self.assertEqual({cell["active_threads"] // 32 for cell in runner.CELLS}, {4096})
        self.assertTrue(all(cell["threads_per_block"] % 32 == 0 for cell in runner.CELLS))

    def test_compiled_header_contains_exact_profile(self) -> None:
        text = (Path(__file__).resolve().parent / "fixed_work_matrix.h").read_text()
        for cell in runner.CELLS:
            marker = (
                f"X({cell['id']}, {cell['blocks']}, {cell['threads_per_block']}, "
                f"{cell['active_threads']}, {cell['counter_key']})"
            )
            self.assertIn(marker, text)
            geometry = (
                f"X({cell['blocks']}, {cell['threads_per_block']}, "
                f"{cell['counter_key']})"
            )
            self.assertIn(geometry, text)
        self.assertIn("#define SCALE_CELL_COUNT 5", text)
        self.assertEqual(len({(cell["blocks"], cell["threads_per_block"])
                              for cell in runner.CELLS}), 5)
        self.assertEqual({cell["counter_key"] for cell in runner.CELLS}, set(range(5)))

    def test_profile_uses_separate_compiled_artifacts(self) -> None:
        self.assertEqual(runner.APPLICATION_BINARY.name, "fixed-work-scaling")
        self.assertEqual(runner.COMPILED_PTX.name, "fixed-work-scaling.ptx")
        self.assertEqual(runner.LOADER_BINARY.name, "fixed-work-probe")
        self.assertEqual(runner.BPF_OBJECT_PREFIX, "fixed-work-probe")
        self.assertEqual(runner.MATRIX_HEADER.name, "fixed_work_matrix.h")
        self.assertTrue(runner.BALANCE_ARM_ORDER)
        self.assertTrue(runner.WRITE_INDEPENDENT_RAW_EVIDENCE)

    def test_raw_evidence_writer_creates_independent_lifecycle_and_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_dir = output / "block-01-order-1-baseline"
            run_dir.mkdir()
            telemetry_dir = output / "telemetry-01"
            telemetry_dir.mkdir()
            telemetry_path = telemetry_dir / "gpu-telemetry.csv"
            telemetry_path.write_text(TELEMETRY)
            item = {
                "block": 0, "order": 0, "arm": "baseline", "run_id": 0,
            }
            phase = runner.phase_parameters("full")
            command = runner.application_command(
                phase["cell_ids"], phase["warmup"], phase["launches"],
                phase["hook_repeats"], 0,
            )
            before = safety_snapshot(10)
            after = safety_snapshot(11)
            runner.write_raw_arm_evidence(
                output, run_dir, item, phase["cell_ids"],
                {"command": command, "application_returncode": 0},
                telemetry_path, before, after,
            )
            lifecycle = json.loads((run_dir / "lifecycle.json").read_text())
            self.assertEqual(lifecycle["application_command"], command)
            self.assertEqual(lifecycle["safety_before"], "safety-before.json")
            self.assertEqual(lifecycle["safety_after"], "safety-after.json")
            self.assertEqual(json.loads((run_dir / "safety-before.json").read_text()), before)
            self.assertEqual(json.loads((run_dir / "safety-after.json").read_text()), after)

    def test_cell_order_is_randomized_and_arm_order_is_balanced(self) -> None:
        schedule = runner.frozen_schedule("full")
        cell_orders = []
        position_counts = {arm: [0, 0, 0] for arm in runner.ARMS}
        for block in range(10):
            items = [item for item in schedule if item["block"] == block]
            cells = [tuple(item["cell_ids"]) for item in items]
            self.assertEqual(len(cells), 3)
            self.assertEqual(len(set(cells)), 1)
            cell_orders.append(cells[0])
            for item in items:
                position_counts[item["arm"]][item["order"]] += 1
        self.assertGreater(len(set(cell_orders)), 1)
        self.assertEqual(schedule, runner.frozen_schedule("full"))
        for counts in position_counts.values():
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_core_application_gate_accepts_variable_block_dimensions(self) -> None:
        phase = runner.phase_parameters("full")
        cells = list(runner.CELLS)
        records = application_events(
            cells, [measurement(cell, 1.0 + cell["id"] / 100.0) for cell in cells], 0,
        )
        actual = runner.validate_application_events(
            records, phase["cell_ids"], phase["warmup"], phase["launches"],
            phase["hook_repeats"], 0,
        )
        self.assertEqual([item["threads_per_block"] for item in actual],
                         [1024, 512, 128, 64, 32])

    def test_full_counter_oracle_is_exact_for_every_fixed_work_cell(self) -> None:
        phase = runner.phase_parameters("full")
        oracle = runner.expected_counter_segments(
            phase["cell_ids"], phase["warmup"], phase["launches"],
            phase["hook_repeats"],
        )
        for key in range(5):
            self.assertEqual(oracle[("target_count", key)], [
                {"begin": 0, "end": 131_072, "value": 160},
                {"begin": 131_072, "end": runner.MAX_THREADS, "value": 0},
            ])


class FixedWorkAnalysisTests(unittest.TestCase):
    def test_constant_increment_supports_bounded_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "raw")
            analysis = analyzer.analyze(result, path)
        self.assertEqual(analysis["run_status"], "valid")
        self.assertEqual(analysis["tested_hypothesis"],
                         "supported_within_predeclared_bound")
        self.assertEqual(analysis["organization_guard"]["status"],
                         "supported_within_predeclared_bound")
        self.assertAlmostEqual(analysis["primary_metric"]["median"], 0.0, places=9)
        self.assertEqual(analysis["raw_evidence_audit"]["timed_cells"], 150)
        self.assertEqual(len(analysis["cells"]), 5)

    def test_material_endpoint_effect_contradicts_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(
                Path(temporary) / "raw", endpoint_effect_pct=3.0,
            )
            analysis = analyzer.analyze(result, path)
        self.assertEqual(analysis["tested_hypothesis"], "contradicted")
        self.assertGreater(analysis["primary_metric"]["ci95_low"], 1.0)

    def test_middle_organization_plus_5_1_ms_fails_all_five_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(
                Path(temporary) / "raw", middle_extra_ms=5.1,
            )
            analysis = analyzer.analyze(result, path)
        self.assertEqual(analysis["primary_metric"]["status"],
                         "supported_within_predeclared_bound")
        middle = next(
            item for item in analysis["organization_guard"]["contrasts"]
            if item["cell"] == 2
        )
        self.assertGreater(middle["ci_low"], 1.0)
        self.assertEqual(analysis["organization_guard"]["status"], "contradicted")
        self.assertEqual(analysis["tested_hypothesis"], "contradicted")

    def test_result_derived_dictionaries_are_not_analysis_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "raw")
            result["records"][0].update(
                valid=False,
                measurements=[{"elapsed_ms": 999999.0}],
                telemetry={"summary": None},
                safety_after=None,
                engagement={"target_counter_exact": False},
                agent_gate={"routing_order_valid": False},
            )
            analysis = analyzer.analyze(result, path)
        self.assertEqual(analysis["tested_hypothesis"],
                         "supported_within_predeclared_bound")

    def test_missing_or_changed_application_raw_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "missing")
            (find_run_dir(result, 0, "baseline") / "application.log").unlink()
            with self.assertRaisesRegex(analyzer.AnalysisError, "missing raw application"):
                analyzer.analyze(result, path)
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "changed")
            app = find_run_dir(result, 0, "baseline") / "application.log"
            mutate_json_event(app, "measurement", "active_warps", 2048)
            with self.assertRaisesRegex(analyzer.AnalysisError, "application.*gate failed"):
                analyzer.analyze(result, path)

    def test_changed_loader_agent_and_telemetry_raw_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "loader")
            loader = find_run_dir(result, 0, "counter") / "loader.log"
            mutate_json_event(loader, "counter_segment", "value", 999)
            with self.assertRaisesRegex(analyzer.AnalysisError, "loader/map engagement"):
                analyzer.analyze(result, path)
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "agent")
            (find_run_dir(result, 0, "noop") / "agent.log").write_text(
                "Verifier mode: WARNING\n"
            )
            with self.assertRaisesRegex(analyzer.AnalysisError, "agent bootstrap"):
                analyzer.analyze(result, path)
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "telemetry")
            run_dir = find_run_dir(result, 0, "baseline")
            lifecycle = json.loads((run_dir / "lifecycle.json").read_text())
            telemetry = (run_dir / lifecycle["telemetry_log"]).resolve()
            telemetry.write_text(TELEMETRY.replace(
                "Not Active, Not Active, Not Active, Not Active, Not Active",
                "Not Active, Active, Not Active, Not Active, Not Active",
            ))
            with self.assertRaisesRegex(analyzer.AnalysisError, "telemetry.*gate failed"):
                analyzer.analyze(result, path)

    def test_changed_safety_and_lifecycle_raw_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "safety")
            after_path = find_run_dir(result, 0, "baseline") / "safety-after.json"
            after = json.loads(after_path.read_text())
            after["uvm_refcount"] = 1
            write_json(after_path, after)
            with self.assertRaisesRegex(analyzer.AnalysisError, "post-safety"):
                analyzer.analyze(result, path)
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "lifecycle")
            lifecycle_path = find_run_dir(result, 0, "noop") / "lifecycle.json"
            lifecycle = json.loads(lifecycle_path.read_text())
            lifecycle["application_returncode"] = 1
            write_json(lifecycle_path, lifecycle)
            with self.assertRaisesRegex(analyzer.AnalysisError, "lifecycle identity"):
                analyzer.analyze(result, path)

    def test_reused_arm_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "raw")
            result["records"][1]["directory"] = result["records"][0]["directory"]
            with self.assertRaisesRegex(
                analyzer.AnalysisError, "directory name does not match|reused",
            ):
                analyzer.analyze(result, path)

    def test_render_and_json_round_trip_are_cpu_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = result_fixture(Path(temporary) / "raw")
            analysis = analyzer.analyze(result, path)
            rendered = analyzer.render_markdown(analysis)
            self.assertIn("All-five organization guard", rendered)
            self.assertEqual(json.loads(json.dumps(analysis)), analysis)
            output = Path(temporary) / "analysis.md"
            output.write_text(rendered)
            self.assertTrue(output.read_text().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
