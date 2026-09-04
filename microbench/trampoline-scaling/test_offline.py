#!/usr/bin/env python3
"""CPU-only fail-closed tests for the trampoline-scaling harness."""

from __future__ import annotations

import fcntl
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("trampoline_scaling_runner", HERE / "run_scaling.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def device_event() -> dict:
    return {
        "event": "device", "name": runner.EXPECTED_GPU,
        "major": 12, "minor": 0, "warp_size": 32,
        "max_threads_per_block": 1024, "max_grid_x": 2_147_483_647,
    }


def measurement_event(cell: dict, warmup: int, launches: int, repeats: int) -> dict:
    return {
        "event": "measurement", "cell": cell["id"], "blocks": cell["blocks"],
        "threads_per_block": runner.THREADS_PER_BLOCK,
        "launched_threads": cell["blocks"] * runner.THREADS_PER_BLOCK,
        "active_threads": cell["active_threads"],
        "active_warps": cell["active_threads"] // 32,
        "counter_key": cell["counter_key"], "warmup": warmup,
        "launches": launches, "hook_repeats": repeats,
        "elapsed_ms": 1.25 + cell["id"], "checked_values": runner.MAX_THREADS,
        "mismatches": 0,
    }


def application_fixture(
    cell_ids: tuple[int, ...] = (0,), warmup: int = 1,
    launches: int = 2, repeats: int = 1, run_id: int = 0,
) -> list[dict]:
    cells = runner.selected_cells(cell_ids)
    return [
        device_event(),
        {"event": "marker", "threads": 32, "mismatches": 0},
        *(measurement_event(cell, warmup, launches, repeats) for cell in cells),
        {"event": "complete", "cells": len(cells), "run_id": run_id},
    ]


def loader_fixture(
    mode: str, cell_ids: tuple[int, ...] = (0,), warmup: int = 1,
    launches: int = 2, repeats: int = 1,
) -> list[dict]:
    expected = runner.expected_counter_segments(cell_ids, warmup, launches, repeats)
    if mode == "noop":
        expected = {key: value for key, value in expected.items() if key[0] == "marker_count"}
    records = [{
        "event": "ready", "mode": mode, "programs": 2,
        "gpu_threads": runner.MAX_THREADS, "target_map": mode == "counter",
    }]
    for (name, key), segments in expected.items():
        records.extend({"event": "counter_segment", "map": name, "key": key, **segment}
                       for segment in segments)
    records.append({"event": "detached", "links": 2})
    return records


def minimal_ptx(*, target_calls: int = 1, marker_call: bool = False) -> str:
    target = "\n".join(
        "call.uni __bpftime_cuda__kernel_trace, ();" for _ in range(target_calls)
    )
    marker = "call.uni __bpftime_cuda__kernel_trace, ();" if marker_call else "ret;"
    return f"""
.func __bpftime_cuda__kernel_trace()
{{
    ret;
}}
.visible .entry trampoline_marker_kernel()
{{
    {marker}
}}
.visible .entry trampoline_scale_kernel()
{{
    {target}
    ret;
}}
"""


class MatrixTests(unittest.TestCase):
    def test_frozen_axes_have_nine_unique_cells(self) -> None:
        self.assertEqual([cell["id"] for cell in runner.CELLS], list(range(9)))
        self.assertEqual(len({(cell["blocks"], cell["active_threads"])
                              for cell in runner.CELLS}), 9)
        block_axis = runner.CELLS[:5]
        self.assertEqual([cell["blocks"] for cell in block_axis], [256, 512, 1024, 2048, 4096])
        self.assertEqual({cell["active_threads"] for cell in block_axis}, {65_536})
        active_axis = (runner.CELLS[0], *runner.CELLS[5:])
        self.assertEqual({cell["blocks"] for cell in active_axis}, {256, 4096})
        fixed_geometry_axis = runner.CELLS[4:]
        self.assertEqual({cell["blocks"] for cell in fixed_geometry_axis}, {4096})
        self.assertEqual(
            [cell["active_threads"] // 32 for cell in fixed_geometry_axis],
            [2048, 4096, 8192, 16384, 32768],
        )

    def test_header_and_python_matrix_agree(self) -> None:
        text = (HERE / "matrix.h").read_text()
        for cell in runner.CELLS:
            marker = (f"X({cell['id']}, {cell['blocks']}, "
                      f"{cell['active_threads']}, {cell['counter_key']})")
            self.assertIn(marker, text)
        self.assertIn("#define SCALE_CELL_COUNT 9", text)
        self.assertIn("#define SCALE_MAX_THREADS 1048576", text)

    def test_schedule_is_deterministic_and_balanced(self) -> None:
        first = runner.frozen_schedule("full")
        self.assertEqual(first, runner.frozen_schedule("full"))
        self.assertEqual(len(first), 30)
        for block in range(10):
            items = [item for item in first if item["block"] == block]
            self.assertEqual({item["arm"] for item in items}, set(runner.ARMS))
            self.assertEqual([item["order"] for item in items], [0, 1, 2])

    def test_command_carries_same_run_id_and_exact_cells(self) -> None:
        command = runner.application_command((0, 3, 8), 2, 8, 2, 7)
        self.assertEqual(command[-10:], [
            "--cells", "0,3,8", "--warmup", "2", "--launches", "8",
            "--hook-repeats", "2", "--run-id", "7",
        ])


class PtxAndRuntimeGateTests(unittest.TestCase):
    def test_accepts_one_target_stub_and_fallback_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scaling.ptx"
            path.write_text(minimal_ptx())
            result = runner.validate_compiled_hook_site(path)
        self.assertEqual(result["target_explicit_stub_calls"], 1)
        self.assertEqual(result["marker_explicit_stub_calls"], 0)

    def test_rejects_missing_or_duplicate_target_stub(self) -> None:
        for count in (0, 2):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "scaling.ptx"
                path.write_text(minimal_ptx(target_calls=count))
                with self.assertRaisesRegex(RuntimeError, "exactly one"):
                    runner.validate_compiled_hook_site(path)

    def test_rejects_marker_with_explicit_stub(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scaling.ptx"
            path.write_text(minimal_ptx(marker_call=True))
            with self.assertRaisesRegex(RuntimeError, "fallback"):
                runner.validate_compiled_hook_site(path)

    def test_runtime_configuration_is_pinned_and_verifier_limitation_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            (build / "CMakeCache.txt").write_text(
                "BPFTIME_ENABLE_CUDA_ATTACH:BOOL=ON\n"
                "BPFTIME_LLVM_JIT:BOOL=ON\n"
                "ENABLE_EBPF_VERIFIER:BOOL=OFF\n"
                "CMAKE_HOME_DIRECTORY:INTERNAL=/runtime\n"
            )
            config = runner.runtime_configuration(build)
            self.assertEqual(config["ENABLE_EBPF_VERIFIER"], "OFF")
            (build / "CMakeCache.txt").write_text(
                "BPFTIME_ENABLE_CUDA_ATTACH:BOOL=OFF\n"
                "BPFTIME_LLVM_JIT:BOOL=ON\n"
                "ENABLE_EBPF_VERIFIER:BOOL=OFF\n"
            )
            with self.assertRaisesRegex(RuntimeError, "feature mismatch"):
                runner.runtime_configuration(build)

    def test_runtime_source_audit_does_not_claim_warp_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "attach/nv_attach_impl/pass/ptxpass_kprobe_entry/main.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("\n".join((
                'params.save_strategy = "minimal";',
                'bool add_register_guard = params.save_strategy == "full";',
                'auto a = "call " + stub_name;',
                'auto b = "call.uni " + stub_name;',
                'log_transform_stats("kprobe_entry_stub", 1, 2, 3);',
                'log_transform_stats("kprobe_entry", 1, 2, 3);',
            )))
            audit = runner.audit_runtime_source(root)
        self.assertFalse(audit["register_guard_for_default"])
        self.assertIn("does not establish", audit["interpretation"])


class ApplicationGateTests(unittest.TestCase):
    def test_accepts_complete_correct_application_record(self) -> None:
        values = runner.validate_application_events(application_fixture((0, 8)), (0, 8), 1, 2, 1, 0)
        self.assertEqual([value["cell"] for value in values], [0, 8])

    def test_rejects_missing_marker(self) -> None:
        records = [item for item in application_fixture() if item["event"] != "marker"]
        with self.assertRaisesRegex(RuntimeError, "marker"):
            runner.validate_application_events(records, (0,), 1, 2, 1, 0)

    def test_rejects_duplicate_or_unknown_events(self) -> None:
        for extra in (
            {"event": "marker", "threads": 32, "mismatches": 0},
            {"event": "unexpected"},
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(RuntimeError):
                    runner.validate_application_events(
                        application_fixture() + [extra], (0,), 1, 2, 1, 0,
                    )

    def test_rejects_wrong_geometry_zero_time_or_mismatch(self) -> None:
        for field, value in (("blocks", 255), ("elapsed_ms", 0.0), ("mismatches", 1)):
            with self.subTest(field=field):
                records = application_fixture()
                next(item for item in records if item["event"] == "measurement")[field] = value
                with self.assertRaises(RuntimeError):
                    runner.validate_application_events(records, (0,), 1, 2, 1, 0)

    def test_json_parser_ignores_logs_but_preserves_json_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "log"
            path.write_text("runtime line\n" + json.dumps({"event": "complete"}) + "\n{broken\n")
            self.assertEqual(runner.json_events(path), [{"event": "complete"}])


class CounterOracleTests(unittest.TestCase):
    def test_preflight_oracle_is_exact(self) -> None:
        oracle = runner.expected_counter_segments((0,), 1, 2, 1)
        self.assertEqual(oracle[("marker_count", 0)], [
            {"begin": 0, "end": 32, "value": 1},
            {"begin": 32, "end": runner.MAX_THREADS, "value": 0},
        ])
        self.assertEqual(oracle[("target_count", 0)], [
            {"begin": 0, "end": 65_536, "value": 3},
            {"begin": 65_536, "end": runner.MAX_THREADS, "value": 0},
        ])
        for key in range(1, 5):
            self.assertEqual(oracle[("target_count", key)], [
                {"begin": 0, "end": runner.MAX_THREADS, "value": 0},
            ])

    def test_full_shared_key_has_nested_prefix_counts(self) -> None:
        oracle = runner.expected_counter_segments(tuple(range(9)), 2, 8, 2)
        self.assertEqual(oracle[("target_count", 4)], [
            {"begin": 0, "end": 65_536, "value": 100},
            {"begin": 65_536, "end": 131_072, "value": 80},
            {"begin": 131_072, "end": 262_144, "value": 60},
            {"begin": 262_144, "end": 524_288, "value": 40},
            {"begin": 524_288, "end": 1_048_576, "value": 20},
        ])

    def test_accepts_noop_and_counter_complete_readback(self) -> None:
        noop = runner.validate_loader_events(loader_fixture("noop"), "noop", (0,), 1, 2, 1)
        counter = runner.validate_loader_events(loader_fixture("counter"), "counter", (0,), 1, 2, 1)
        self.assertEqual(noop["marker_callbacks"], 32)
        self.assertTrue(counter["target_counter_exact"])

    def test_rejects_missing_marker_wrong_target_or_missing_detach(self) -> None:
        cases = []
        records = loader_fixture("counter")
        cases.append([item for item in records
                      if not (item.get("event") == "counter_segment"
                              and item.get("map") == "marker_count")])
        records = loader_fixture("counter")
        target = next(item for item in records
                      if item.get("map") == "target_count" and item.get("value") == 3)
        target["value"] = 2
        cases.append(records)
        cases.append([item for item in loader_fixture("counter") if item["event"] != "detached"])
        for records in cases:
            with self.subTest(events=len(records)), self.assertRaises(RuntimeError):
                runner.validate_loader_events(records, "counter", (0,), 1, 2, 1)

    def test_noop_rejects_target_map_records(self) -> None:
        records = loader_fixture("noop")
        records.insert(-1, {
            "event": "counter_segment", "map": "target_count", "key": 0,
            "begin": 0, "end": runner.MAX_THREADS, "value": 0,
        })
        with self.assertRaises(RuntimeError):
            runner.validate_loader_events(records, "noop", (0,), 1, 2, 1)


class AgentEvidenceTests(unittest.TestCase):
    LOG = "\n".join((
        "Recorded pass /runtime/libptxpass_kprobe_entry.so for func trampoline_marker_kernel",
        "Recorded pass /runtime/libptxpass_kprobe_entry.so for func trampoline_scale_kernel",
        "[ptxpass] kprobe_entry: matched=1, in=1, out=2",
        "[ptxpass] kprobe_entry_stub: matched=1, in=1, out=2",
        "Loaded module: scaling.fatbin",
        "Attach successfully",
    ))

    def test_accepts_exact_marker_and_target_transform_chain(self) -> None:
        result = runner.validate_agent_log(self.LOG)
        self.assertEqual(result["target_recorded"], 1)
        self.assertEqual(result["target_stub_transform"], 1)

    def test_marker_only_is_not_target_engagement(self) -> None:
        text = self.LOG.replace(
            "Recorded pass /runtime/libptxpass_kprobe_entry.so for func trampoline_scale_kernel\n", ""
        ).replace("[ptxpass] kprobe_entry_stub: matched=1, in=1, out=2\n", "")
        with self.assertRaises(RuntimeError):
            runner.validate_agent_log(text)

    def test_duplicate_target_record_is_rejected(self) -> None:
        duplicate = self.LOG + "\nRecorded pass /runtime/libptxpass_kprobe_entry.so for func trampoline_scale_kernel"
        with self.assertRaises(RuntimeError):
            runner.validate_agent_log(duplicate)


class LeaseAndEnvironmentTests(unittest.TestCase):
    def test_read_only_lease_preserves_precreated_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lease"
            path.write_text("pre-created\n")
            before = path.stat()
            with runner.ReadOnlyLeases((path,)):
                self.assertEqual(path.read_text(), "pre-created\n")
            after = path.stat()
            self.assertEqual((after.st_size, after.st_mtime_ns), (before.st_size, before.st_mtime_ns))

    def test_missing_lease_is_never_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing"
            with self.assertRaises(FileNotFoundError):
                runner.ReadOnlyLeases((path,))
            self.assertFalse(path.exists())

    def test_symlink_fifo_and_contended_lease_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "regular"
            regular.write_text("lock\n")
            symlink = root / "symlink"
            symlink.symlink_to(regular)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            for path in (symlink, fifo):
                with self.subTest(path=path.name), self.assertRaises(RuntimeError):
                    runner.ReadOnlyLeases((path,))
            with regular.open("r") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    runner.ReadOnlyLeases((regular,))

    def test_ambient_injection_is_rejected(self) -> None:
        runner.reject_ambient_injection({"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0"})
        for environment in (
            {"LD_PRELOAD": "/tmp/x.so"},
            {"BPFTIME_GLOBAL_SHM_NAME": "other"},
            {"CUDA_VISIBLE_DEVICES": "1"},
        ):
            with self.subTest(environment=environment), self.assertRaises(RuntimeError):
                runner.reject_ambient_injection(environment)

    def test_attached_environment_is_private_and_pinned(self) -> None:
        loader, agent = runner.attached_environment(Path("/runtime"), "private", Path("/tmp/agent.log"))
        for environment in (loader, agent):
            self.assertEqual(environment["BPFTIME_GLOBAL_SHM_NAME"], "private")
            self.assertEqual(environment["BPFTIME_MAP_GPU_THREAD_COUNT"], str(runner.MAX_THREADS))
            self.assertEqual(environment["BPFTIME_VERIFIER_LEVEL"], "WARNING")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertIn("syscall-server", loader["LD_PRELOAD"])
        self.assertIn("agent", agent["LD_PRELOAD"])


class SummaryTests(unittest.TestCase):
    def test_summary_requires_complete_pairs(self) -> None:
        records = []
        for block in range(2):
            for arm, elapsed in (("baseline", 10.0), ("noop", 11.0), ("counter", 12.0)):
                records.append({
                    "valid": True, "block": block, "arm": arm,
                    "measurements": [{"cell": 0, "elapsed_ms": elapsed}],
                })
        rows = runner.summarize(records, 2)
        noop = next(row for row in rows if row["arm"] == "noop")
        self.assertAlmostEqual(noop["median_delta_ms"], 1.0)
        self.assertAlmostEqual(noop["median_overhead_pct"], 10.0)
        with self.assertRaisesRegex(RuntimeError, "incomplete pair"):
            runner.summarize(records[:-1], 2)

    def test_resume_parameters_use_lists_for_json_stability(self) -> None:
        args = mock.Mock(
            phase="preflight", bpftime_root=Path("/runtime"),
            bpftime_build=Path("/runtime/build"),
        )
        params = runner.defining_parameters(args)
        self.assertIsInstance(params["cell_ids"], list)
        self.assertEqual(json.loads(json.dumps(params)), params)


class CampaignStateTests(unittest.TestCase):
    def test_arm_is_not_checkpointed_valid_before_post_safety(self) -> None:
        snapshot = {"gpu": {"driver": runner.EXPECTED_DRIVER, "name": runner.EXPECTED_GPU}}
        fake_record = {
            "valid": True, "arm": "baseline",
            "measurements": [measurement_event(runner.CELLS[0], 1, 2, 1)],
        }
        args = mock.Mock(
            phase="preflight", output=None,
            bpftime_root=Path("/runtime"), bpftime_build=Path("/runtime/build"),
            resume=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary) / "run"
            with (
                mock.patch.object(runner, "build_harness", return_value=args.output / "build-01.log"),
                mock.patch.object(runner, "runtime_configuration", return_value={"runtime": "ok"}),
                mock.patch.object(runner, "validate_compiled_hook_site", return_value={"ptx": "ok"}),
                mock.patch.object(runner, "audit_runtime_source", return_value={"source": "ok"}),
                mock.patch.object(runner, "source_manifest", return_value=[{"source": "ok"}]),
                mock.patch.object(runner, "ReadOnlyLeases", return_value=mock.MagicMock()),
                mock.patch.object(runner, "run_baseline", return_value=fake_record),
                mock.patch.object(runner.safety, "safety_snapshot", return_value=snapshot),
                mock.patch.object(runner.safety, "validate_pre_server_safety"),
                mock.patch.object(
                    runner.safety, "start_gpu_telemetry",
                    return_value=(mock.Mock(), io.StringIO(), args.output / "telemetry.csv"),
                ),
                mock.patch.object(
                    runner.safety, "wait_for_post_server_safety",
                    side_effect=[RuntimeError("did not settle"), snapshot],
                ),
                mock.patch.object(runner.safety, "validate_gpu_telemetry", return_value={"ok": True}),
                mock.patch.object(runner, "stop_owned"),
            ):
                with self.assertRaisesRegex(RuntimeError, "did not settle"):
                    runner.run_campaign(args)
            state = json.loads((args.output / "result.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["records"], [])
        self.assertIn("did not settle", state["campaign_error"])


if __name__ == "__main__":
    unittest.main()
