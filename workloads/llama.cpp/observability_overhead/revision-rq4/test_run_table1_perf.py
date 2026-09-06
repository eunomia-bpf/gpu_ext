#!/usr/bin/env python3
"""Unittest for run_table1_perf.py: rotation schedule, dry-run, and the
idempotence of the declared kernelretsnoop capacity patch."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


table1 = load_module("run_table1_perf", "run_table1_perf.py")


class Table1RotationAndDryRunTests(unittest.TestCase):
    def test_seven_arms_rotate_every_block(self):
        arms = [
            "baseline",
            "gpubpf_kernelretsnoop",
            "nvbit_kernelretsnoop",
            "gpubpf_threadhist",
            "nvbit_threadhist",
            "gpubpf_launchlate",
            "nvbit_launchlate",
        ]
        self.assertEqual(list(table1.ARMS), arms)
        schedule = table1.build_schedule(8)
        self.assertEqual(len(schedule), 8)
        for block in range(1, 9):
            offset = (block - 1) % len(arms)
            self.assertEqual(schedule[str(block)], arms[offset:] + arms[:offset])
        for order in schedule.values():
            self.assertEqual(sorted(order), sorted(arms))
        self.assertEqual(schedule["1"], arms)
        self.assertEqual(schedule["8"], schedule["1"])
        self.assertEqual(table1.build_schedule(1)["1"], arms)
        self.assertEqual(len(table1.build_schedule(10)["10"]), 7)

    def test_dry_run_prints_schedule_and_does_no_build_or_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "table1-dry"
            forbidden = [
                mock.patch.object(
                    table1, "run_campaign",
                    side_effect=AssertionError("dry-run must not run the campaign"),
                ),
                mock.patch.object(
                    table1, "build_tools",
                    side_effect=AssertionError("dry-run must not build tools"),
                ),
                mock.patch.object(
                    table1, "run_arm_cell",
                    side_effect=AssertionError("dry-run must not run cells"),
                ),
                mock.patch.object(
                    table1, "write_records",
                    side_effect=AssertionError("dry-run must not write records"),
                ),
                mock.patch.object(
                    table1.core, "prepare_tool_source",
                    side_effect=AssertionError("dry-run must not prepare tool sources"),
                ),
                mock.patch.object(
                    table1.core, "build_tool",
                    side_effect=AssertionError("dry-run must not build gpubpf tools"),
                ),
                mock.patch.object(
                    table1.runner, "build_nvbit",
                    side_effect=AssertionError("dry-run must not build NVBit"),
                ),
                mock.patch.object(
                    table1.runner, "private_probe",
                    side_effect=AssertionError("dry-run must not start probes"),
                ),
                mock.patch.object(
                    table1.runner, "run_bench",
                    side_effect=AssertionError("dry-run must not run benches"),
                ),
                mock.patch.object(
                    table1.core, "nvidia_smi_snapshot",
                    side_effect=AssertionError("dry-run must not query the GPU"),
                ),
                mock.patch.object(
                    table1.subprocess, "run",
                    side_effect=AssertionError("dry-run must not launch processes"),
                ),
                mock.patch.object(
                    table1.subprocess, "Popen",
                    side_effect=AssertionError("dry-run must not launch processes"),
                ),
                mock.patch.object(
                    table1.shutil, "copytree",
                    side_effect=AssertionError("dry-run must not copy tool sources"),
                ),
            ]
            with contextlib.ExitStack() as stack:
                for patch in forbidden:
                    stack.enter_context(patch)
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    code = table1.main(
                        ["--dry-run", "--blocks", "3", "--output-dir", str(out_dir)]
                    )
            plan = json.loads(buffer.getvalue())
        self.assertEqual(code, 0)
        self.assertIs(plan["dry_run"], True)
        self.assertEqual(plan["blocks"], 3)
        self.assertEqual(plan["pp"], 512)
        self.assertEqual(plan["tg"], 0)
        self.assertEqual(plan["arms"], list(table1.ARMS))
        self.assertEqual(plan["cell_count"], 21)
        self.assertEqual(plan["schedule"], table1.build_schedule(3))
        self.assertFalse(out_dir.exists())


class RunArmCellProbeTeardownErrorTests(unittest.TestCase):
    def test_gpubpf_completed_bench_survives_runtime_error_on_probe_exit(self):
        probe_env = {
            "LD_PRELOAD": "/tools/kernelretsnoop/libbpftime.so",
            "BPFTIME_GLOBAL_SHM_NAME": "rq4_table1_test_1",
        }
        bench_calls = []

        @contextlib.contextmanager
        def fake_private_probe(tool_name, args_ns, tool_dir, run_dir):
            yield probe_env
            raise RuntimeError("private probe exited unsuccessfully: 1")

        def fake_run_bench(label, run_id, args_ns, output_dir_arg, env_extra=None):
            bench_calls.append(env_extra)
            return {
                "run": run_id,
                "log": None,
                "returncode": 0,
                "valid": True,
                "metrics": {"pp_tokens": 512, "pp_tok_s": 100.0},
            }

        args = SimpleNamespace(
            uvm=False,
            llama_bench=Path("/bench/llama-bench"),
            model=Path("/models/model.gguf"),
            pp=512,
            tg=0,
            n_gpu_layers=99,
            no_warmup=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with (
                mock.patch.object(
                    table1.runner, "private_probe", side_effect=fake_private_probe
                ),
                mock.patch.object(table1.runner, "run_bench", side_effect=fake_run_bench),
            ):
                record = table1.run_arm_cell(
                    "gpubpf_kernelretsnoop",
                    4,
                    args,
                    output_dir,
                    {"kernelretsnoop": output_dir / "tool"},
                    output_dir / "nvbit_build",
                )
        self.assertEqual(bench_calls, [probe_env])
        self.assertEqual(record["returncode"], 0)
        self.assertFalse(record["timed_out"])
        self.assertEqual(record["throughput_tok_s"], 100.0)
        self.assertEqual(record["metrics"], {"pp_tokens": 512, "pp_tok_s": 100.0})
        self.assertEqual(
            record["probe_teardown_error"],
            "RuntimeError: private probe exited unsuccessfully: 1",
        )
        self.assertNotIn("error", record)
        self.assertIn("LD_PRELOAD=/tools/kernelretsnoop/libbpftime.so", record["command"])
        self.assertIn(str(args.llama_bench), record["command"])


class CapacityPatchIdempotenceTests(unittest.TestCase):
    APPLIED_BPF = (
        "struct data {\n"
        "\tu64 coordinate_x, coordinate_y, coordinate_z;\n"
        "\tu64 timestamp;\n"
        "};\n"
        "\n"
        "int cuda__retprobe()\n"
        "{\n"
        "\tstruct data data = {};\n"
        "\tu64 block_x = 0;\n"
        "\tu64 linear_thread = 0;\n"
        "\tu64 warps_per_block = 0;\n"
        "\tdata.coordinate_x = block_x * warps_per_block + (linear_thread >> 5);\n"
        "\tdata.coordinate_y = 0;\n"
        "\tdata.coordinate_z = 0;\n"
        "\treturn 0;\n"
        "}\n"
    )
    APPLIED_USER = (
        "#include <stdio.h>\n"
        "\n"
        "struct data {\n"
        "\tuint64_t coordinate_x, coordinate_y, coordinate_z;\n"
        "\tuint64_t timestamp;\n"
        "};\n"
        "\n"
        "static uint64_t requested_ring_entries(void)\n"
        "{\n"
        "\tconst char *value = getenv(\"BPFTIME_KERNELRETSNOOP_RING_ENTRIES\");\n"
        "\treturn value ? 1 : 0;\n"
        "}\n"
        "\n"
        "int main(int argc, char **argv)\n"
        "{\n"
        "\tint err = bpf_map__set_max_entries(skel->maps.rb, requested_entries);\n"
        "\treturn err;\n"
        "}\n"
    )
    OLD_BPF = (
        "struct data {\n"
        "\tu64 block_x, block_y, block_z;\n"
        "\tu64 thread_x, thread_y, thread_z;\n"
        "\tu64 block_dim_x, block_dim_y, block_dim_z;\n"
        "\tu64 timestamp;\n"
        "};\n"
        "\n"
        "int cuda__retprobe()\n"
        "{\n"
        "\tstruct data data = {};\n"
        "\n"
        "\tbpf_get_block_idx(&data.block_x, &data.block_y, &data.block_z);\n"
        "\tbpf_get_thread_idx(&data.thread_x, &data.thread_y, &data.thread_z);\n"
        "\tbpf_get_block_dim(&data.block_dim_x, &data.block_dim_y,\n"
        "\t\t\t  &data.block_dim_z);\n"
        "\tdata.timestamp = bpf_get_globaltimer();\n"
        "\treturn bpf_perf_event_output(NULL, &rb, 0, &data,\n"
        "\t\t\t\t     sizeof(struct data));\n"
        "}\n"
    )
    OLD_USER = (
        "#include <stdio.h>\n"
        "\n"
        "struct data {\n"
        "\tuint64_t block_x, block_y, block_z;\n"
        "\tuint64_t thread_x, thread_y, thread_z;\n"
        "\tuint64_t block_dim_x, block_dim_y, block_dim_z;\n"
        "\tuint64_t timestamp;\n"
        "};\n"
        "\n"
        "int main(int argc, char **argv)\n"
        "{\n"
        "\treturn 0;\n"
        "}\n"
    )

    def _bpftime_root(self, root: Path) -> Path:
        (root / "runtime" / "include").mkdir(parents=True)
        return root

    def _write_source(self, directory: Path, bpf: str, user: str, makefile: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "kernelretsnoop.bpf.c").write_text(bpf, encoding="utf-8")
        (directory / "kernelretsnoop.c").write_text(user, encoding="utf-8")
        (directory / "Makefile").write_text(makefile, encoding="utf-8")

    def test_pure_runner_does_not_reapply_capacity_patch_when_capacity_form_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bpftime_root = self._bpftime_root(root / "bpftime")
            output_dir = root / "out"
            output_dir.mkdir()
            source_dir = root / "kernelretsnoop"
            self._write_source(
                source_dir,
                self.APPLIED_BPF,
                self.APPLIED_USER,
                "INCLUDES := -I../../../runtime/include\n",
            )
            built = []
            args = SimpleNamespace(bpftime_root=bpftime_root, target_symbol="target")
            with (
                mock.patch.object(
                    table1.core, "prepare_tool_source", return_value=source_dir
                ),
                mock.patch.object(
                    table1.core,
                    "build_tool",
                    side_effect=lambda spec, directory: built.append(spec.name),
                ),
                mock.patch.object(
                    table1.shutil,
                    "copytree",
                    side_effect=lambda src, dst, ignore=None: dst.mkdir(parents=True),
                ),
                mock.patch.object(
                    table1.runner,
                    "build_nvbit",
                    return_value=output_dir / "nvbit_tool_build" / "observability.so",
                ),
                mock.patch.object(
                    table1.runner.subprocess,
                    "run",
                    side_effect=AssertionError(
                        "pure runner must not reapply the declared capacity patch"
                    ),
                ),
            ):
                tool_dirs, nvbit_tool = table1.build_tools(args, output_dir)
            self.assertEqual(sorted(tool_dirs), sorted(table1.TASKS))
            for tool in table1.TASKS:
                self.assertIs(tool_dirs[tool], source_dir)
            self.assertEqual(built, list(table1.TASKS))
            self.assertEqual(
                nvbit_tool, output_dir / "nvbit_tool_build" / "observability.so"
            )
            self.assertEqual(
                (source_dir / "kernelretsnoop.bpf.c").read_text(encoding="utf-8"),
                self.APPLIED_BPF,
            )
            self.assertEqual(
                (source_dir / "kernelretsnoop.c").read_text(encoding="utf-8"),
                self.APPLIED_USER,
            )
            self.assertEqual([], list(source_dir.glob("*.rej")))
            self.assertEqual([], list(source_dir.glob("*.orig")))
            makefile = (source_dir / "Makefile").read_text(encoding="utf-8")
            self.assertNotIn("../../../runtime/include", makefile)
            self.assertIn(
                str((bpftime_root / "runtime/include").resolve()), makefile
            )

    def test_capacity_patch_still_applies_to_genuinely_old_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bpftime_root = self._bpftime_root(root / "bpftime")
            source_dir = root / "kernelretsnoop"
            self._write_source(source_dir, self.OLD_BPF, self.OLD_USER, "all:\n\techo old\n")
            calls = []

            def fake_run(command, **kwargs):
                calls.append((list(command), kwargs.get("cwd")))
                return SimpleNamespace(
                    returncode=0,
                    stdout="patching file kernelretsnoop.bpf.c\n",
                )

            with (
                mock.patch.object(
                    table1.core, "prepare_tool_source", return_value=source_dir
                ),
                mock.patch.object(table1.runner.subprocess, "run", side_effect=fake_run),
            ):
                tool_dir = table1.runner.prepare_tool_source(
                    table1.core.TOOLS["kernelretsnoop"],
                    bpftime_root=bpftime_root,
                    build_root=root / "build",
                    target_symbol="target",
                )
        self.assertIs(tool_dir, source_dir)
        self.assertEqual(len(calls), 1)
        command, cwd = calls[0]
        self.assertEqual(
            command,
            [
                "patch", "--batch", "--forward", "--fuzz=0", "-p1",
                "-i", str(table1.runner.KERNELRETSNOOP_CAPACITY_PATCH),
            ],
        )
        self.assertEqual(cwd, source_dir)

    def test_capacity_patch_failure_on_genuinely_old_source_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bpftime_root = self._bpftime_root(root / "bpftime")
            source_dir = root / "kernelretsnoop"
            self._write_source(source_dir, self.OLD_BPF, self.OLD_USER, "all:\n\techo old\n")
            with (
                mock.patch.object(
                    table1.core, "prepare_tool_source", return_value=source_dir
                ),
                mock.patch.object(
                    table1.runner.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=1,
                        stdout="Reversed (or previously applied) patch detected!  "
                               "Skipping patch.\n",
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "failed to apply declared kernelretsnoop capacity patch",
                ):
                    table1.runner.prepare_tool_source(
                        table1.core.TOOLS["kernelretsnoop"],
                        bpftime_root=bpftime_root,
                        build_root=root / "build",
                        target_symbol="target",
                    )


if __name__ == "__main__":
    unittest.main()
