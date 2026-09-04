#!/usr/bin/env python3
"""CPU-only unit tests for the SPIR-V/OpenCL evidence gates."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_spirv_opencl_preflight as analyzer
import run_spirv_opencl_preflight as runner


POSITIVE = """SPIR-V target found successfully
Generating SPIR-V from eBPF program...
Generated SPIR-V binary: 400 bytes
Patching SPIR-V to add kernel entry point...
Patched SPIR-V binary: 420 bytes
SPIR-V binary saved to bpf_program.spv
Found GPU on platform: NVIDIA CUDA
Using OpenCL device: NVIDIA GeForce RTX 5090
Loading SPIR-V binary into OpenCL...
Building OpenCL program...
Creating kernel 'bpf_main'...
Executing eBPF program on GPU via OpenCL...
Input value (arr[0]): 100
Expected output (arr[1]): 142
Actual output (arr[1]): 142
✓ Test PASSED!
"""


class OutputGateTests(unittest.TestCase):
    def test_accepts_exact_positive_output(self):
        self.assertEqual(runner.parse_positive_output(POSITIVE)["actual"], 142)

    def test_rejects_each_missing_marker_and_bad_values(self):
        for marker in (
            "SPIR-V target found successfully", "Found GPU on platform: NVIDIA CUDA",
            "Actual output (arr[1]): 142", "Test PASSED!",
        ):
            with self.subTest(marker=marker), self.assertRaises(RuntimeError):
                runner.parse_positive_output(POSITIVE.replace(marker, ""))
        for text in (
            POSITIVE.replace("Generated SPIR-V binary: 400", "Generated SPIR-V binary: 20"),
            POSITIVE.replace("Patched SPIR-V binary: 420", "Patched SPIR-V binary: 400"),
            POSITIVE + "Generated SPIR-V binary: 400 bytes\n",
        ):
            with self.assertRaises(RuntimeError):
                runner.parse_positive_output(text)

    def test_structure_requires_exact_entry_and_memory_model(self):
        valid = ('OpEntryPoint Kernel %9 "bpf_main"\n'
                 'OpMemoryModel Physical64 OpenCL\n%9 = OpFunction\n')
        self.assertEqual(runner.require_structure(valid)["entry_point"], 1)
        for invalid in ("", valid + 'OpEntryPoint Kernel %10 "bpf_main"\n',
                        valid.replace("Physical64 OpenCL", "Logical GLSL450"),
                        valid.replace("OpFunction", "OpLabel")):
            with self.assertRaises(RuntimeError):
                runner.require_structure(invalid)

    def test_header_requires_size_alignment_and_magic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "module.spv"
            path.write_bytes(runner.SPIRV_MAGIC.to_bytes(4, "little") + bytes(20))
            self.assertEqual(runner.require_spirv_header(path, 24)["size_bytes"], 24)
            for data, size in ((bytes(24), 24), (path.read_bytes(), 28),
                               (path.read_bytes() + b"x", 25)):
                path.write_bytes(data)
                with self.assertRaises(RuntimeError):
                    runner.require_spirv_header(path, size)


class IsolationTests(unittest.TestCase):
    def test_rejects_ambient_injection(self):
        runner.reject_ambient_injection({"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0"})
        for environment in (
            {"LD_PRELOAD": "bad"}, {"BPFTIME_GLOBAL_SHM_NAME": "bad"},
            {"OCL_ICD_VENDORS": "bad"}, {"CUDA_VISIBLE_DEVICES": "1"},
        ):
            with self.subTest(environment=environment), self.assertRaises(RuntimeError):
                runner.reject_ambient_injection(environment)

    def test_read_only_lease_does_not_create_missing_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with self.assertRaises(FileNotFoundError):
                runner.ReadOnlyLeases((missing,))
            self.assertFalse(missing.exists())

    def test_read_only_lease_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lease"
            path.touch()
            before = path.stat()
            lease = runner.ReadOnlyLeases((path,))
            lease.close()
            after = path.stat()
            self.assertEqual((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))

    def test_owned_orphan_cleanup(self):
        code = "import os,time\npid=os.fork()\nif pid: os._exit(0)\ntime.sleep(15)\n"
        child = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
        try:
            child.wait(timeout=2)
            self.assertTrue(runner.group_members(child.pid))
            runner.stop_owned(child)
            self.assertEqual(runner.group_members(child.pid), [])
        finally:
            runner.stop_owned(child)


class AnalyzerPrimitiveTests(unittest.TestCase):
    def test_final_exit_is_unique_and_terminal(self):
        self.assertEqual(analyzer.final_exit("body\n# exit: 0\n"), 0)
        for invalid in ("", "# exit: 0\ntrailing\n", "# exit: 0\n# exit: 0\n"):
            with self.assertRaises(RuntimeError):
                analyzer.final_exit(invalid)

    def test_missing_campaign_is_invalid_not_an_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = analyzer.analyze(Path(temporary))
        self.assertEqual(result["run_status"], "invalid")
        self.assertFalse(result["complete"])


class SourceBuildTests(unittest.TestCase):
    def test_current_independent_build_has_source_native_identity(self):
        workspace = HERE.parents[3]
        source = workspace / "bpftime-table1-575/vm/llvm-jit/example/spirv/spirv_opencl_test.cpp"
        build = workspace / "bpftime-spirv-build-20260904"
        binary = build / "example/spirv/spirv_opencl_test"
        if not binary.is_file():
            self.skipTest("independent CPU-only SPIR-V build is absent")
        identity = runner.source_build_identity(source, binary, build)
        self.assertTrue(identity["tracked_inputs_unmodified"])
        self.assertTrue(identity["binary_newer_than_inputs"])
        self.assertTrue(identity["opencl_loader_linked"])


if __name__ == "__main__":
    unittest.main()
