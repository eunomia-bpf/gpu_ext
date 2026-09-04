#!/usr/bin/env python3
"""CPU-only unit tests for the SPIR-V/OpenCL evidence gates."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


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

    def test_accepts_entry_patch_that_shrinks_valid_module(self):
        shrinking = POSITIVE.replace(
            "Generated SPIR-V binary: 400 bytes", "Generated SPIR-V binary: 540 bytes"
        ).replace(
            "Patched SPIR-V binary: 420 bytes", "Patched SPIR-V binary: 528 bytes"
        )
        parsed = runner.parse_positive_output(shrinking)
        self.assertEqual((parsed["generated_bytes"], parsed["patched_bytes"]), (540, 528))

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
            POSITIVE.replace("Patched SPIR-V binary: 420", "Patched SPIR-V binary: 21"),
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
        child = subprocess.Popen(
            [sys.executable, "-c", code], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
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


class CapabilityGateTests(unittest.TestCase):
    def capability(self, *, legacy="", versioned=None, extension=False, driver=None):
        capability = {
            "device_name": runner.EXPECTED_DEVICE,
            "driver_version": driver or runner.EXPECTED_DRIVER,
            "cl_device_il_version": legacy,
            "cl_device_ils_with_version": versioned or [],
            "has_cl_khr_il_program": extension,
        }
        capability["supports_spirv_il"] = runner.spirv_il_advertised(capability)
        return capability

    def test_capability_query_runs_in_short_lived_child(self):
        expected = self.capability()
        completed = subprocess.CompletedProcess(
            ["helper"], 0, json.dumps(expected), ""
        )
        with mock.patch.object(runner.subprocess, "run", return_value=completed) as run:
            self.assertEqual(runner.query_opencl_capability_isolated(), expected)
        argv = run.call_args.args[0]
        self.assertEqual(argv[-1], str(runner.CAPABILITY_HELPER))
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")

    def test_decodes_opencl_numeric_version(self):
        encoded = (3 << 22) | (1 << 12) | 7
        self.assertEqual(
            runner.decode_opencl_version(encoded),
            {"major": 3, "minor": 1, "patch": 7},
        )

    def test_decodes_standard_name_version_query_layout(self):
        entry = runner.OpenCLNameVersion()
        entry.version = (1 << 22) | (4 << 12)
        entry.name = b"SPIR-V"
        payload = bytes(entry)

        def query(_handle, _parameter, size, value, size_out):
            if size == 0:
                ctypes.cast(size_out, ctypes.POINTER(ctypes.c_size_t)).contents.value = len(
                    payload
                )
            else:
                ctypes.memmove(value, payload, len(payload))
            return runner.CL_SUCCESS

        self.assertEqual(
            runner._query_name_versions(query, ctypes.c_void_p(), 0x1061),
            [{"name": "SPIR-V", "version": {"major": 1, "minor": 4, "patch": 0}}],
        )

    def test_accepts_spirv_from_either_standard_il_query(self):
        for capability in (
            self.capability(legacy="SPIR-V_1.0"),
            self.capability(versioned=[{
                "name": "SPIR-V", "version": {"major": 1, "minor": 4, "patch": 0},
            }]),
        ):
            with self.subTest(capability=capability):
                runner.require_spirv_il_capability(capability)

    def test_rejects_empty_or_non_spirv_il_without_fallback(self):
        for capability in (
            self.capability(),
            self.capability(legacy="OTHER_1.0", extension=True),
            self.capability(versioned=[{
                "name": "OTHER", "version": {"major": 1, "minor": 0, "patch": 0},
            }]),
        ):
            with self.subTest(capability=capability), self.assertRaises(
                runner.UnsupportedSPIRVIL
            ):
                runner.require_spirv_il_capability(capability)

    def test_rejects_target_or_derived_flag_mismatch(self):
        wrong_device = self.capability(legacy="SPIR-V_1.0")
        wrong_device["device_name"] = "other"
        wrong_driver = self.capability(legacy="SPIR-V_1.0", driver="other")
        wrong_flag = self.capability()
        wrong_flag["supports_spirv_il"] = True
        for capability in (wrong_device, wrong_driver, wrong_flag):
            with self.subTest(capability=capability), self.assertRaises(RuntimeError):
                runner.require_spirv_il_capability(capability)


class UnsupportedAnalyzerTests(unittest.TestCase):
    def fixture(self, directory: Path) -> None:
        binary = directory / "demo"
        source = directory / "demo.cpp"
        binary.write_bytes(b"executable")
        binary.chmod(0o755)
        source.write_text("source\n", encoding="utf-8")
        capability = {
            "loader": runner.OPENCL_LOADER,
            "platform_count": 1,
            "selected_platform_index": 0,
            "platform_name": "NVIDIA CUDA",
            "platform_vendor": "NVIDIA Corporation",
            "platform_version": "OpenCL 3.0 CUDA 12.9.76",
            "device_name": runner.EXPECTED_DEVICE,
            "device_vendor": "NVIDIA Corporation",
            "driver_version": runner.EXPECTED_DRIVER,
            "device_version": "OpenCL 3.0 CUDA",
            "device_numeric_version": {"major": 3, "minor": 0, "patch": 0},
            "opencl_c_version": "OpenCL C 1.2",
            "cl_device_il_version": "",
            "cl_device_ils_with_version": [],
            "has_cl_khr_il_program": False,
            "extensions": ["cl_khr_fp64"],
            "supports_intermediate_language_programs": False,
            "supports_spirv_il": False,
        }
        snapshot = {
            "dmesg_abnormal": [], "journal_abnormal": [], "xids": [],
            "power_limit_service": "active", "power_limit_w": 400.0,
            "uvm_refcount": 0, "struct_ops": {"maps": [], "links": []},
            "timestamp_ns": 1,
            "gpu": {
                "compute_apps": [], "driver": runner.EXPECTED_DRIVER,
                "index": 0, "memory_used_mib": 15, "name": runner.EXPECTED_DEVICE,
                "utilization_gpu_percent": 0,
            },
        }
        after = dict(snapshot)
        after["timestamp_ns"] = 2
        result = {
            "schema": 2,
            "status": "unsupported",
            "scope": analyzer.EXPECTED_SCOPE,
            "exclusions": analyzer.EXPECTED_EXCLUSIONS,
            "binary": runner.file_record(binary),
            "source": runner.file_record(source),
            "build": analyzer.EXPECTED_BUILD,
            "source_build_identity": {
                "tracked_inputs_unmodified": True,
                "binary_newer_than_inputs": True,
                "opencl_loader_linked": True,
            },
            "boot_id": "boot",
            "boot_id_after": "boot",
            "capability_checked_before_demo_process": True,
            "demo_process_started": False,
            "device_capability": capability,
            "error": "UnsupportedSPIRVIL: selected device does not advertise SPIR-V",
            "owned_process_survivors": [],
            "safety_before": snapshot,
            "safety_after": after,
        }
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (directory / "device-capability.json").write_text(
            json.dumps(capability), encoding="utf-8"
        )

    def test_unsupported_capability_is_valid_negative_not_device_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            result = analyzer.analyze(directory)
        self.assertTrue(result["complete"])
        self.assertEqual(result["run_status"], "valid")
        self.assertEqual(result["tested_hypothesis"], "contradicted")
        self.assertFalse(result["observations"]["host_spirv_generated"])
        self.assertFalse(result["observations"]["device_kernel_executed"])

    def test_unsupported_capability_mutations_fail_closed(self):
        mutations = ("support", "process", "artifact", "capability_file")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.fixture(directory)
                result_path = directory / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if mutation == "support":
                    result["device_capability"]["cl_device_il_version"] = "SPIR-V_1.0"
                    result["device_capability"]["supports_intermediate_language_programs"] = True
                    result["device_capability"]["supports_spirv_il"] = True
                    (directory / "device-capability.json").write_text(
                        json.dumps(result["device_capability"]), encoding="utf-8"
                    )
                elif mutation == "process":
                    result["demo_process_started"] = True
                elif mutation == "artifact":
                    (directory / "execution.json").write_text("{}", encoding="utf-8")
                else:
                    (directory / "device-capability.json").unlink()
                result_path.write_text(json.dumps(result), encoding="utf-8")
                analyzed = analyzer.analyze(directory)
                self.assertFalse(analyzed["complete"])
                self.assertEqual(analyzed["run_status"], "invalid")


class RetainedFailureTests(unittest.TestCase):
    def test_attempt01_is_valid_capability_failure_not_device_execution(self):
        directory = HERE / "raw/spirv-opencl-575-01"
        if not directory.is_dir():
            self.skipTest("retained attempt 01 is absent")
        result = analyzer.analyze(directory)
        self.assertTrue(result["complete"], result["errors"])
        self.assertEqual(result["tested_hypothesis"], "contradicted")
        self.assertTrue(result["observations"]["host_spirv_validated"])
        self.assertFalse(result["observations"]["opencl_program_created"])
        self.assertFalse(result["observations"]["device_kernel_executed"])
        self.assertEqual(result["observations"]["opencl_error_code"], -59)

    def test_attempt01_failure_mutations_are_rejected(self):
        source = HERE / "raw/spirv-opencl-575-01"
        if not source.is_dir():
            self.skipTest("retained attempt 01 is absent")
        for mutation in ("error", "success_marker", "returncode", "module"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "attempt"
                shutil.copytree(source, directory)
                if mutation == "error":
                    log = (directory / "positive.log").read_text(encoding="utf-8")
                    (directory / "positive.log").write_text(
                        log.replace("error code: -59", "error code: -30"),
                        encoding="utf-8",
                    )
                elif mutation == "success_marker":
                    log = (directory / "positive.log").read_text(encoding="utf-8")
                    (directory / "positive.log").write_text(
                        log.replace(
                            "# exit: 1",
                            "Executing eBPF program on GPU via OpenCL...\n# exit: 1",
                        ),
                        encoding="utf-8",
                    )
                elif mutation == "returncode":
                    path = directory / "execution.json"
                    execution = json.loads(path.read_text(encoding="utf-8"))
                    execution["returncode"] = 0
                    path.write_text(json.dumps(execution), encoding="utf-8")
                else:
                    path = directory / "positive/bpf_program.spv"
                    path.write_bytes(path.read_bytes()[:-4])
                analyzed = analyzer.analyze(directory)
                self.assertFalse(analyzed["complete"])
                self.assertEqual(analyzed["run_status"], "invalid")


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
