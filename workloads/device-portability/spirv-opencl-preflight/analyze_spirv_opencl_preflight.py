#!/usr/bin/env python3
"""Independently replay one SPIR-V/OpenCL preflight evidence directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


SPIRV_MAGIC = 0x07230203
EXPECTED_DEVICE = "NVIDIA GeForce RTX 5090"
EXPECTED_DRIVER = "575.57.08"
EXPECTED_SCOPE = "standalone eBPF-to-SPIR-V OpenCL demo only"
EXPECTED_EXCLUSIONS = [
    "gpubpf device-hook attach backend", "cross-layer maps", "SIMT verifier",
    "application-kernel policy execution", "AMD or Intel execution", "performance",
]
EXPECTED_BUILD = {
    "LLVMBPF_ENABLE_SPIRV": "ON",
    "LLVM_DIR": "/usr/lib/llvm-20/lib/cmake/llvm",
    "CMAKE_HOME_DIRECTORY": "/home/yunwei37/workspace/gpu/bpftime-table1-575/vm/llvm-jit",
}
EXPECTED_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    "CUDA_VISIBLE_DEVICES": "0",
    "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64",
}
HOST_PREFIX_MARKERS = (
    "SPIR-V target found successfully",
    "Generating SPIR-V from eBPF program...",
    "Patching SPIR-V to add kernel entry point...",
    "SPIR-V binary saved to bpf_program.spv",
    "Found GPU on platform: NVIDIA CUDA",
    f"Using OpenCL device: {EXPECTED_DEVICE}",
    "Loading SPIR-V binary into OpenCL...",
)
DEVICE_SUCCESS_MARKERS = (
    "Building OpenCL program...",
    "Creating kernel 'bpf_main'...",
    "Executing eBPF program on GPU via OpenCL...",
    "Input value (arr[0]): 100",
    "Expected output (arr[1]): 142",
    "Actual output (arr[1]): 142",
    "Test PASSED!",
)


def atomic_write_json(path: Path, value: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def final_exit(log: str) -> int:
    matches = re.findall(r"(?m)^# exit: (-?[0-9]+)$", log)
    if len(matches) != 1 or not log.rstrip().endswith("# exit: " + matches[0]):
        raise RuntimeError("log requires exactly one final exit footer")
    return int(matches[0])


def parse_byte_counts(log: str) -> tuple[int, int]:
    generated = re.findall(r"Generated SPIR-V binary: ([0-9]+) bytes", log)
    patched = re.findall(r"Patched SPIR-V binary: ([0-9]+) bytes", log)
    if len(generated) != 1 or len(patched) != 1:
        raise RuntimeError("SPIR-V byte counts are missing or ambiguous")
    generated_bytes, patched_bytes = int(generated[0]), int(patched[0])
    if (generated_bytes <= 20 or patched_bytes <= 20
            or generated_bytes % 4 or patched_bytes % 4
            or generated_bytes == patched_bytes):
        raise RuntimeError("SPIR-V byte counts are invalid")
    return generated_bytes, patched_bytes


def advertised_spirv(capability: dict[str, Any]) -> bool:
    legacy = capability.get("cl_device_il_version", "")
    legacy_names = [token.partition("_")[0] for token in legacy.split()]
    versioned_names = [entry.get("name") for entry in capability.get(
        "cl_device_ils_with_version", []
    ) if isinstance(entry, dict)]
    return "SPIR-V" in legacy_names or "SPIR-V" in versioned_names


def validate_scope_and_build(recorded: dict[str, Any], errors: list[str]) -> None:
    if recorded.get("schema") not in (1, 2):
        errors.append("unsupported result schema")
    if (recorded.get("scope") != EXPECTED_SCOPE
            or recorded.get("exclusions") != EXPECTED_EXCLUSIONS):
        errors.append("run scope or explicit exclusions differ from the frozen plan")
    if recorded.get("build") != EXPECTED_BUILD:
        errors.append("recorded build configuration differs from the frozen LLVM 20 build")
    identity = recorded.get("source_build_identity", {})
    if (identity.get("tracked_inputs_unmodified") is not True
            or identity.get("binary_newer_than_inputs") is not True
            or identity.get("opencl_loader_linked") is not True):
        errors.append("source/build identity gates were not recorded as passed")
    for field in ("binary", "source"):
        item = recorded.get(field, {})
        path = Path(item.get("path", ""))
        try:
            info = path.stat()
            if info.st_size != item.get("size_bytes") or info.st_mtime_ns != item.get("mtime_ns"):
                errors.append(f"recorded {field} metadata changed")
            if field == "binary" and (not item.get("executable") or not os.access(path, os.X_OK)):
                errors.append("recorded binary is not executable")
        except OSError:
            errors.append(f"recorded {field} is missing")


def validate_safety(recorded: dict[str, Any], errors: list[str]) -> None:
    before, after = recorded.get("safety_before"), recorded.get("safety_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        errors.append("safety snapshots are missing")
        return
    for label, snapshot in (("before", before), ("after", after)):
        gpu = snapshot.get("gpu", {})
        if snapshot.get("power_limit_service") != "active":
            errors.append(f"{label} power service is not active")
        try:
            if abs(float(snapshot.get("power_limit_w", -1)) - 400.0) > 0.01:
                errors.append(f"{label} power limit is not 400 W")
        except (TypeError, ValueError):
            errors.append(f"{label} power limit is malformed")
        if (gpu.get("driver") != EXPECTED_DRIVER or gpu.get("name") != EXPECTED_DEVICE
                or gpu.get("compute_apps")
                or type(gpu.get("memory_used_mib")) is not int
                or gpu.get("memory_used_mib", 257) > 256
                or gpu.get("utilization_gpu_percent") != 0):
            errors.append(f"{label} GPU state is not the fixed idle driver state")
        if snapshot.get("uvm_refcount") != 0:
            errors.append(f"{label} UVM reference count is not zero")
        if snapshot.get("struct_ops") != {"maps": [], "links": []}:
            errors.append(f"{label} struct_ops state is not empty")
        if snapshot.get("dmesg_abnormal") or snapshot.get("journal_abnormal"):
            errors.append(f"{label} kernel safety diagnostics are not clean")
    if (type(before.get("timestamp_ns")) is not int
            or type(after.get("timestamp_ns")) is not int
            or after.get("timestamp_ns", 0) <= before.get("timestamp_ns", 0)):
        errors.append("safety snapshot timestamps are invalid")
    for field in ("dmesg_abnormal", "journal_abnormal", "xids"):
        if before.get(field) != after.get(field):
            errors.append(f"kernel safety history changed in {field}")
    if recorded.get("owned_process_survivors") != []:
        errors.append("owned process cleanup is incomplete")
    if recorded.get("cleanup_errors"):
        errors.append("runner recorded cleanup errors")
    if recorded.get("schema") == 2 and recorded.get("boot_id_after") != recorded.get("boot_id"):
        errors.append("runner did not record boot continuity")


def validate_execution(
    directory: Path, recorded: dict[str, Any], expected_returncode: int,
    errors: list[str],
) -> tuple[dict[str, Any], str]:
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    log = (directory / "positive.log").read_text(encoding="utf-8", errors="replace")
    if execution.get("returncode") != expected_returncode:
        errors.append("execution record return code differs from the expected outcome")
    if final_exit(log) != expected_returncode:
        errors.append("positive log exit differs from the expected outcome")
    if Path(execution.get("cwd", "")) != directory / "positive":
        errors.append("execution cwd is not the retained positive directory")
    if execution.get("environment") != EXPECTED_ENVIRONMENT:
        errors.append("execution environment differs from the frozen environment")
    binary_path = Path(recorded.get("binary", {}).get("path", ""))
    if execution.get("argv") != [str(binary_path)]:
        errors.append("execution argv is not the exact recorded source-native binary")
    if (type(execution.get("pid")) is not int or execution["pid"] <= 0
            or type(execution.get("started_ns")) is not int
            or type(execution.get("ended_ns")) is not int
            or execution["ended_ns"] <= execution["started_ns"]):
        errors.append("execution identity or timestamp interval is invalid")
    return execution, log


def validate_capability(
    directory: Path, recorded: dict[str, Any], expected_support: bool,
    errors: list[str],
) -> dict[str, Any]:
    capability = recorded.get("device_capability")
    if not isinstance(capability, dict):
        errors.append("device capability record is missing")
        return {}
    retained = json.loads((directory / "device-capability.json").read_text(encoding="utf-8"))
    if retained != capability:
        errors.append("standalone capability record differs from result.json")
    if (capability.get("device_name") != EXPECTED_DEVICE
            or capability.get("driver_version") != EXPECTED_DRIVER
            or capability.get("platform_name") != "NVIDIA CUDA"):
        errors.append("capability record names a different OpenCL target")
    if capability.get("device_numeric_version") != {"major": 3, "minor": 0, "patch": 0}:
        errors.append("capability record does not describe OpenCL 3.0")
    extensions = capability.get("extensions")
    if not isinstance(extensions, list):
        errors.append("OpenCL extension inventory is malformed")
        extensions = []
    if capability.get("has_cl_khr_il_program") is not ("cl_khr_il_program" in extensions):
        errors.append("cl_khr_il_program summary differs from extension inventory")
    il_version = capability.get("cl_device_il_version")
    ils = capability.get("cl_device_ils_with_version")
    if not isinstance(il_version, str) or not isinstance(ils, list):
        errors.append("OpenCL IL query results are malformed")
        il_version, ils = "", []
    if capability.get("supports_intermediate_language_programs") is not bool(il_version or ils):
        errors.append("general IL support flag differs from the raw IL queries")
    derived = advertised_spirv(capability)
    if capability.get("supports_spirv_il") is not derived:
        errors.append("SPIR-V support flag differs from the raw IL queries")
    if derived is not expected_support:
        errors.append("advertised SPIR-V support differs from the recorded outcome")
    return capability


def validate_module(module: Path, expected_bytes: int, errors: list[str]) -> dict[str, Any]:
    data = module.read_bytes()
    if len(data) != expected_bytes or len(data) < 20 or len(data) % 4:
        errors.append("retained SPIR-V size is invalid")
    if len(data) < 4 or int.from_bytes(data[:4], "little") != SPIRV_MAGIC:
        errors.append("retained SPIR-V magic is invalid")
    validation = subprocess.run(
        ["/usr/bin/spirv-val", str(module)], text=True, capture_output=True, check=False
    )
    if validation.returncode != 0:
        errors.append("independent spirv-val rejected the retained module")
    disassembly = subprocess.run(
        ["/usr/bin/spirv-dis", "--raw-id", str(module)],
        text=True, capture_output=True, check=False,
    )
    structure: dict[str, int] = {}
    if disassembly.returncode != 0:
        errors.append("independent spirv-dis rejected the retained module")
    else:
        structure = {
            "entry_point": len(re.findall(
                r'OpEntryPoint\s+Kernel\s+%\S+\s+"bpf_main"', disassembly.stdout
            )),
            "memory_model": len(re.findall(
                r"OpMemoryModel\s+Physical64\s+OpenCL", disassembly.stdout
            )),
            "function": len(re.findall(r"\bOpFunction\b", disassembly.stdout)),
        }
        if (structure["entry_point"] != 1 or structure["memory_model"] != 1
                or structure["function"] < 1):
            errors.append(f"independent structure gates failed: {structure}")
    return {
        "size_bytes": len(data),
        "validator_returncode": validation.returncode,
        "disassembler_returncode": disassembly.returncode,
        "structure": structure,
    }


def conclusion(errors: list[str], *, supported: bool, observations: dict[str, Any]) -> dict[str, Any]:
    valid = not errors
    return {
        "complete": valid,
        "run_status": "valid" if valid else "invalid",
        "tested_hypothesis": ("supported" if supported else "contradicted") if valid else "inconclusive",
        "research_value": "supporting",
        "paper_impact": "narrow standalone SPIR-V runtime capability boundary",
        "next_paper_decision": (
            "describe only a working standalone NVIDIA OpenCL SPIR-V path"
            if valid and supported else
            "do not claim a runnable SPIR-V device path on RTX 5090 with driver 575.57.08"
        ),
        "observations": observations,
        "errors": errors,
    }


def analyze_unsupported(directory: Path, recorded: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    validate_scope_and_build(recorded, errors)
    validate_safety(recorded, errors)
    validate_capability(directory, recorded, False, errors)
    if recorded.get("schema") != 2 or recorded.get("status") != "unsupported":
        errors.append("unsupported-capability outcome requires schema 2 and status unsupported")
    if recorded.get("capability_checked_before_demo_process") is not True:
        errors.append("capability was not recorded before the demo process gate")
    if recorded.get("demo_process_started") is not False:
        errors.append("demo process started despite unsupported SPIR-V capability")
    if not str(recorded.get("error", "")).startswith("UnsupportedSPIRVIL:"):
        errors.append("runner did not record the explicit unsupported-SPIR-V outcome")
    forbidden = [
        directory / "execution.json", directory / "positive.log", directory / "positive"
    ]
    if any(path.exists() for path in forbidden):
        errors.append("unsupported-capability attempt contains demo execution artifacts")
    return conclusion(errors, supported=False, observations={
        "failure_stage": "pre-execution capability gate",
        "host_spirv_generated": False,
        "host_spirv_validated": False,
        "opencl_program_created": False,
        "device_kernel_executed": False,
        "opencl_error_code": None,
    })


def analyze_legacy_invalid_operation(
    directory: Path, recorded: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    validate_scope_and_build(recorded, errors)
    validate_safety(recorded, errors)
    if recorded.get("schema") != 1 or recorded.get("status") != "invalid":
        errors.append("legacy failure requires schema 1 and runner status invalid")
    if recorded.get("error") != "RuntimeError: source-native OpenCL demo exited 1":
        errors.append("runner error differs from the retained program-creation failure")
    _, log = validate_execution(directory, recorded, 1, errors)
    missing = [marker for marker in HOST_PREFIX_MARKERS if marker not in log]
    if missing:
        errors.append(f"host generation/program-creation prefix missing markers: {missing}")
    if "Failed to create program from SPIR-V (error code: -59)" not in log:
        errors.append("retained log does not contain clCreateProgramWithIL error -59")
    unexpected = [marker for marker in DEVICE_SUCCESS_MARKERS if marker in log]
    if unexpected:
        errors.append(f"failure log contains post-program-creation markers: {unexpected}")
    generated_bytes, patched_bytes = parse_byte_counts(log)
    module_evidence = validate_module(directory / "positive/bpf_program.spv", patched_bytes, errors)
    return conclusion(errors, supported=False, observations={
        "failure_stage": "clCreateProgramWithIL",
        "generated_bytes": generated_bytes,
        "patched_bytes": patched_bytes,
        "host_spirv_generated": True,
        "host_spirv_validated": module_evidence["validator_returncode"] == 0,
        "opencl_program_created": False,
        "device_kernel_executed": False,
        "opencl_error_code": -59,
        "capability_recorded_before_demo_process": False,
        "module": module_evidence,
    })


def analyze_complete(directory: Path, recorded: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    validate_scope_and_build(recorded, errors)
    validate_safety(recorded, errors)
    validate_capability(directory, recorded, True, errors)
    if recorded.get("schema") != 2 or recorded.get("status") != "complete":
        errors.append("successful outcome requires schema 2 and status complete")
    if (recorded.get("capability_checked_before_demo_process") is not True
            or recorded.get("demo_process_started") is not True):
        errors.append("successful run lacks ordered capability/process gates")
    _, log = validate_execution(directory, recorded, 0, errors)
    missing = [marker for marker in HOST_PREFIX_MARKERS + DEVICE_SUCCESS_MARKERS if marker not in log]
    if missing:
        errors.append(f"positive output missing markers: {missing}")
    generated_bytes, patched_bytes = parse_byte_counts(log)
    expected_positive = {
        "generated_bytes": generated_bytes, "patched_bytes": patched_bytes,
        "input": 100, "expected": 142, "actual": 142, "device": EXPECTED_DEVICE,
    }
    if recorded.get("positive") != expected_positive:
        errors.append("recorded positive result differs from raw output")
    module = directory / "positive/bpf_program.spv"
    module_evidence = validate_module(module, patched_bytes, errors)
    if recorded.get("spirv_header") != {
        "size_bytes": module_evidence["size_bytes"], "magic_word": SPIRV_MAGIC,
    }:
        errors.append("recorded SPIR-V header differs from retained bytes")
    if recorded.get("structure") != module_evidence["structure"]:
        errors.append("recorded SPIR-V structure differs from independent disassembly")
    for name in ("positive-spirv-val.log", "positive-spirv-dis.log"):
        if final_exit((directory / name).read_text(encoding="utf-8", errors="replace")) != 0:
            errors.append(f"{name} did not retain the expected exit")
    tampered = directory / "tampered-magic.spv"
    source_data, tampered_data = module.read_bytes(), tampered.read_bytes()
    if (len(tampered_data) != len(source_data) or tampered_data[:4] != bytes(4)
            or tampered_data[4:] != source_data[4:]):
        errors.append("tampered module is not an exact magic-word-only mutation")
    negative = subprocess.run(
        ["/usr/bin/spirv-val", str(tampered)], text=True, capture_output=True, check=False
    )
    if negative.returncode == 0:
        errors.append("independent spirv-val accepted the tampered module")
    if final_exit((directory / "tampered-spirv-val.log").read_text(
        encoding="utf-8", errors="replace"
    )) == 0:
        errors.append("retained spirv-val accepted the tampered module")
    if (recorded.get("positive_validation_returncode") != 0
            or type(recorded.get("tampered_validation_returncode")) is not int
            or recorded["tampered_validation_returncode"] == 0
            or recorded.get("tampered_submitted_to_opencl") is not False):
        errors.append("recorded validator controls are inconsistent")
    return conclusion(errors, supported=True, observations={
        "failure_stage": None,
        "generated_bytes": generated_bytes,
        "patched_bytes": patched_bytes,
        "host_spirv_generated": True,
        "host_spirv_validated": module_evidence["validator_returncode"] == 0,
        "opencl_program_created": True,
        "device_kernel_executed": True,
        "opencl_error_code": 0,
        "module": module_evidence,
    })


def _analyze(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    recorded = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if recorded.get("schema") == 2 and recorded.get("status") == "unsupported":
        return analyze_unsupported(directory, recorded)
    if recorded.get("schema") == 2 and recorded.get("status") == "complete":
        return analyze_complete(directory, recorded)
    log_path = directory / "positive.log"
    if (recorded.get("schema") == 1 and recorded.get("status") == "invalid"
            and log_path.is_file()
            and "Failed to create program from SPIR-V (error code: -59)" in
                log_path.read_text(encoding="utf-8", errors="replace")):
        return analyze_legacy_invalid_operation(directory, recorded)
    errors: list[str] = []
    validate_scope_and_build(recorded, errors)
    validate_safety(recorded, errors)
    errors.append("attempt is neither a complete success nor a recognized capability failure")
    return conclusion(errors, supported=False, observations={
        "host_spirv_generated": False,
        "host_spirv_validated": False,
        "opencl_program_created": False,
        "device_kernel_executed": False,
    })


def analyze(directory: Path) -> dict[str, Any]:
    try:
        return _analyze(directory)
    except BaseException as error:
        return conclusion(
            [f"{type(error).__name__}: {error}"], supported=False,
            observations={
                "host_spirv_generated": False,
                "host_spirv_validated": False,
                "opencl_program_created": False,
                "device_kernel_executed": False,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    analysis = analyze(args.directory)
    atomic_write_json(args.directory.resolve() / "analysis.json", analysis)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0 if analysis["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
