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
EXPECTED_MARKERS = (
    "SPIR-V target found successfully",
    "Generating SPIR-V from eBPF program...",
    "Patching SPIR-V to add kernel entry point...",
    "SPIR-V binary saved to bpf_program.spv",
    "Found GPU on platform: NVIDIA CUDA",
    "Using OpenCL device: NVIDIA GeForce RTX 5090",
    "Loading SPIR-V binary into OpenCL...",
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


def _analyze(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    errors: list[str] = []
    try:
        recorded = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    except BaseException as error:
        return {"complete": False, "run_status": "invalid", "errors": [str(error)]}
    expected_exclusions = [
        "gpubpf device-hook attach backend", "cross-layer maps", "SIMT verifier",
        "application-kernel policy execution", "AMD or Intel execution", "performance",
    ]
    if (recorded.get("schema") != 1
            or recorded.get("scope") != "standalone eBPF-to-SPIR-V OpenCL demo only"
            or recorded.get("exclusions") != expected_exclusions):
        errors.append("run scope or explicit exclusions differ from the frozen plan")

    positive_log = (directory / "positive.log").read_text(encoding="utf-8", errors="replace")
    missing = [marker for marker in EXPECTED_MARKERS if marker not in positive_log]
    if missing:
        errors.append(f"positive output missing markers: {missing}")
    if final_exit(positive_log) != 0:
        errors.append("positive process did not exit zero")
    generated = re.findall(r"Generated SPIR-V binary: ([0-9]+) bytes", positive_log)
    patched = re.findall(r"Patched SPIR-V binary: ([0-9]+) bytes", positive_log)
    if len(generated) != 1 or len(patched) != 1:
        errors.append("positive output byte counts are ambiguous")
        patched_bytes = -1
    else:
        generated_bytes, patched_bytes = int(generated[0]), int(patched[0])
        if generated_bytes <= 20 or patched_bytes <= generated_bytes:
            errors.append("positive output byte counts are invalid")
        expected_positive = {
            "generated_bytes": generated_bytes, "patched_bytes": patched_bytes,
            "input": 100, "expected": 142, "actual": 142,
            "device": "NVIDIA GeForce RTX 5090",
        }
        if recorded.get("positive") != expected_positive:
            errors.append("recorded positive result differs from raw output")

    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    if execution.get("returncode") != 0:
        errors.append("execution record return code is not zero")
    if Path(execution.get("cwd", "")) != directory / "positive":
        errors.append("execution cwd is not the retained positive directory")
    if execution.get("environment") != {
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64",
    }:
        errors.append("execution environment differs from the frozen environment")
    binary_path = Path(recorded.get("binary", {}).get("path", ""))
    if execution.get("argv") != [str(binary_path)]:
        errors.append("execution argv is not the exact recorded source-native binary")
    if (type(execution.get("pid")) is not int or execution["pid"] <= 0
            or type(execution.get("started_ns")) is not int
            or type(execution.get("ended_ns")) is not int
            or execution["ended_ns"] <= execution["started_ns"]):
        errors.append("execution identity or timestamp interval is invalid")
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
    expected_build = {
        "LLVMBPF_ENABLE_SPIRV": "ON",
        "LLVM_DIR": "/usr/lib/llvm-20/lib/cmake/llvm",
        "CMAKE_HOME_DIRECTORY": "/home/yunwei37/workspace/gpu/bpftime-table1-575/vm/llvm-jit",
    }
    if recorded.get("build") != expected_build:
        errors.append("recorded build configuration differs from the frozen LLVM 20 build")
    identity = recorded.get("source_build_identity", {})
    if (identity.get("tracked_inputs_unmodified") is not True
            or identity.get("binary_newer_than_inputs") is not True
            or identity.get("opencl_loader_linked") is not True):
        errors.append("source/build identity gates were not recorded as passed")

    module = directory / "positive/bpf_program.spv"
    data = module.read_bytes()
    if len(data) != patched_bytes or len(data) < 20 or len(data) % 4:
        errors.append("retained SPIR-V size is invalid")
    if len(data) < 4 or int.from_bytes(data[:4], "little") != SPIRV_MAGIC:
        errors.append("retained SPIR-V magic is invalid")
    if recorded.get("spirv_header") != {
        "size_bytes": len(data), "magic_word": SPIRV_MAGIC,
    }:
        errors.append("recorded SPIR-V header differs from retained bytes")
    validation = subprocess.run(
        ["/usr/bin/spirv-val", str(module)], text=True, capture_output=True, check=False
    )
    if validation.returncode != 0:
        errors.append("independent spirv-val rejected the positive module")
    positive_validation_log = (directory / "positive-spirv-val.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if final_exit(positive_validation_log) != 0:
        errors.append("retained positive spirv-val did not exit zero")
    disassembly = subprocess.run(
        ["/usr/bin/spirv-dis", "--raw-id", str(module)],
        text=True, capture_output=True, check=False,
    )
    if disassembly.returncode != 0:
        errors.append("independent spirv-dis rejected the positive module")
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
        if structure["entry_point"] != 1 or structure["memory_model"] != 1 or structure["function"] < 1:
            errors.append(f"independent structure gates failed: {structure}")
        if recorded.get("structure") != structure:
            errors.append("recorded SPIR-V structure does not match independent disassembly")
    disassembly_log = (directory / "positive-spirv-dis.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if final_exit(disassembly_log) != 0:
        errors.append("retained positive spirv-dis did not exit zero")

    tampered = directory / "tampered-magic.spv"
    tampered_data = tampered.read_bytes()
    if len(tampered_data) != len(data) or tampered_data[:4] != bytes(4) or tampered_data[4:] != data[4:]:
        errors.append("tampered module is not an exact magic-word-only mutation")
    negative = subprocess.run(
        ["/usr/bin/spirv-val", str(tampered)], text=True, capture_output=True, check=False
    )
    if negative.returncode == 0:
        errors.append("independent spirv-val accepted the tampered module")
    negative_log = (directory / "tampered-spirv-val.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if final_exit(negative_log) == 0:
        errors.append("retained spirv-val accepted the tampered module")
    if (recorded.get("positive_validation_returncode") != 0
            or type(recorded.get("tampered_validation_returncode")) is not int
            or recorded["tampered_validation_returncode"] == 0):
        errors.append("recorded validator return codes are inconsistent")
    if recorded.get("tampered_submitted_to_opencl") is not False:
        errors.append("record does not explicitly exclude tampered OpenCL submission")
    if recorded.get("owned_process_survivors") != []:
        errors.append("owned process cleanup is incomplete")

    before, after = recorded.get("safety_before"), recorded.get("safety_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        errors.append("safety snapshots are missing")
    else:
        for label, snapshot in (("before", before), ("after", after)):
            gpu = snapshot.get("gpu", {})
            if snapshot.get("power_limit_service") != "active":
                errors.append(f"{label} power service is not active")
            if abs(float(snapshot.get("power_limit_w", -1)) - 400.0) > 0.01:
                errors.append(f"{label} power limit is not 400 W")
            if (gpu.get("driver") != "575.57.08" or gpu.get("compute_apps")
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
        for field in ("dmesg_abnormal", "journal_abnormal", "xids"):
            if before.get(field) != after.get(field):
                errors.append(f"kernel safety history changed in {field}")

    if recorded.get("status") != "complete":
        errors.append("runner status is not complete")
    if recorded.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        errors.append("analysis is not on the run's recorded boot")
    return {
        "complete": not errors,
        "run_status": "valid" if not errors else "invalid",
        "tested_hypothesis": "supported" if not errors else "inconclusive",
        "research_value": "supporting",
        "paper_impact": "narrow standalone SPIR-V implementation evidence",
        "next_paper_decision": (
            "describe only a working standalone NVIDIA OpenCL SPIR-V path"
            if not errors else "do not claim a runnable SPIR-V path"
        ),
        "errors": errors,
    }


def analyze(directory: Path) -> dict[str, Any]:
    try:
        return _analyze(directory)
    except BaseException as error:
        return {
            "complete": False,
            "run_status": "invalid",
            "tested_hypothesis": "inconclusive",
            "research_value": "supporting",
            "paper_impact": "narrow standalone SPIR-V implementation evidence",
            "next_paper_decision": "do not claim a runnable SPIR-V path",
            "errors": [f"{type(error).__name__}: {error}"],
        }


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
