#!/usr/bin/env python3
"""Run one source-native eBPF -> SPIR-V -> OpenCL correctness preflight."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


HERE = Path(__file__).resolve().parent
CAPABILITY_HELPER = HERE / "query_opencl_capability.py"
GPU_EXT = HERE.parents[2]
SAFETY_SOURCE = GPU_EXT / "workloads/moe-infinity/run_moe_head_to_head.py"
LEASE_PATHS = (
    Path("/tmp/gpubpf-revision-gpu0.lock"),
    Path("/tmp/gpubpf-revision-struct-ops.lock"),
)
EXPECTED_DRIVER = "575.57.08"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 5090"
OPENCL_LOADER = "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1"
SPIRV_MAGIC = 0x07230203
CL_SUCCESS = 0
CL_DEVICE_NOT_FOUND = -1
CL_DEVICE_TYPE_GPU = 1 << 2
CL_PLATFORM_VERSION = 0x0901
CL_PLATFORM_NAME = 0x0902
CL_PLATFORM_VENDOR = 0x0903
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_OPENCL_C_VERSION = 0x103D
CL_DEVICE_IL_VERSION = 0x105B
CL_DEVICE_NUMERIC_VERSION = 0x105E
CL_DEVICE_ILS_WITH_VERSION = 0x1061
FORBIDDEN_ENV_NAMES = {
    "LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH",
    "CUDA_INJECTION32_PATH", "OCL_ICD_VENDORS",
}
FORBIDDEN_ENV_PREFIXES = ("BPFTIME_", "NVBIT_", "OBS_", "GGML_")


class OpenCLNameVersion(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("name", ctypes.c_char * 64)]


class UnsupportedSPIRVIL(RuntimeError):
    """The selected OpenCL device does not advertise SPIR-V ingestion."""


def decode_opencl_version(value: int) -> dict[str, int]:
    return {
        "major": value >> 22,
        "minor": (value >> 12) & 0x3FF,
        "patch": value & 0xFFF,
    }


def spirv_il_advertised(capability: dict[str, Any]) -> bool:
    legacy = capability.get("cl_device_il_version", "")
    legacy_names = [token.partition("_")[0] for token in legacy.split()]
    versioned_names = [entry.get("name") for entry in capability.get(
        "cl_device_ils_with_version", []
    ) if isinstance(entry, dict)]
    return "SPIR-V" in legacy_names or "SPIR-V" in versioned_names


def require_spirv_il_capability(capability: dict[str, Any]) -> None:
    if capability.get("device_name") != EXPECTED_DEVICE:
        raise RuntimeError(
            f"selected OpenCL device is {capability.get('device_name')!r}, "
            f"expected {EXPECTED_DEVICE!r}"
        )
    if capability.get("driver_version") != EXPECTED_DRIVER:
        raise RuntimeError(
            f"selected OpenCL driver is {capability.get('driver_version')!r}, "
            f"expected {EXPECTED_DRIVER!r}"
        )
    if capability.get("supports_spirv_il") is not spirv_il_advertised(capability):
        raise RuntimeError("derived SPIR-V capability flag is inconsistent")
    if not capability.get("supports_spirv_il"):
        extension = capability.get("has_cl_khr_il_program", False)
        raise UnsupportedSPIRVIL(
            "selected OpenCL device does not advertise SPIR-V IL support: "
            f"CL_DEVICE_IL_VERSION={capability.get('cl_device_il_version')!r}, "
            "CL_DEVICE_ILS_WITH_VERSION="
            f"{capability.get('cl_device_ils_with_version')!r}, "
            f"cl_khr_il_program={extension}"
        )


def _configure_opencl(library: ctypes.CDLL) -> None:
    void_pointer = ctypes.c_void_p
    library.clGetPlatformIDs.argtypes = [
        ctypes.c_uint, ctypes.POINTER(void_pointer), ctypes.POINTER(ctypes.c_uint)
    ]
    library.clGetPlatformIDs.restype = ctypes.c_int
    library.clGetPlatformInfo.argtypes = [
        void_pointer, ctypes.c_uint, ctypes.c_size_t, void_pointer,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.clGetPlatformInfo.restype = ctypes.c_int
    library.clGetDeviceIDs.argtypes = [
        void_pointer, ctypes.c_ulong, ctypes.c_uint,
        ctypes.POINTER(void_pointer), ctypes.POINTER(ctypes.c_uint),
    ]
    library.clGetDeviceIDs.restype = ctypes.c_int
    library.clGetDeviceInfo.argtypes = [
        void_pointer, ctypes.c_uint, ctypes.c_size_t, void_pointer,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.clGetDeviceInfo.restype = ctypes.c_int


def _query_bytes(function: Any, handle: ctypes.c_void_p, parameter: int) -> bytes:
    size = ctypes.c_size_t()
    error = function(handle, parameter, 0, None, ctypes.byref(size))
    if error != CL_SUCCESS:
        raise RuntimeError(f"OpenCL query 0x{parameter:x} size failed with {error}")
    if size.value == 0:
        return b""
    buffer = (ctypes.c_ubyte * size.value)()
    error = function(handle, parameter, size.value, buffer, None)
    if error != CL_SUCCESS:
        raise RuntimeError(f"OpenCL query 0x{parameter:x} value failed with {error}")
    return bytes(buffer)


def _query_string(function: Any, handle: ctypes.c_void_p, parameter: int) -> str:
    raw = _query_bytes(function, handle, parameter)
    return raw.rstrip(b"\0").decode("utf-8", errors="strict")


def _query_uint(function: Any, handle: ctypes.c_void_p, parameter: int) -> int:
    value = ctypes.c_uint()
    size = ctypes.c_size_t()
    error = function(
        handle, parameter, ctypes.sizeof(value), ctypes.byref(value), ctypes.byref(size)
    )
    if error != CL_SUCCESS or size.value != ctypes.sizeof(value):
        raise RuntimeError(
            f"OpenCL uint query 0x{parameter:x} failed with {error}, size {size.value}"
        )
    return value.value


def _query_name_versions(
    function: Any, handle: ctypes.c_void_p, parameter: int,
) -> list[dict[str, Any]]:
    raw = _query_bytes(function, handle, parameter)
    entry_size = ctypes.sizeof(OpenCLNameVersion)
    if len(raw) % entry_size:
        raise RuntimeError(
            f"OpenCL name-version query 0x{parameter:x} returned {len(raw)} bytes"
        )
    records = []
    for offset in range(0, len(raw), entry_size):
        entry = OpenCLNameVersion.from_buffer_copy(raw[offset:offset + entry_size])
        records.append({
            "name": bytes(entry.name).split(b"\0", 1)[0].decode("ascii", errors="strict"),
            "version": decode_opencl_version(entry.version),
        })
    return records


def query_opencl_capability(loader_name: str = OPENCL_LOADER) -> dict[str, Any]:
    """Mirror the demo's first-GPU selection and query its advertised ILs."""

    library = ctypes.CDLL(loader_name)
    _configure_opencl(library)
    count = ctypes.c_uint()
    error = library.clGetPlatformIDs(0, None, ctypes.byref(count))
    if error != CL_SUCCESS or count.value == 0:
        raise RuntimeError(f"OpenCL platform enumeration failed with {error}")
    platforms = (ctypes.c_void_p * count.value)()
    error = library.clGetPlatformIDs(count.value, platforms, None)
    if error != CL_SUCCESS:
        raise RuntimeError(f"OpenCL platform retrieval failed with {error}")

    selected_platform = None
    selected_device = None
    selected_platform_index = None
    for index, platform in enumerate(platforms):
        device_count = ctypes.c_uint()
        error = library.clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(device_count)
        )
        if error == CL_DEVICE_NOT_FOUND:
            continue
        if error != CL_SUCCESS:
            raise RuntimeError(f"OpenCL GPU enumeration failed with {error}")
        if device_count.value == 0:
            continue
        devices = (ctypes.c_void_p * device_count.value)()
        error = library.clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, device_count.value, devices, None
        )
        if error != CL_SUCCESS:
            raise RuntimeError(f"OpenCL GPU retrieval failed with {error}")
        selected_platform, selected_device = platform, devices[0]
        selected_platform_index = index
        break
    if selected_platform is None or selected_device is None:
        raise RuntimeError("no OpenCL GPU device is available")

    platform_info = library.clGetPlatformInfo
    device_info = library.clGetDeviceInfo
    extensions = _query_string(device_info, selected_device, CL_DEVICE_EXTENSIONS).split()
    il_version = _query_string(device_info, selected_device, CL_DEVICE_IL_VERSION)
    ils_with_version = _query_name_versions(
        device_info, selected_device, CL_DEVICE_ILS_WITH_VERSION
    )
    capability = {
        "loader": loader_name,
        "platform_count": count.value,
        "selected_platform_index": selected_platform_index,
        "platform_name": _query_string(
            platform_info, selected_platform, CL_PLATFORM_NAME
        ),
        "platform_vendor": _query_string(
            platform_info, selected_platform, CL_PLATFORM_VENDOR
        ),
        "platform_version": _query_string(
            platform_info, selected_platform, CL_PLATFORM_VERSION
        ),
        "device_name": _query_string(device_info, selected_device, CL_DEVICE_NAME),
        "device_vendor": _query_string(device_info, selected_device, CL_DEVICE_VENDOR),
        "driver_version": _query_string(device_info, selected_device, CL_DRIVER_VERSION),
        "device_version": _query_string(device_info, selected_device, CL_DEVICE_VERSION),
        "device_numeric_version": decode_opencl_version(
            _query_uint(device_info, selected_device, CL_DEVICE_NUMERIC_VERSION)
        ),
        "opencl_c_version": _query_string(
            device_info, selected_device, CL_DEVICE_OPENCL_C_VERSION
        ),
        "cl_device_il_version": il_version,
        "cl_device_ils_with_version": ils_with_version,
        "has_cl_khr_il_program": "cl_khr_il_program" in extensions,
        "extensions": extensions,
    }
    capability["supports_intermediate_language_programs"] = bool(
        il_version or ils_with_version
    )
    capability["supports_spirv_il"] = spirv_il_advertised(capability)
    return capability


def query_opencl_capability_isolated() -> dict[str, Any]:
    """Query in a child so the OpenCL library releases its UVM references."""

    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64",
    }
    execution = subprocess.run(
        [sys.executable, "-B", str(CAPABILITY_HELPER)],
        cwd=HERE,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if execution.returncode != 0:
        raise RuntimeError(
            "isolated OpenCL capability query failed "
            f"({execution.returncode}): {execution.stderr.strip()}"
        )
    try:
        capability = json.loads(execution.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("isolated OpenCL capability query returned invalid JSON") from error
    if not isinstance(capability, dict):
        raise RuntimeError("isolated OpenCL capability query did not return an object")
    return capability


def load_safety_module():
    spec = importlib.util.spec_from_file_location("spirv_preflight_safety", SAFETY_SOURCE)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load safety module: {SAFETY_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def command_result(
    argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=cwd, env=env, text=True, capture_output=True,
        timeout=timeout, check=False,
    )


def write_command_log(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        "$ " + " ".join(result.args) + "\n\n## stdout\n" + result.stdout
        + "\n## stderr\n" + result.stderr
        + f"\n# exit: {result.returncode}\n",
        encoding="utf-8",
    )


class ReadOnlyLeases:
    """Acquire pre-created experiment locks without modifying their inodes."""

    def __init__(self, paths: tuple[Path, ...] = LEASE_PATHS):
        self.streams = []
        try:
            for path in paths:
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode):
                    raise RuntimeError(f"lease is not a regular file: {path}")
                stream = path.open("r")
                opened = os.fstat(stream.fileno())
                current = path.lstat()
                identity = (before.st_dev, before.st_ino)
                if ((opened.st_dev, opened.st_ino) != identity
                        or (current.st_dev, current.st_ino) != identity):
                    stream.close()
                    raise RuntimeError(f"lease inode changed while opening: {path}")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    stream.close()
                    raise
                self.streams.append(stream)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for stream in reversed(self.streams):
            stream.close()
        self.streams.clear()


def group_members(group: int) -> list[int]:
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == group and int(fields[3]) == group:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    for sig, delay in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        process.poll()
        if not group_members(process.pid):
            process.wait(timeout=1)
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            process.poll()
            if not group_members(process.pid):
                process.wait(timeout=1)
                return
            time.sleep(0.1)
    raise RuntimeError(f"owned process group survived cleanup: {process.pid}")


def reject_ambient_injection(environment: dict[str, str]) -> None:
    conflicts = sorted(
        key for key in environment
        if key in FORBIDDEN_ENV_NAMES or key.startswith(FORBIDDEN_ENV_PREFIXES)
    )
    if conflicts:
        raise RuntimeError(f"ambient injection variables are forbidden: {conflicts}")
    if environment.get("CUDA_VISIBLE_DEVICES", "0") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be absent or exactly 0")


def parse_cache(build_dir: Path) -> dict[str, str]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"missing CMake cache: {cache}")
    values = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        left, separator, value = line.partition("=")
        if separator and ":" in left:
            values[left.partition(":")[0]] = value
    required = {
        "LLVMBPF_ENABLE_SPIRV": "ON",
        "LLVM_DIR": "/usr/lib/llvm-20/lib/cmake/llvm",
        "CMAKE_HOME_DIRECTORY": "/home/yunwei37/workspace/gpu/bpftime-table1-575/vm/llvm-jit",
    }
    for key, expected in required.items():
        if values.get(key) != expected:
            raise RuntimeError(f"CMake cache {key}={values.get(key)!r}, expected {expected!r}")
    return {key: values[key] for key in required}


def file_record(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "executable": os.access(path, os.X_OK),
    }


def source_build_identity(source: Path, binary: Path, build_dir: Path) -> dict[str, Any]:
    source_repo = source.parents[2]
    expected_source = source_repo / "example/spirv/spirv_opencl_test.cpp"
    expected_binary = build_dir / "example/spirv/spirv_opencl_test"
    if source != expected_source or binary != expected_binary:
        raise RuntimeError("source/binary do not match the frozen source-native build layout")
    relevant = (
        source,
        source_repo / "CMakeLists.txt",
        source_repo / "example/spirv/CMakeLists.txt",
        source_repo / "src/llvm_jit_context.cpp",
        source_repo / "src/vm.cpp",
    )
    if any(not path.is_file() for path in relevant):
        raise RuntimeError("a required source-native SPIR-V build input is missing")
    revision = command_result(["git", "rev-parse", "HEAD"], cwd=source_repo, timeout=10)
    if revision.returncode != 0 or not revision.stdout.strip():
        raise RuntimeError("cannot record the llvm-jit source revision")
    relative = [str(path.relative_to(source_repo)) for path in relevant]
    listed = command_result(
        ["git", "ls-files", "--error-unmatch", "--", *relative],
        cwd=source_repo, timeout=10,
    )
    if listed.returncode != 0 or set(listed.stdout.splitlines()) != set(relative):
        raise RuntimeError("a required SPIR-V build input is not tracked at the recorded revision")
    tracked = command_result(["git", "diff", "--quiet", "--", *relative], cwd=source_repo, timeout=10)
    staged = command_result(
        ["git", "diff", "--cached", "--quiet", "--", *relative], cwd=source_repo, timeout=10
    )
    if tracked.returncode != 0 or staged.returncode != 0:
        raise RuntimeError("tracked SPIR-V source/build inputs contain local edits")
    binary_info = binary.stat()
    newest_source_mtime = max(path.stat().st_mtime_ns for path in relevant)
    if binary_info.st_mtime_ns < newest_source_mtime:
        raise RuntimeError("SPIR-V demo executable is older than a required source input")
    linked = command_result(["/usr/bin/ldd", str(binary)], timeout=10)
    if linked.returncode != 0 or "libOpenCL.so.1" not in linked.stdout or "not found" in linked.stdout:
        raise RuntimeError("SPIR-V demo is not linked to an available OpenCL loader")
    return {
        "source_repo": str(source_repo),
        "source_revision": revision.stdout.strip(),
        "tracked_inputs_unmodified": True,
        "binary_newer_than_inputs": True,
        "opencl_loader_linked": True,
        "relevant_inputs": [file_record(path) for path in relevant],
    }


def parse_positive_output(text: str) -> dict[str, Any]:
    required = (
        "SPIR-V target found successfully",
        "Generating SPIR-V from eBPF program...",
        "Patching SPIR-V to add kernel entry point...",
        "SPIR-V binary saved to bpf_program.spv",
        "Found GPU on platform: NVIDIA CUDA",
        f"Using OpenCL device: {EXPECTED_DEVICE}",
        "Loading SPIR-V binary into OpenCL...",
        "Building OpenCL program...",
        "Creating kernel 'bpf_main'...",
        "Executing eBPF program on GPU via OpenCL...",
        "Input value (arr[0]): 100",
        "Expected output (arr[1]): 142",
        "Actual output (arr[1]): 142",
        "Test PASSED!",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"positive demo output missing markers: {missing}")
    generated = re.findall(r"Generated SPIR-V binary: ([0-9]+) bytes", text)
    patched = re.findall(r"Patched SPIR-V binary: ([0-9]+) bytes", text)
    if len(generated) != 1 or len(patched) != 1:
        raise RuntimeError("expected exactly one generated and patched byte count")
    generated_bytes, patched_bytes = int(generated[0]), int(patched[0])
    if (generated_bytes <= 20 or patched_bytes <= 20
            or generated_bytes % 4 or patched_bytes % 4
            or patched_bytes == generated_bytes):
        raise RuntimeError("SPIR-V generation/entry-point patch byte counts are invalid")
    return {
        "generated_bytes": generated_bytes,
        "patched_bytes": patched_bytes,
        "input": 100,
        "expected": 142,
        "actual": 142,
        "device": EXPECTED_DEVICE,
    }


def require_spirv_header(path: Path, expected_bytes: int) -> dict[str, int]:
    data = path.read_bytes()
    if len(data) != expected_bytes or len(data) < 20 or len(data) % 4:
        raise RuntimeError("SPIR-V size is inconsistent with the demo record")
    magic = int.from_bytes(data[:4], "little")
    if magic != SPIRV_MAGIC:
        raise RuntimeError("SPIR-V magic word is invalid")
    return {"size_bytes": len(data), "magic_word": magic}


def require_structure(disassembly: str) -> dict[str, int]:
    gates = {
        "entry_point": len(re.findall(r'OpEntryPoint\s+Kernel\s+%\S+\s+"bpf_main"', disassembly)),
        "memory_model": len(re.findall(r"OpMemoryModel\s+Physical64\s+OpenCL", disassembly)),
        "function": len(re.findall(r"\bOpFunction\b", disassembly)),
    }
    if gates["entry_point"] != 1 or gates["memory_model"] != 1 or gates["function"] < 1:
        raise RuntimeError(f"SPIR-V structural gates failed: {gates}")
    return gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-device", action="store_true")
    args = parser.parse_args()

    if not args.execute_device:
        parser.error("device execution is fail-closed; pass --execute-device explicitly")
    binary, source, build_dir = (
        args.binary.resolve(), args.source.resolve(), args.build_dir.resolve()
    )
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"missing executable: {binary}")
    if not source.is_file():
        raise RuntimeError(f"missing source: {source}")
    build_record = parse_cache(build_dir)
    identity_record = source_build_identity(source, binary, build_dir)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    result: dict[str, Any] = {
        "schema": 2,
        "status": "running",
        "scope": "standalone eBPF-to-SPIR-V OpenCL demo only",
        "exclusions": [
            "gpubpf device-hook attach backend", "cross-layer maps",
            "SIMT verifier", "application-kernel policy execution",
            "AMD or Intel execution", "performance",
        ],
        "binary": file_record(binary),
        "source": file_record(source),
        "build": build_record,
        "source_build_identity": identity_record,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "capability_checked_before_demo_process": False,
        "demo_process_started": False,
    }
    atomic_write_json(output / "result.json", result)

    safety = load_safety_module()
    leases = None
    process = None
    before = None
    cleanup_errors: list[str] = []
    pending_error: BaseException | None = None
    exit_code = 0
    try:
        reject_ambient_injection(dict(os.environ))
        leases = ReadOnlyLeases()
        before = safety.safety_snapshot()
        safety.validate_pre_server_safety(before)
        if before["gpu"]["driver"] != EXPECTED_DRIVER:
            raise RuntimeError(f"driver must be {EXPECTED_DRIVER}")
        capability = query_opencl_capability_isolated()
        result["device_capability"] = capability
        result["capability_checked_before_demo_process"] = True
        atomic_write_json(output / "device-capability.json", capability)
        atomic_write_json(output / "result.json", result)
        require_spirv_il_capability(capability)

        positive_dir = output / "positive"
        positive_dir.mkdir()
        environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "CUDA_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64",
        }
        argv = [str(binary)]
        started_ns = time.time_ns()
        result["demo_process_started"] = True
        atomic_write_json(output / "result.json", result)
        process = subprocess.Popen(
            argv, cwd=positive_dir, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            stop_owned(process)
            stdout, stderr = process.communicate()
            raise RuntimeError("source-native OpenCL demo exceeded 120 seconds")
        execution = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        write_command_log(output / "positive.log", execution)
        atomic_write_json(output / "execution.json", {
            "argv": argv, "cwd": str(positive_dir), "environment": environment,
            "started_ns": started_ns, "ended_ns": time.time_ns(),
            "pid": process.pid, "returncode": process.returncode,
        })
        if process.returncode != 0:
            raise RuntimeError(f"source-native OpenCL demo exited {process.returncode}")
        positive = parse_positive_output(stdout + stderr)
        module = positive_dir / "bpf_program.spv"
        header = require_spirv_header(module, positive["patched_bytes"])

        validation = command_result(["/usr/bin/spirv-val", str(module)])
        write_command_log(output / "positive-spirv-val.log", validation)
        if validation.returncode != 0:
            raise RuntimeError("emitted SPIR-V failed spirv-val")
        disassembly = command_result(["/usr/bin/spirv-dis", "--raw-id", str(module)])
        write_command_log(output / "positive-spirv-dis.log", disassembly)
        if disassembly.returncode != 0:
            raise RuntimeError("emitted SPIR-V failed spirv-dis")
        structure = require_structure(disassembly.stdout)

        tampered_path = output / "tampered-magic.spv"
        tampered = bytearray(module.read_bytes())
        tampered[:4] = (0).to_bytes(4, "little")
        tampered_path.write_bytes(tampered)
        negative = command_result(["/usr/bin/spirv-val", str(tampered_path)])
        write_command_log(output / "tampered-spirv-val.log", negative)
        if negative.returncode == 0:
            raise RuntimeError("spirv-val accepted the deliberately invalid magic word")
        result.update(
            positive=positive,
            spirv_header=header,
            structure=structure,
            positive_validation_returncode=validation.returncode,
            tampered_validation_returncode=negative.returncode,
            tampered_submitted_to_opencl=False,
        )
    except UnsupportedSPIRVIL as error:
        result.update(status="unsupported", error=f"{type(error).__name__}: {error}")
        exit_code = 2
    except BaseException as error:
        result.update(status="invalid", error=f"{type(error).__name__}: {error}")
        pending_error = error
        exit_code = 1
    finally:
        try:
            stop_owned(process)
            survivors = group_members(process.pid) if process is not None else []
            if survivors:
                raise RuntimeError(f"owned process survivors: {survivors}")
            result["owned_process_survivors"] = survivors
        except BaseException as error:
            cleanup_errors.append(str(error))
        try:
            if before is not None:
                after = safety.wait_for_post_server_safety(before)
                result["safety_before"] = before
                result["safety_after"] = after
                result["boot_id_after"] = Path(
                    "/proc/sys/kernel/random/boot_id"
                ).read_text().strip()
                if result["boot_id"] != result["boot_id_after"]:
                    raise RuntimeError("boot changed during the preflight")
        except BaseException as error:
            cleanup_errors.append(str(error))
        if leases is not None:
            leases.close()
        if cleanup_errors:
            result.update(status="invalid", cleanup_errors=cleanup_errors)
        elif result.get("error") is None:
            result["status"] = "complete"
        atomic_write_json(output / "result.json", result)
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
    if pending_error is not None:
        raise pending_error
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
