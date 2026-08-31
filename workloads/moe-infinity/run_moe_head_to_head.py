#!/usr/bin/env python3
"""Fail-closed harness for the approved MoE-Infinity head-to-head plan.

The default ``admit`` action is read-only and never launches a GPU workload.
Any execution action uses the same admission gate and refuses foreign GPU or
struct_ops state; cleanup is limited to exact child PIDs and recorded link IDs.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import http.client
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
EXTENSION = GPU_EXT / "extension"
LLAMA_ROOT = GPU_EXT / "workloads/llama.cpp"
LLAMA_SOURCE = LLAMA_ROOT / "llama.cpp"
LLAMA_SERVER = LLAMA_ROOT / "build/bin/llama-server"
LLAMA_TOKENIZE = LLAMA_ROOT / "build/bin/llama-tokenize"
MOE_SOURCE = HERE / "deps/MoE-Infinity"
MOE_PYTHON = HERE / ".venv/bin/python"
POLICY_LOADER = EXTENSION / "prefetch_stride_lfu"
POLICY_BPF_OBJECT = EXTENSION / ".output/prefetch_stride_lfu.bpf.o"
EVICTION_MONITOR = HERE / "uvm_eviction_monitor"
KERNEL_MODULE_ROOT = GPU_EXT.parent / "gpu_ext-kernel-610/kernel-open"
LOADED_UVM_VERSION = Path("/sys/module/nvidia_uvm/version")
LOADED_UVM_BTF = Path("/sys/kernel/btf/nvidia_uvm")
ARTIFACTS = HERE / "artifacts-current.json"
WORKLOAD_MANIFEST = HERE / "workload-manifest.json"
PROMPTS = HERE / "prompts.json"
SCHEDULE = HERE / "schedule.json"
PLAN = HERE / "plan.md"
RUNNER = Path(__file__).resolve()

HF_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
GGUF_REVISION = "238abdd290bb874b90a5da1b4549881b7d05c091"
HF_SNAPSHOT = (
    HERE / "deps/hf-cache/hub/models--openai--gpt-oss-120b/snapshots" / HF_REVISION
)
MODEL_VIEW_PARENT = HERE / "deps/model-view"
MODEL_VIEW = MODEL_VIEW_PARENT / HF_REVISION
GGUF_MODEL = (
    HERE
    / "deps/hf-cache/hub/models--ggml-org--gpt-oss-120b-GGUF/snapshots"
    / GGUF_REVISION
    / "gpt-oss-120b-MXFP4.gguf"
)

EXPECTED_DRIVER = "610.43.02"
EXPECTED_GPU = "NVIDIA GeForce RTX 5090"
EXPECTED_LLAMA_COMMIT = "26836b27ae1ec9d6e94c6b56306cca75c7e86814"
EXPECTED_MOE_COMMIT = "b766f8f1f6379fac6cd23594713ba6f4c7650ad9"
EXPECTED_MOUNT_SOURCE = "/dev/disk/by-uuid/864c5664-999e-43c2-9967-4edaeee79d57"
EXPECTED_MOUNT_FSTYPE = "ext4"
CONFIGS = (
    "llama_ncmoe32",
    "llama_uvm",
    "gpubpf_host_stride_lfu",
    "moe_infinity_075",
)
TELEMETRY_CPU = 16


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run_checked(argv: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        argv, cwd=cwd, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise GateError(
            f"command failed ({result.returncode}): {argv!r}\n"
            f"{result.stderr[-4000:]}"
        )
    return result.stdout.strip()


def git_revision(repo: Path, expected: str, allow_instrumentation: bool = False) -> dict[str, Any]:
    actual = run_checked(["git", "rev-parse", "HEAD"], repo)
    if actual != expected:
        raise GateError(f"{repo}: expected commit {expected}, found {actual}")
    status = run_checked(["git", "status", "--porcelain"], repo).splitlines()
    if status and not allow_instrumentation:
        raise GateError(f"{repo}: source tree is dirty: {status}")
    if allow_instrumentation:
        patch = HERE / "instrumentation.patch"
        run_checked(["git", "apply", "--check", "--reverse", str(patch)], repo)
    return {"path": str(repo), "commit": actual, "status": status}


def controlled_environment(config: str) -> dict[str, str]:
    env = {
        "PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin",
        "HOME": "/home/yunwei37",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "CUDA_HOME": "/usr/local/cuda-12.9",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": str(HERE / "deps/hf-cache"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
    }
    if config in {"llama_uvm", "gpubpf_host_stride_lfu"}:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    if config == "moe_infinity_075":
        env.update(
            OMP_NUM_THREADS="8",
            MKL_NUM_THREADS="8",
            OPENBLAS_NUM_THREADS="8",
            NUMEXPR_NUM_THREADS="8",
            MOE_ENABLE_SM120="1",
            MOE_ENABLE_SM90="0",
            NVTX_DISABLE="1",
        )
    return env


def server_command(config: str, port: int, attempt_dir: Path) -> tuple[list[str], Path]:
    if config not in CONFIGS:
        raise GateError(f"unknown configuration: {config}")
    if config == "moe_infinity_075":
        offload = (attempt_dir / "moe-offload").resolve()
        return (
            [
                "taskset", "-c", "0-7", str(MOE_PYTHON), "-m",
                "moe_infinity.entrypoints.openai.revision_server",
                "--model", HF_REVISION,
                "--offload-dir", str(offload),
                "--host", "127.0.0.1", "--port", str(port),
                "--device-memory-ratio", "0.75", "--kv-cache-ratio", "0",
                "--max-batch-size", "1", "--startup-timeout", "1800",
                "--decode-step-timeout", "600",
            ],
            MODEL_VIEW_PARENT,
        )

    command = [
        "taskset", "-c", "0-7", str(LLAMA_SERVER),
        "--model", str(GGUF_MODEL), "--alias", "gpt-oss-120b",
        "--host", "127.0.0.1", "--port", str(port),
        "--n-gpu-layers", "99", "--parallel", "1", "--ctx-size", "4096",
        "--threads", "8", "--threads-batch", "8", "--cache-ram", "0",
        "--flash-attn", "on", "--no-warmup", "--timeout", "600",
    ]
    if config == "llama_ncmoe32":
        command.extend(["--n-cpu-moe", "32"])
    return command, LLAMA_ROOT


def policy_command() -> list[str]:
    return ["sudo", "-n", str(POLICY_LOADER), "-t", "2", "-n", "2", "-m", "128"]


def frozen_commands(attempt: int, port: int, raw_root: Path) -> dict[str, Any]:
    attempt_dir = (raw_root / f"attempt-{attempt:02d}").resolve()
    configurations: dict[str, Any] = {}
    for config in CONFIGS:
        argv, cwd = server_command(config, port, attempt_dir / config)
        configurations[config] = {
            "argv": argv,
            "cwd": str(cwd.resolve()),
            "environment": controlled_environment(config),
            "policy_argv": policy_command() if config == "gpubpf_host_stride_lfu" else None,
        }
    return {
        "schema": 1,
        "attempt": attempt,
        "port": port,
        "configuration_order": json.loads(SCHEDULE.read_text())["attempts"][attempt - 1][
            "configuration_order"
        ],
        "configurations": configurations,
    }


def gpu_state() -> dict[str, Any]:
    row = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.used,memory.total,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if len(row) != 1:
        raise GateError(f"expected exactly one GPU, found {len(row)}")
    fields = [field.strip() for field in row[0].split(",")]
    if len(fields) != 9:
        raise GateError(f"unexpected nvidia-smi output: {row[0]}")
    applications_raw = run_checked(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    applications = []
    for line in applications_raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            applications.append(
                {"pid": int(parts[0]), "name": parts[1], "memory_mib": int(parts[2])}
            )
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "driver": fields[2],
        "memory_used_mib": int(fields[3]),
        "memory_total_mib": int(fields[4]),
        "temperature_c": int(fields[5]),
        "sm_clock_mhz": int(fields[6]),
        "memory_clock_mhz": int(fields[7]),
        "power_w": float(fields[8]),
        "compute_apps": applications,
    }


def struct_ops_inventory() -> dict[str, Any]:
    maps = json.loads(run_checked(["sudo", "-n", "bpftool", "map", "show", "-j"]) or "[]")
    links = json.loads(run_checked(["sudo", "-n", "bpftool", "link", "show", "-j"]) or "[]")
    return {
        "maps": [item for item in maps if item.get("type") == "struct_ops"],
        "links": [item for item in links if item.get("type") == "struct_ops"],
    }


def mount_state(path: Path) -> dict[str, Any]:
    data = json.loads(
        run_checked(["findmnt", "-J", "-T", str(path), "-o", "SOURCE,FSTYPE,TARGET"])
    )
    entry = data["filesystems"][0]
    stat = os.statvfs(path)
    return {
        "source": entry["source"],
        "fstype": entry["fstype"],
        "target": entry["target"],
        "free_bytes": stat.f_bavail * stat.f_frsize,
    }


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def verify_small_artifacts() -> dict[str, Any]:
    frozen = json.loads(ARTIFACTS.read_text())
    workload = json.loads(WORKLOAD_MANIFEST.read_text())
    observed: dict[str, Any] = {
        "artifacts_manifest_sha256": sha256(ARTIFACTS),
        "workload_manifest_sha256": sha256(WORKLOAD_MANIFEST),
    }
    for key in ("prompts", "schedule", "bootstrap"):
        path = HERE / workload[key]["path"]
        actual = sha256(path)
        if actual != workload[key]["sha256"]:
            raise GateError(f"{key} artifact hash mismatch: {path}")
        observed[f"{key}_sha256"] = actual

    expected_hashes = {
        LLAMA_SERVER: frozen["llama_comparison"]["server_sha256"],
        LLAMA_TOKENIZE: workload["inputs"]["llama_tokenize_sha256"],
        HERE / frozen["measurement_instrumentation"]["patch"]: frozen[
            "measurement_instrumentation"
        ]["patch_sha256"],
        EXTENSION / frozen["combined_policy"]["bpf_source"].replace("../../extension/", ""):
            frozen["combined_policy"]["bpf_source_sha256"],
        EXTENSION / frozen["combined_policy"]["loader_source"].replace("../../extension/", ""):
            frozen["combined_policy"]["loader_source_sha256"],
        POLICY_BPF_OBJECT: frozen["combined_policy"]["bpf_object_sha256"],
        POLICY_LOADER: frozen["combined_policy"]["loader_sha256"],
        EVICTION_MONITOR: frozen["experiment_harness"]["uvm_eviction_monitor_sha256"],
        HERE / frozen["experiment_harness"]["uvm_eviction_monitor_source"]:
            frozen["experiment_harness"]["uvm_eviction_monitor_source_sha256"],
        HERE / frozen["experiment_harness"]["commands"]:
            frozen["experiment_harness"]["commands_sha256"],
        RUNNER: frozen["experiment_harness"]["runner_sha256"],
        PLAN: frozen["experiment_harness"]["plan_sha256"],
    }
    for name, metadata in frozen["custom_driver_modules"].items():
        expected_hashes[KERNEL_MODULE_ROOT / name] = metadata["sha256"]
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha256(path) != expected:
            raise GateError(f"executable/instrumentation hash mismatch: {path}")

    required_runtime = (MOE_PYTHON, POLICY_LOADER, POLICY_BPF_OBJECT, EVICTION_MONITOR)
    for path in required_runtime:
        if not path.is_file():
            raise GateError(f"required runtime artifact is missing: {path}")
    observed["runtime_sha256"] = {str(path): sha256(path) for path in required_runtime}
    for name, metadata in frozen["custom_driver_modules"].items():
        path = KERNEL_MODULE_ROOT / name
        modinfo = run_checked(["modinfo", str(path)])
        version = re.search(r"^version:\s+(\S+)", modinfo, re.MULTILINE)
        vermagic = re.search(r"^vermagic:\s+(.+)$", modinfo, re.MULTILINE)
        if not version or version.group(1) != EXPECTED_DRIVER:
            raise GateError(f"custom module version mismatch: {path}")
        if not vermagic or vermagic.group(1) != metadata["vermagic"]:
            raise GateError(f"custom module vermagic mismatch: {path}")
    return observed


def verify_loaded_uvm_interface() -> dict[str, Any]:
    """Prove that the live UVM module exposes the frozen gpubpf ABI."""
    if not LOADED_UVM_VERSION.is_file():
        raise GateError(f"loaded UVM version is unavailable: {LOADED_UVM_VERSION}")
    version = LOADED_UVM_VERSION.read_text().strip()
    if version != EXPECTED_DRIVER:
        raise GateError(f"loaded UVM version mismatch: expected {EXPECTED_DRIVER}, found {version}")
    if not LOADED_UVM_BTF.is_file():
        raise GateError(
            "loaded nvidia_uvm has no module BTF; the custom gpubpf UVM interface "
            "is not proven"
        )
    raw = run_checked(
        ["sudo", "-n", "bpftool", "btf", "dump", "file", str(LOADED_UVM_BTF),
         "format", "raw"]
    )
    struct_match = re.search(
        r"STRUCT 'gpu_mem_ops' size=48 vlen=6\n(?P<members>(?:\t[^\n]+\n){6})",
        raw,
    )
    expected_members = (
        "gpu_test_trigger",
        "gpu_page_prefetch",
        "gpu_page_prefetch_iter",
        "gpu_block_activate",
        "gpu_block_access",
        "gpu_evict_prepare",
    )
    if struct_match is None:
        raise GateError("loaded UVM BTF lacks the exact six-member gpu_mem_ops ABI")
    members = tuple(re.findall(r"\t'([^']+)'", struct_match.group("members")))
    if members != expected_members:
        raise GateError(f"loaded gpu_mem_ops member mismatch: {members}")
    required_kfuncs = (
        "bpf_gpu_block_move_head",
        "bpf_gpu_block_move_tail",
        "bpf_gpu_set_prefetch_region",
    )
    missing_kfuncs = [name for name in required_kfuncs if f"FUNC '{name}'" not in raw]
    if missing_kfuncs:
        raise GateError(f"loaded UVM BTF lacks policy kfuncs: {missing_kfuncs}")
    return {
        "version": version,
        "btf_path": str(LOADED_UVM_BTF),
        "btf_dump_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "gpu_mem_ops_members": list(members),
        "required_kfuncs": list(required_kfuncs),
    }


def verify_model_artifacts(full_hashes: bool) -> dict[str, Any]:
    frozen = json.loads(ARTIFACTS.read_text())
    expected_weights = frozen["model"]["weight_sha256"]
    actual_weights = {path.name for path in HF_SNAPSHOT.glob("*.safetensors")}
    if actual_weights != set(expected_weights):
        raise GateError(
            f"HF root shard set mismatch: missing={set(expected_weights) - actual_weights}, "
            f"extra={actual_weights - set(expected_weights)}"
        )
    expected_view = set(expected_weights) | set(frozen["model"]["metadata_sha256"])
    actual_view = {path.name for path in MODEL_VIEW.iterdir()}
    if actual_view != expected_view:
        raise GateError(
            f"admitted model view mismatch: missing={expected_view - actual_view}, "
            f"extra={actual_view - expected_view}"
        )
    for path in MODEL_VIEW.iterdir():
        if not path.is_symlink():
            raise GateError(f"model-view member is not a snapshot symlink: {path}")
        target = path.readlink()
        if not target.is_absolute():
            target = path.parent / target
        if not target.absolute().is_relative_to(HF_SNAPSHOT.absolute()):
            raise GateError(f"model-view link escapes the frozen snapshot: {path} -> {target}")

    result: dict[str, Any] = {
        "hf_snapshot": str(HF_SNAPSHOT.resolve()),
        "model_view": str(MODEL_VIEW.resolve()),
        "view_members": sorted(actual_view),
        "gguf": str(GGUF_MODEL.resolve()),
        "full_hashes": full_hashes,
    }
    if full_hashes:
        for name, expected in expected_weights.items():
            if sha256(HF_SNAPSHOT / name) != expected:
                raise GateError(f"HF shard hash mismatch: {name}")
        for name, expected in frozen["model"]["metadata_sha256"].items():
            if sha256(HF_SNAPSHOT / name) != expected:
                raise GateError(f"HF metadata hash mismatch: {name}")
        if sha256(GGUF_MODEL) != frozen["llama_comparison"]["model_sha256"]:
            raise GateError("GGUF hash mismatch")
        result["all_content_hashes_verified"] = True
    else:
        result["all_sizes"] = {
            name: (HF_SNAPSHOT / name).stat().st_size for name in sorted(actual_view)
        }
        if GGUF_MODEL.stat().st_size != frozen["llama_comparison"]["model_bytes"]:
            raise GateError("GGUF size mismatch")
    return result


def admission(port: int, full_hashes: bool = False) -> dict[str, Any]:
    evidence: dict[str, Any] = {"timestamp_ns": time.time_ns(), "errors": []}
    errors: list[str] = evidence["errors"]
    try:
        evidence["gpu"] = gpu_state()
        gpu = evidence["gpu"]
        if gpu["name"] != EXPECTED_GPU:
            errors.append(f"GPU: expected {EXPECTED_GPU}, found {gpu['name']}")
        if gpu["driver"] != EXPECTED_DRIVER:
            errors.append(f"driver: expected {EXPECTED_DRIVER}, found {gpu['driver']}")
        if gpu["compute_apps"]:
            errors.append(f"foreign GPU compute processes: {gpu['compute_apps']}")
        if gpu["memory_used_mib"] > 256:
            errors.append(f"GPU residual memory {gpu['memory_used_mib']} MiB exceeds 256 MiB")
    except Exception as exc:
        errors.append(f"GPU inventory: {exc}")
    try:
        evidence["struct_ops"] = struct_ops_inventory()
        if evidence["struct_ops"]["maps"] or evidence["struct_ops"]["links"]:
            errors.append(f"pre-existing struct_ops state: {evidence['struct_ops']}")
    except Exception as exc:
        errors.append(f"struct_ops inventory: {exc}")
    try:
        evidence["loaded_uvm_interface"] = verify_loaded_uvm_interface()
    except Exception as exc:
        errors.append(f"loaded UVM interface: {exc}")
    try:
        evidence["mount"] = mount_state(HERE)
        mount = evidence["mount"]
        if Path(mount["source"]).resolve() != Path(EXPECTED_MOUNT_SOURCE).resolve() or mount["fstype"] != EXPECTED_MOUNT_FSTYPE:
            errors.append(f"storage is not {EXPECTED_MOUNT_SOURCE} {EXPECTED_MOUNT_FSTYPE}: {mount}")
        if mount["free_bytes"] < 200 * 1024**3:
            errors.append(f"less than 200 GiB storage free: {mount['free_bytes']}")
    except Exception as exc:
        errors.append(f"storage inventory: {exc}")
    try:
        evidence["llama_source"] = git_revision(LLAMA_SOURCE, EXPECTED_LLAMA_COMMIT)
        evidence["moe_source"] = git_revision(
            MOE_SOURCE, EXPECTED_MOE_COMMIT, allow_instrumentation=True
        )
        evidence["small_artifacts"] = verify_small_artifacts()
        evidence["models"] = verify_model_artifacts(full_hashes)
    except Exception as exc:
        errors.append(f"frozen artifacts: {exc}")
    if not port_is_free(port):
        errors.append(f"port {port} is already in use")
    evidence["admitted"] = not errors
    return evidence


@dataclass
class LeaseSet:
    files: list[Any]

    @classmethod
    def acquire(cls) -> "LeaseSet":
        files = []
        for path in (
            Path("/tmp/gpubpf-revision-gpu0.lock"),
            Path("/tmp/gpubpf-revision-struct-ops.lock"),
        ):
            stream = path.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                stream.close()
                for held in files:
                    held.close()
                raise GateError(f"exclusive experiment lease is busy: {path}") from exc
            files.append(stream)
        return cls(files)

    def close(self) -> None:
        for stream in reversed(self.files):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        self.files.clear()


def http_json(port: int, path: str, payload: dict[str, Any] | None = None,
              timeout: float = 10) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            if response.status != 200:
                raise GateError(f"HTTP {response.status} from {path}: {body[-2000:]!r}")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        raise GateError(
            f"HTTP {exc.code} from {path}: {exc.read()[-2000:].decode(errors='replace')}"
        ) from exc


def http_text(port: int, path: str, timeout: float = 10) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
            body = response.read()
            if response.status != 200:
                raise GateError(f"HTTP {response.status} from {path}: {body[-2000:]!r}")
            return body.decode("utf-8", errors="strict")
    except urllib.error.HTTPError as exc:
        raise GateError(f"HTTP {exc.code} from {path}: {exc.read()[-2000:]!r}") from exc


def moe_snapshot(port: int) -> dict[str, Any]:
    text = http_text(port, "/metrics")
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, value = line.split(None, 1)
            metrics[name] = float(value)
    required = {"moe_tokens_generated_total", "moe_engine_steps_total",
                "moe_kv_cache_total_blocks"}
    if not required.issubset(metrics):
        raise GateError(f"MoE /metrics is missing required series: {required - set(metrics)}")
    return {"revision": http_json(port, "/revision/stats"), "metrics": metrics,
            "metrics_raw_sha256": hashlib.sha256(text.encode()).hexdigest()}


def completion_payload(config: str, token_ids: list[int], stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "gpt-oss-120b",
        "prompt": token_ids,
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": [],
        "stream": stream,
    }
    if config == "moe_infinity_075":
        payload.update(n=1, best_of=1, echo=False)
    else:
        payload.update(cache_prompt=False, return_tokens=True)
    return payload


def validate_completion_response(response: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise GateError(f"completion response does not have exactly one choice: {response}")
    choice = choices[0]
    if choice.get("finish_reason") != "length":
        raise GateError(f"completion did not reach max_tokens: {choice.get('finish_reason')}")
    text = choice.get("text")
    if not isinstance(text, str) or not text:
        raise GateError("completion returned empty or non-string text")
    text.encode("utf-8", errors="strict")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise GateError("completion response has no usage accounting")
    if int(usage.get("prompt_tokens", -1)) != prompt_tokens:
        raise GateError(f"prompt-token accounting mismatch: {usage}")
    if int(usage.get("completion_tokens", -1)) != 64:
        raise GateError(f"completion-token accounting mismatch: {usage}")
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "finish_reason": choice["finish_reason"],
        "usage": usage,
    }


def nonstream_completion(config: str, port: int, token_ids: list[int],
                         output: Path) -> dict[str, Any]:
    start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    response = http_json(
        port, "/v1/completions", completion_payload(config, token_ids, False), 600
    )
    end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    atomic_write_json(output, response)
    validated = validate_completion_response(response, len(token_ids))
    validated.update(start_ns=start_ns, end_ns=end_ns, e2e_ms=(end_ns - start_ns) / 1e6)
    return validated


def wait_ready(process: subprocess.Popen[Any], port: int, log_path: Path,
               timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace")[-6000:] if log_path.exists() else ""
            raise GateError(f"server exited {process.returncode} before ready\n{tail}")
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise GateError(f"server readiness exceeded {timeout} seconds")


def json_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            result.append(value)
    return result


def wait_event(process: subprocess.Popen[Any], path: Path, event: str,
               timeout: float = 30) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for value in reversed(json_events(path)):
            if value.get("event") == event:
                return value
        if process.poll() is not None:
            tail = path.read_text(errors="replace")[-4000:] if path.exists() else ""
            raise GateError(f"probe exited {process.returncode} before {event}: {tail}")
        time.sleep(0.1)
    raise GateError(f"timed out waiting for {event} from PID {process.pid}")


def latest_counter_event(path: Path, event: str) -> dict[str, Any]:
    for value in reversed(json_events(path)):
        if value.get("event") == event:
            return value
    raise GateError(f"no {event} record in {path}")


def descendants(root_pid: int) -> list[int]:
    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                text = stat.read_text()
                right = text.rfind(")")
                fields = text[right + 2 :].split()
                pid = int(stat.parent.name)
                ppid = int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid in found and pid not in found:
                found.add(pid)
                changed = True
    return sorted(found)


def process_tree_io(root_pid: int) -> dict[str, Any]:
    members = []
    read_bytes = 0
    write_bytes = 0
    cpu_ticks = 0
    for pid in descendants(root_pid):
        try:
            values = {}
            for line in (Path("/proc") / str(pid) / "io").read_text().splitlines():
                key, value = line.split(":", 1)
                values[key] = int(value.strip())
            affinity = sorted(os.sched_getaffinity(pid))
            stat_text = (Path("/proc") / str(pid) / "stat").read_text()
            stat_fields = stat_text[stat_text.rfind(")") + 2 :].split()
            member_ticks = int(stat_fields[11]) + int(stat_fields[12])
            members.append({"pid": pid, "read_bytes": values["read_bytes"],
                            "write_bytes": values["write_bytes"], "affinity": affinity,
                            "cpu_ticks": member_ticks})
            read_bytes += values["read_bytes"]
            write_bytes += values["write_bytes"]
            cpu_ticks += member_ticks
        except (OSError, ValueError, KeyError):
            continue
    return {"members": members, "read_bytes": read_bytes, "write_bytes": write_bytes,
            "cpu_time_s": cpu_ticks / os.sysconf("SC_CLK_TCK")}


def find_uvm_fd(root_pid: int) -> tuple[int, int]:
    matches = []
    for pid in descendants(root_pid):
        for fd_path in (Path("/proc") / str(pid) / "fd").glob("[0-9]*"):
            try:
                target = os.readlink(fd_path)
            except OSError:
                continue
            if target == "/dev/nvidia-uvm":
                matches.append((pid, int(fd_path.name)))
    if len(matches) != 1:
        raise GateError(f"expected one owned /dev/nvidia-uvm fd, found {matches}")
    return matches[0]


def duplicate_child_fd(pid: int, target_fd: int) -> int:
    pidfd = os.pidfd_open(pid, 0)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.syscall(438, pidfd, target_fd, 0)  # pidfd_getfd on x86_64
        if result < 0:
            error = ctypes.get_errno()
            raise GateError(f"pidfd_getfd({pid}, {target_fd}) failed: {os.strerror(error)}")
        return int(result)
    finally:
        os.close(pidfd)


def stop_owned_process_group(process: subprocess.Popen[Any], timeout: float = 60) -> None:
    if process.poll() is not None:
        return
    pgid = os.getpgid(process.pid)
    if pgid != process.pid:
        raise GateError(f"refusing cleanup: owned process-group ID drifted ({pgid} != {process.pid})")
    os.killpg(pgid, signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=20)


def stop_exact_process(process: subprocess.Popen[Any], timeout: float = 20) -> None:
    if process.poll() is not None:
        return
    os.kill(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=10)


def start_gpu_telemetry(run_dir: Path) -> tuple[subprocess.Popen[Any], Any, Path]:
    if TELEMETRY_CPU not in os.sched_getaffinity(0):
        raise GateError(f"telemetry CPU {TELEMETRY_CPU} is not available")
    path = run_dir / "gpu-telemetry.csv"
    log = path.open("x", buffering=1)
    query = ",".join((
        "timestamp", "memory.used", "temperature.gpu", "power.draw",
        "clocks.current.sm", "clocks.current.memory",
        "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
        "clocks_event_reasons.hw_thermal_slowdown",
        "clocks_event_reasons.hw_power_brake_slowdown",
        "clocks_event_reasons.sw_thermal_slowdown",
    ))
    process = subprocess.Popen(
        ["taskset", "-c", str(TELEMETRY_CPU), "nvidia-smi", f"--query-gpu={query}",
         "--format=csv", "--loop-ms=200"],
        stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    time.sleep(0.3)
    if process.poll() is not None:
        log.close()
        raise GateError(f"GPU telemetry exited early: {path.read_text(errors='replace')}")
    return process, log, path


def validate_gpu_telemetry(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if len(lines) < 2:
        raise GateError(f"GPU telemetry has no samples: {path}")
    headers = [item.strip() for item in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != len(headers):
            continue
        rows.append(dict(zip(headers, fields)))
    if not rows:
        raise GateError("GPU telemetry has no parseable rows")
    reason_headers = headers[6:]
    throttled = [
        {key: row[key] for key in reason_headers}
        for row in rows if any(row[key].lower() not in {"not active", "n/a"} for key in reason_headers)
    ]
    if throttled:
        raise GateError(f"GPU throttling observed in measured interval: {throttled[:5]}")
    def numeric(header: str) -> list[float]:
        return [float(row[header].split()[0]) for row in rows]
    return {
        "samples": len(rows), "peak_memory_mib": max(numeric(headers[1])),
        "peak_temperature_c": max(numeric(headers[2])),
        "mean_power_w": sum(numeric(headers[3])) / len(rows),
        "min_sm_clock_mhz": min(numeric(headers[4])),
        "max_sm_clock_mhz": max(numeric(headers[4])), "throttled": False,
    }


def start_policy(run_dir: Path) -> tuple[subprocess.Popen[Any], Any, dict[str, Any]]:
    log_path = run_dir / "policy.jsonl"
    log = log_path.open("x", buffering=1)
    process = subprocess.Popen(
        policy_command(), stdout=log, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    try:
        ready = wait_event(process, log_path, "ready", 30)
        inventory = struct_ops_inventory()
        link_ids = {int(item["id"]) for item in inventory["links"]}
        map_ids = {int(item["id"]) for item in inventory["maps"]}
        if link_ids != {int(ready["struct_link_id"])}:
            raise GateError(f"policy link ownership mismatch: ready={ready}, inventory={inventory}")
        if map_ids != {int(ready["struct_map_id"])}:
            raise GateError(f"policy map ownership mismatch: ready={ready}, inventory={inventory}")
        return process, log, ready
    except Exception:
        stop_exact_process(process)
        log.close()
        raise


def start_eviction_monitor(server_pid: int, run_dir: Path) -> tuple[subprocess.Popen[Any], Any, dict[str, Any]]:
    pid, target_fd = find_uvm_fd(server_pid)
    inherited_fd = duplicate_child_fd(pid, target_fd)
    log_path = run_dir / "uvm-evictions.jsonl"
    log = log_path.open("x", buffering=1)
    try:
        process = subprocess.Popen(
            [str(EVICTION_MONITOR), "--uvm-fd", str(inherited_fd)],
            stdout=log, stderr=subprocess.STDOUT, text=True,
            pass_fds=(inherited_fd,), start_new_session=True,
        )
    finally:
        os.close(inherited_fd)
    try:
        ready = wait_event(process, log_path, "ready", 30)
        ready.update(target_pid=pid, target_fd=target_fd)
        return process, log, ready
    except Exception:
        stop_exact_process(process)
        log.close()
        raise


def validate_log(log_path: Path) -> None:
    text = log_path.read_text(errors="replace")
    patterns = (
        r"Traceback", r"CUDA error", r"out of memory",
        r"fall(?:ing)? back[^\n]*(?:CPU|unsupported|kernel|buffered)",
        r"failed to load", r"illegal memory access", r"unsupported gpu architecture",
    )
    matches = [pattern for pattern in patterns if re.search(pattern, text, re.I)]
    if matches:
        raise GateError(f"fatal/fallback patterns in {log_path}: {matches}")


def validate_moe_odirect(trace_dir: Path, offload_dir: Path) -> dict[str, Any]:
    records = []
    root = offload_dir.resolve()
    for trace in sorted(trace_dir.glob("open.trace*")):
        for line_number, line in enumerate(trace.read_text(errors="replace").splitlines(), 1):
            match = re.search(r'open(?:at|at2)?\([^\n]*?"([^"]+)"([^\n]*)\)\s+=\s+(-?\d+)', line)
            if not match:
                continue
            path = Path(match.group(1))
            resolved = path.resolve(strict=False) if path.is_absolute() else (MODEL_VIEW_PARENT / path).resolve(strict=False)
            if resolved.is_relative_to(root):
                records.append({"trace": trace.name, "line": line_number,
                                "path": str(resolved), "flags": match.group(2),
                                "result_fd": int(match.group(3))})
    if not records:
        raise GateError("strace observed no expert-store opens under the admitted offload directory")
    bad = [item for item in records if "O_DIRECT" not in item["flags"] or item["result_fd"] < 0]
    if bad:
        raise GateError(f"expert-store open without successful O_DIRECT: {bad[:8]}")
    reads = [item for item in records if "O_RDONLY" in item["flags"] or "O_RDWR" in item["flags"]]
    writes = [item for item in records if "O_WRONLY" in item["flags"] or "O_RDWR" in item["flags"]]
    if not reads or not writes:
        raise GateError(f"O_DIRECT preflight lacks read/write coverage: reads={len(reads)}, writes={len(writes)}")
    return {"opens": len(records), "reads": len(reads), "writes": len(writes),
            "paths": sorted({item["path"] for item in records})}


def check_server_identity(config: str, port: int, prompts: dict[str, Any]) -> dict[str, Any]:
    models = http_json(port, "/v1/models")
    ids = {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}
    if "gpt-oss-120b" not in ids and HF_REVISION not in ids:
        raise GateError(f"unexpected served-model identity: {models}")
    result: dict[str, Any] = {"models": models}
    if config != "moe_infinity_075":
        detokenized = http_json(
            port, "/detokenize", {"tokens": prompts["records"][0]["prompt_token_ids"]}
        )
        if detokenized.get("content") != prompts["records"][0]["prompt_text"]:
            raise GateError("llama /detokenize differs from frozen canonical prompt")
        result["detokenize_sha256"] = hashlib.sha256(
            detokenized["content"].encode()
        ).hexdigest()
    return result


def counter_delta(before: dict[str, Any], after: dict[str, Any], keys: tuple[str, ...]) -> dict[str, int]:
    result = {}
    for key in keys:
        start = int(before[key])
        end = int(after[key])
        if end < start:
            raise GateError(f"counter {key} decreased: {start} -> {end}")
        result[key] = end - start
    return result


def run_correctness_config(config: str, run_dir: Path, port: int,
                           prompts: dict[str, Any]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    policy = None
    policy_log = None
    policy_ready = None
    monitor = None
    monitor_log = None
    server = None
    server_log = None
    log_path = run_dir / "server.log"
    try:
        if config == "gpubpf_host_stride_lfu":
            policy, policy_log, policy_ready = start_policy(run_dir)
        argv, cwd = server_command(config, port, run_dir)
        atomic_write_json(
            run_dir / "launch.json",
            {"argv": argv, "cwd": str(cwd), "environment": controlled_environment(config),
             "policy_ready": policy_ready},
        )
        launch_argv = argv
        trace_dir = None
        if config == "moe_infinity_075":
            trace_dir = run_dir / "strace"
            trace_dir.mkdir()
            launch_argv = [
                "/usr/bin/strace", "-ff", "-qq", "-s", "4096",
                "-e", "trace=open,openat,openat2", "-o", str(trace_dir / "open.trace"),
            ] + argv
        server_log = log_path.open("x", buffering=1)
        server = subprocess.Popen(
            launch_argv, cwd=cwd, env=controlled_environment(config), stdout=server_log,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        wait_ready(server, port, log_path, 1800 if config == "moe_infinity_075" else 900)
        identity = check_server_identity(config, port, prompts)
        if config == "gpubpf_host_stride_lfu":
            monitor, monitor_log, monitor_ready = start_eviction_monitor(server.pid, run_dir)
        else:
            monitor_ready = None

        warmup = nonstream_completion(
            config, port, prompts["records"][0]["prompt_token_ids"], run_dir / "warmup.json"
        )
        time.sleep(1.1)
        io_before = process_tree_io(server.pid)
        if any(item["affinity"] != list(range(8)) for item in io_before["members"]):
            raise GateError(f"owned process tree escaped CPU 0-7: {io_before['members']}")
        moe_before = moe_snapshot(port) if config == "moe_infinity_075" else None
        policy_before = (
            latest_counter_event(run_dir / "policy.jsonl", "engagement")
            if config == "gpubpf_host_stride_lfu" else None
        )
        eviction_before = (
            latest_counter_event(run_dir / "uvm-evictions.jsonl", "eviction_stats")
            if config == "gpubpf_host_stride_lfu" else None
        )

        passes: list[list[dict[str, Any]]] = []
        for pass_number in (1, 2):
            outputs = []
            for prompt_number, record in enumerate(prompts["records"][1:], start=1):
                outputs.append(
                    nonstream_completion(
                        config, port, record["prompt_token_ids"],
                        run_dir / f"smoke-pass{pass_number}-prompt{prompt_number}.json",
                    )
                )
            passes.append(outputs)
        for index, (first, second) in enumerate(zip(*passes), start=1):
            if first["text"] != second["text"]:
                raise GateError(f"non-deterministic smoke output for prompt {index}")

        time.sleep(1.1)
        io_after = process_tree_io(server.pid)
        io_delta = {
            "read_bytes": io_after["read_bytes"] - io_before["read_bytes"],
            "write_bytes": io_after["write_bytes"] - io_before["write_bytes"],
        }
        engagement: dict[str, Any] = {"process_io_delta": io_delta}
        if config == "moe_infinity_075":
            moe_after = moe_snapshot(port)
            delta = counter_delta(
                moe_before["revision"], moe_after["revision"],
                ("engine_generated_tokens", "engine_steps", "expert_cache_accesses",
                 "expert_cache_hits", "expert_cache_misses"),
            )
            metrics_delta = {
                "tokens": int(moe_after["metrics"]["moe_tokens_generated_total"])
                - int(moe_before["metrics"]["moe_tokens_generated_total"]),
                "steps": int(moe_after["metrics"]["moe_engine_steps_total"])
                - int(moe_before["metrics"]["moe_engine_steps_total"]),
            }
            if delta["engine_generated_tokens"] != 1024 or delta["engine_steps"] <= 0:
                raise GateError(f"MoE smoke token/step gate failed: {delta}")
            if metrics_delta != {"tokens": 1024, "steps": delta["engine_steps"]}:
                raise GateError(f"MoE /metrics and /revision/stats disagree: {metrics_delta}, {delta}")
            if delta["expert_cache_accesses"] <= 0 or (
                delta["expert_cache_hits"] + delta["expert_cache_misses"]
                != delta["expert_cache_accesses"]
            ):
                raise GateError(f"MoE expert cache gate failed: {delta}")
            if int(moe_after["revision"]["kv_cache_num_blocks"]) != 128 or io_delta["read_bytes"] <= 0:
                raise GateError(f"MoE KV/direct-read gate failed: stats={moe_after}, io={io_delta}")
            offload_files = sorted((run_dir / "moe-offload").rglob("*"))
            if not any(path.is_file() and path.stat().st_size > 0 for path in offload_files):
                raise GateError("MoE expert offload store is empty")
            engagement.update(before=moe_before, after=moe_after, delta=delta,
                              metrics_delta=metrics_delta)
            engagement["odirect"] = validate_moe_odirect(
                trace_dir, run_dir / "moe-offload"
            )
        elif config == "gpubpf_host_stride_lfu":
            policy_after = latest_counter_event(run_dir / "policy.jsonl", "engagement")
            eviction_after = latest_counter_event(
                run_dir / "uvm-evictions.jsonl", "eviction_stats"
            )
            policy_delta = counter_delta(
                policy_before, policy_after,
                ("page_fault_calls", "stride_detections", "prefetches_issued",
                 "lfu_activations", "lfu_accesses", "eviction_prepares"),
            )
            eviction_delta = counter_delta(
                eviction_before, eviction_after,
                ("evictions", "evicted_bytes", "dropped_evictions"),
            )
            if any(policy_delta[key] <= 0 for key in policy_delta):
                raise GateError(f"combined-policy smoke engagement gate failed: {policy_delta}")
            if eviction_delta["evictions"] <= 0 or eviction_delta["evicted_bytes"] <= 0:
                raise GateError(f"completed UVM eviction gate failed: {eviction_delta}")
            if eviction_delta["dropped_evictions"] != 0:
                raise GateError(f"UVM eviction event queue dropped records: {eviction_delta}")
            engagement.update(
                policy_before=policy_before, policy_after=policy_after,
                policy_delta=policy_delta, eviction_before=eviction_before,
                eviction_after=eviction_after, eviction_delta=eviction_delta,
            )
        result = {
            "config": config, "identity": identity, "monitor_ready": monitor_ready,
            "warmup": warmup, "goldens": [item["text_sha256"] for item in passes[0]],
            "passes": passes, "engagement": engagement,
        }
        atomic_write_json(run_dir / "result.json", result)
        return result
    finally:
        if server is not None:
            stop_owned_process_group(server)
        if server_log is not None:
            server_log.close()
        if monitor is not None:
            stop_exact_process(monitor)
        if monitor_log is not None:
            monitor_log.close()
        if policy is not None:
            stop_exact_process(policy)
        if policy_log is not None:
            policy_log.close()
        if log_path.exists():
            validate_log(log_path)
        if policy_ready is not None:
            inventory = struct_ops_inventory()
            if inventory["maps"] or inventory["links"]:
                raise GateError(f"owned policy did not detach cleanly: {inventory}")


def run_correctness_preflight(output: Path, port: int) -> dict[str, Any]:
    lease = LeaseSet.acquire()
    try:
        admitted = admission(port, full_hashes=True)
        if not admitted["admitted"]:
            raise GateError("admission refused:\n- " + "\n- ".join(admitted["errors"]))
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(output / "admission.json", admitted)
        prompts = json.loads(PROMPTS.read_text())
        order = json.loads(SCHEDULE.read_text())["attempts"][0]["configuration_order"]
        results = {}
        for config in order:
            results[config] = run_correctness_config(config, output / config, port, prompts)
        llama_goldens = [results[name]["goldens"] for name in (
            "llama_ncmoe32", "llama_uvm", "gpubpf_host_stride_lfu"
        )]
        if not all(value == llama_goldens[0] for value in llama_goldens[1:]):
            raise GateError("the three llama configurations have different smoke goldens")
        result = {"status": "passed", "configuration_order": order, "results": results}
        atomic_write_json(output / "preflight-result.json", result)
        return result
    finally:
        lease.close()


def streamed_completion(config: str, port: int, token_ids: list[int],
                        golden_text: str, raw_path: Path) -> dict[str, Any]:
    body = json.dumps(
        completion_payload(config, token_ids, True), separators=(",", ":")
    ).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    connection.request("POST", "/v1/completions", body=body,
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        error = response.read()[-2000:]
        connection.close()
        raise GateError(f"stream HTTP {response.status}: {error!r}")
    first_text_ns = None
    finish_reason = None
    done_ns = None
    fragments = []
    frames = []
    raw = bytearray()
    usage: dict[str, Any] | None = None
    while True:
        line = response.readline()
        if not line:
            break
        timestamp = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        raw.extend(line)
        stripped = line.strip()
        if not stripped.startswith(b"data: "):
            continue
        if stripped == b"data: [DONE]":
            done_ns = timestamp
            frames.append({"timestamp_ns": timestamp, "done": True})
            continue
        try:
            value = json.loads(stripped[6:])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GateError(f"invalid SSE JSON frame: {stripped[:500]!r}") from exc
        frame = {"timestamp_ns": timestamp, "sha256": hashlib.sha256(stripped[6:]).hexdigest()}
        if isinstance(value.get("usage"), dict):
            usage = value["usage"]
        for choice in value.get("choices", []):
            text_piece = choice.get("text") or ""
            if text_piece:
                if first_text_ns is None:
                    first_text_ns = timestamp
                fragments.append(text_piece)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
        frames.append(frame)
    eof_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    connection.close()
    with raw_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    text = "".join(fragments)
    if first_text_ns is None or done_ns is None or finish_reason != "length":
        raise GateError(
            f"incomplete SSE lifecycle: first={first_text_ns}, done={done_ns}, finish={finish_reason}"
        )
    if text != golden_text:
        raise GateError("streamed visible output differs from correctness-smoke golden")
    if config != "moe_infinity_075":
        if not isinstance(usage, dict) or int(usage.get("completion_tokens", -1)) != 64:
            raise GateError(f"llama stream token accounting mismatch: {usage}")
        if int(usage.get("prompt_tokens", -1)) != 512:
            raise GateError(f"llama stream prompt accounting mismatch: {usage}")
    return {
        "start_ns": start_ns, "first_text_ns": first_text_ns,
        "done_ns": done_ns, "eof_ns": eof_ns,
        "ttft_ms": (first_text_ns - start_ns) / 1e6,
        "e2e_ms": (eof_ns - start_ns) / 1e6,
        "finish_reason": finish_reason, "usage": usage,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "raw_sse_sha256": hashlib.sha256(raw).hexdigest(), "frames": frames,
    }


def engagement_snapshot(config: str, port: int, run_dir: Path,
                        server_pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"process_io": process_tree_io(server_pid)}
    if config == "moe_infinity_075":
        result["moe"] = moe_snapshot(port)
    elif config == "gpubpf_host_stride_lfu":
        result["policy"] = latest_counter_event(run_dir / "policy.jsonl", "engagement")
        result["evictions"] = latest_counter_event(
            run_dir / "uvm-evictions.jsonl", "eviction_stats"
        )
    return result


def validate_measured_engagement(config: str, before: dict[str, Any],
                                 after: dict[str, Any]) -> dict[str, Any]:
    io_delta = after["process_io"]["read_bytes"] - before["process_io"]["read_bytes"]
    result: dict[str, Any] = {
        "read_bytes": io_delta,
        "process_tree_cpu_time_s": after["process_io"]["cpu_time_s"]
        - before["process_io"]["cpu_time_s"],
    }
    if config == "moe_infinity_075":
        delta = counter_delta(
            before["moe"]["revision"], after["moe"]["revision"],
            ("engine_generated_tokens", "engine_steps", "expert_cache_accesses",
             "expert_cache_hits", "expert_cache_misses"),
        )
        metric_tokens = int(after["moe"]["metrics"]["moe_tokens_generated_total"]) - int(
            before["moe"]["metrics"]["moe_tokens_generated_total"]
        )
        metric_steps = int(after["moe"]["metrics"]["moe_engine_steps_total"]) - int(
            before["moe"]["metrics"]["moe_engine_steps_total"]
        )
        if delta["engine_generated_tokens"] != 512 or delta["engine_steps"] <= 0:
            raise GateError(f"MoE measured token/step gate failed: {delta}")
        if delta["expert_cache_accesses"] <= 0 or (
            delta["expert_cache_hits"] + delta["expert_cache_misses"]
            != delta["expert_cache_accesses"]
        ) or io_delta <= 0:
            raise GateError(f"MoE measured offload gate failed: delta={delta}, read_bytes={io_delta}")
        if metric_tokens != 512 or metric_steps != delta["engine_steps"]:
            raise GateError(f"MoE measured /metrics disagreement: tokens={metric_tokens}, steps={metric_steps}")
        if int(after["moe"]["revision"]["kv_cache_num_blocks"]) != 128:
            raise GateError(f"MoE KV gauge changed: {after['moe']}")
        result["moe_delta"] = delta
        result["metrics_delta"] = {"tokens": metric_tokens, "steps": metric_steps}
    elif config == "gpubpf_host_stride_lfu":
        policy_delta = counter_delta(
            before["policy"], after["policy"],
            ("page_fault_calls", "stride_detections", "prefetches_issued",
             "lfu_activations", "lfu_accesses", "eviction_prepares"),
        )
        eviction_delta = counter_delta(
            before["evictions"], after["evictions"],
            ("evictions", "evicted_bytes", "dropped_evictions"),
        )
        if any(value <= 0 for value in policy_delta.values()):
            raise GateError(f"gpubpf measured hook gate failed: {policy_delta}")
        if eviction_delta["evictions"] <= 0 or eviction_delta["evicted_bytes"] <= 0:
            raise GateError(f"gpubpf completed-eviction gate failed: {eviction_delta}")
        if eviction_delta["dropped_evictions"] != 0:
            raise GateError(f"gpubpf eviction queue dropped events: {eviction_delta}")
        result.update(policy_delta=policy_delta, eviction_delta=eviction_delta)
    return result


def run_measured_config(config: str, run_dir: Path, port: int,
                        prompts: dict[str, Any], prompt_order: list[int],
                        goldens: list[str]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    policy = monitor = server = None
    telemetry = None
    policy_log = monitor_log = server_log = None
    telemetry_log = None
    log_path = run_dir / "server.log"
    try:
        if config == "gpubpf_host_stride_lfu":
            policy, policy_log, policy_ready = start_policy(run_dir)
        else:
            policy_ready = None
        argv, cwd = server_command(config, port, run_dir)
        atomic_write_json(run_dir / "launch.json", {
            "argv": argv, "cwd": str(cwd), "environment": controlled_environment(config),
            "policy_ready": policy_ready,
        })
        server_log = log_path.open("x", buffering=1)
        server = subprocess.Popen(
            argv, cwd=cwd, env=controlled_environment(config), stdout=server_log,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        wait_ready(server, port, log_path, 1800 if config == "moe_infinity_075" else 900)
        identity = check_server_identity(config, port, prompts)
        if config == "gpubpf_host_stride_lfu":
            monitor, monitor_log, monitor_ready = start_eviction_monitor(server.pid, run_dir)
        else:
            monitor_ready = None
        warmup = nonstream_completion(
            config, port, prompts["records"][0]["prompt_token_ids"], run_dir / "warmup.json"
        )
        time.sleep(1.1)
        before = engagement_snapshot(config, port, run_dir, server.pid)
        if any(item["affinity"] != list(range(8)) for item in before["process_io"]["members"]):
            raise GateError("owned measured process tree escaped CPU 0-7")
        telemetry, telemetry_log, telemetry_path = start_gpu_telemetry(run_dir)
        block_start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        requests = []
        for sequence, prompt_number in enumerate(prompt_order, start=1):
            record = prompts["records"][prompt_number]
            requests.append(streamed_completion(
                config, port, record["prompt_token_ids"], goldens[prompt_number - 1],
                run_dir / f"request-{sequence:02d}-prompt-{prompt_number}.sse",
            ))
        block_end_ns = requests[-1]["eof_ns"]
        time.sleep(1.1)
        after = engagement_snapshot(config, port, run_dir, server.pid)
        stop_exact_process(telemetry)
        telemetry_log.close()
        telemetry = None
        telemetry_log = None
        gpu_telemetry = validate_gpu_telemetry(telemetry_path)
        engagement = validate_measured_engagement(config, before, after)
        duration_s = (block_end_ns - block_start_ns) / 1e9
        ttfts = [request["ttft_ms"] for request in requests]
        e2es = [request["e2e_ms"] for request in requests]
        secondary = {
            "median_ttft_ms": statistics.median(ttfts),
            "p95_ttft_ms": float(__import__("numpy").quantile(ttfts, 0.95, method="linear")),
            "max_ttft_ms": max(ttfts), "median_e2e_ms": statistics.median(e2es),
            "max_e2e_ms": max(e2es),
        }
        result = {
            "config": config, "identity": identity, "monitor_ready": monitor_ready,
            "warmup": warmup, "prompt_order": prompt_order, "requests": requests,
            "block_start_ns": block_start_ns, "block_end_ns": block_end_ns,
            "duration_s": duration_s, "verified_output_tokens": 512,
            "output_throughput_tokens_per_s": 512 / duration_s,
            "engagement_before": before, "engagement_after": after,
            "engagement_delta": engagement,
            "gpu_telemetry": gpu_telemetry,
            "secondary": secondary,
        }
        atomic_write_json(run_dir / "result.json", result)
        return result
    finally:
        if server is not None:
            stop_owned_process_group(server)
        if server_log is not None:
            server_log.close()
        if monitor is not None:
            stop_exact_process(monitor)
        if monitor_log is not None:
            monitor_log.close()
        if telemetry is not None:
            stop_exact_process(telemetry)
        if telemetry_log is not None:
            telemetry_log.close()
        if policy is not None:
            stop_exact_process(policy)
        if policy_log is not None:
            policy_log.close()
        if log_path.exists():
            validate_log(log_path)
        if policy is not None:
            inventory = struct_ops_inventory()
            if inventory["maps"] or inventory["links"]:
                raise GateError(f"residual owned struct_ops state: {inventory}")


def analyze_valid_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    if len(blocks) != 5:
        return {"outcome": "inconclusive", "valid_blocks": len(blocks)}
    indices = np.load(HERE / "bootstrap-indices.npy", allow_pickle=False)
    values = {}
    moe = np.array([
        block["results"]["moe_infinity_075"]["output_throughput_tokens_per_s"]
        for block in blocks
    ])
    for config in CONFIGS:
        sample = np.array([
            block["results"][config]["output_throughput_tokens_per_s"] for block in blocks
        ])
        ratios = sample / moe
        estimate = float(np.exp(np.mean(np.log(ratios))))
        boot = np.exp(np.mean(np.log(ratios[indices]), axis=1))
        lower, upper = np.quantile(boot, [0.025, 0.975], method="linear")
        values[config] = {"geometric_mean_ratio_vs_moe": estimate,
                          "ci95": [float(lower), float(upper)]}
    lower, upper = values["gpubpf_host_stride_lfu"]["ci95"]
    interpretation = (
        "higher output-token throughput" if lower > 1 else
        "lower output-token throughput" if upper < 1 else "no resolved difference"
    )
    prompt_medians = np.array([
        np.median([
            gpubpf["ttft_ms"] - moe_request["ttft_ms"]
            for gpubpf, moe_request in zip(
                block["results"]["gpubpf_host_stride_lfu"]["requests"],
                block["results"]["moe_infinity_075"]["requests"],
            )
        ])
        for block in blocks
    ])
    ttft_boot = np.mean(prompt_medians[indices], axis=1)
    ttft_ci = np.quantile(ttft_boot, [0.025, 0.975], method="linear")
    return {
        "outcome": interpretation, "valid_blocks": 5, "ratios": values,
        "ttft_gpubpf_minus_moe_ms": {
            "mean_of_block_medians": float(np.mean(prompt_medians)),
            "ci95": [float(ttft_ci[0]), float(ttft_ci[1])],
        },
    }


def run_full_schedule(output: Path, preflight: Path, port: int) -> dict[str, Any]:
    lease = LeaseSet.acquire()
    try:
        admitted = admission(port, full_hashes=True)
        if not admitted["admitted"]:
            raise GateError("admission refused:\n- " + "\n- ".join(admitted["errors"]))
        preflight_result = json.loads((preflight / "preflight-result.json").read_text())
        if preflight_result.get("status") != "passed":
            raise GateError("correctness preflight is missing or not passed")
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(output / "admission.json", admitted)
        prompts = json.loads(PROMPTS.read_text())
        schedule = json.loads(SCHEDULE.read_text())
        goldens = {
            config: [item["text"] for item in preflight_result["results"][config]["passes"][0]]
            for config in CONFIGS
        }
        attempts = []
        valid_blocks = []
        for scheduled in schedule["attempts"]:
            if len(valid_blocks) == 5:
                break
            attempt_number = int(scheduled["attempt"])
            attempt_dir = output / f"attempt-{attempt_number:02d}.partial"
            attempt_dir.mkdir()
            block: dict[str, Any] = {"attempt": attempt_number, "results": {}, "errors": []}
            for position, config in enumerate(scheduled["configuration_order"]):
                idle = admission(port, full_hashes=False)
                if not idle["admitted"]:
                    block["errors"].append({"config": config, "stage": "admission", "errors": idle["errors"]})
                    continue
                try:
                    block["results"][config] = run_measured_config(
                        config, attempt_dir / config, port, prompts,
                        scheduled["prompt_order"], goldens[config],
                    )
                except Exception as exc:
                    block["errors"].append({"config": config, "stage": "execution", "error": str(exc)})
                if position != len(scheduled["configuration_order"]) - 1:
                    time.sleep(60)
            block["valid"] = not block["errors"] and set(block["results"]) == set(CONFIGS)
            final_dir = output / f"attempt-{attempt_number:02d}"
            atomic_write_json(attempt_dir / "block.json", block)
            os.replace(attempt_dir, final_dir)
            attempts.append(block)
            if block["valid"]:
                valid_blocks.append(block)
        analysis = analyze_valid_blocks(valid_blocks)
        result = {"attempts": attempts, "valid_blocks": len(valid_blocks), "analysis": analysis}
        atomic_write_json(output / "experiment-result.json", result)
        return result
    finally:
        lease.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    admit = subparsers.add_parser("admit", help="read-only, no-launch admission")
    admit.add_argument("--port", type=int, default=18080)
    admit.add_argument("--full-hashes", action="store_true")
    admit.add_argument("--output", type=Path)
    commands = subparsers.add_parser("commands", help="emit frozen command/environment manifest")
    commands.add_argument("--attempt", type=int, choices=range(1, 9), default=1)
    commands.add_argument("--port", type=int, default=18080)
    commands.add_argument("--raw-root", type=Path, default=HERE / "raw")
    commands.add_argument("--output", type=Path)
    preflight = subparsers.add_parser(
        "preflight", help="admitted four-configuration correctness/engagement smoke"
    )
    preflight.add_argument("--port", type=int, default=18080)
    preflight.add_argument(
        "--output", type=Path,
        default=HERE / "raw" / "correctness-preflight",
    )
    run = subparsers.add_parser("run", help="execute the admitted frozen eight-attempt schedule")
    run.add_argument("--port", type=int, default=18080)
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--output", type=Path, default=HERE / "raw" / "full-run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "admit":
        result = admission(args.port, full_hashes=args.full_hashes)
        if args.output:
            atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["admitted"] else 2
    if args.action == "commands":
        result = frozen_commands(args.attempt, args.port, args.raw_root)
        if args.output:
            atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.action == "preflight":
        try:
            result = run_correctness_preflight(args.output.resolve(), args.port)
        except GateError as exc:
            print(str(exc), file=__import__("sys").stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.action == "run":
        try:
            result = run_full_schedule(
                args.output.resolve(), args.preflight.resolve(), args.port
            )
        except GateError as exc:
            print(str(exc), file=__import__("sys").stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
