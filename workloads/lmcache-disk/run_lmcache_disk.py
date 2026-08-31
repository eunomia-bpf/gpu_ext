#!/usr/bin/env python3
"""Fail-closed LMCache local-NVMe revision experiment.

The primary environment is LMCache 0.5.4 with the newest CUDA-12.9 vLLM
release for which a public cu129 wheel is available (0.27.1). The harness
never kills a process it did not create and never turns an invalid attempt
into a performance observation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
VLLM_WORKLOAD = GPU_EXT / "workloads" / "vllm"
CURRENT_ENV = HERE / "current-venv"
PYTHON = CURRENT_ENV / "bin" / "python"
VLLM = CURRENT_ENV / "bin" / "vllm"
UV = Path("/home/yunwei37/.local/bin/uv")
DATASET = VLLM_WORKLOAD / "datasets" / "ShareGPT_V3_unfiltered_cleaned_split.json"
LMCACHE_REPO = HERE / "deps" / "LMCache-v0.5.4"
ARTIFACTS = HERE / "artifacts-current.json"
SCHEDULE = HERE / "schedule.json"
PROMPTS = HERE / "prompts.json"
PLAN = HERE / "plan-v2.md"
RUNNER = Path(__file__).resolve()

EXPECTED_DRIVER = "610.43.02"
EXPECTED_MOUNT_SOURCE = "/dev/disk/by-uuid/864c5664-999e-43c2-9967-4edaeee79d57"
EXPECTED_VLLM_VERSION = "0.27.1+cu129"
EXPECTED_LMCACHE_VERSION = "0.5.4"
LMCACHE_COMMIT = "3e11b8ed191631e6f098b8038235823f1a410b24"
MODEL_ID = "Qwen/Qwen3-30B-A3B-FP8"
MODEL_REVISION = "d206ba732169f29bb77fbf80fc2c4b81d4d30782"
ORDER_SEED = 2709
BOOTSTRAP_SEED = 2710
TARGET_BLOCKS = 10
MAX_ATTEMPTS = 15
PREFIXES = 8
PREFIX_TOKENS = 1536
CHUNK_TOKENS = 256
CHUNKS_PER_PREFIX = 6
OUTPUT_TOKENS = 16
KV_BYTES_PER_TOKEN = 48 * 2 * 4 * 128 * 2
KV_CHUNK_BYTES = CHUNK_TOKENS * KV_BYTES_PER_TOKEN
EXPECTED_DISK_BYTES = PREFIXES * CHUNKS_PER_PREFIX * KV_CHUNK_BYTES
ROW_STARTS = [0, 173, 509, 997, 1499, 2203, 3109, 4211]
CONFIGS = ("recompute", "lmcache_cpu", "lmcache_disk")
FATAL_LOG_PATTERNS = (
    r"Traceback",
    r"CUDA error",
    r"File not found",
    r"Cannot use O_DIRECT",
    r"Disk space under pressure",
    r"No eviction candidates",
    r"choosing to not store",
    r"fall(?:ing)? back[^\n]*(?:buffered|disk|I/O)",
    r"partial write",
    r"failed for key",
)


class GateError(RuntimeError):
    pass


def file_identity(path: Path) -> dict[str, Any]:
    """Describe a file without reading or fingerprinting its contents."""
    logical = path.expanduser().absolute()
    if not logical.is_file():
        raise GateError(f"required evidence file is missing: {logical}")
    stat = logical.stat()
    return {
        "path": str(logical),
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def run_checked(argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    if proc.returncode:
        raise GateError(f"command failed ({proc.returncode}): {argv!r}\n{proc.stderr[-3000:]}")
    return proc.stdout.strip()


def git_clean_at(repo: Path, commit: str) -> dict[str, str]:
    actual = run_checked(["git", "rev-parse", "HEAD"], repo)
    if actual != commit:
        raise GateError(f"{repo}: expected {commit}, found {actual}")
    dirty = run_checked(["git", "status", "--porcelain", "--untracked-files=no"], repo)
    if dirty:
        raise GateError(f"{repo}: tracked source is dirty: {dirty}")
    return {"path": str(repo), "commit": actual}


def gpu_state() -> dict[str, Any]:
    rows = run_checked([
        "nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]).splitlines()
    if len(rows) != 1:
        raise GateError(f"expected exactly one GPU, found {len(rows)}")
    fields = [x.strip() for x in rows[0].split(",")]
    if len(fields) != 5:
        raise GateError(f"unexpected nvidia-smi row: {rows[0]}")
    apps_raw = run_checked([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    apps = []
    for row in apps_raw.splitlines():
        bits = [x.strip() for x in row.split(",")]
        if len(bits) == 3 and bits[0].isdigit():
            apps.append({"pid": int(bits[0]), "name": bits[1], "memory_mib": int(bits[2])})
    return {
        "index": int(fields[0]), "name": fields[1], "driver": fields[2],
        "memory_used_mib": int(fields[3]), "memory_total_mib": int(fields[4]),
        "compute_apps": apps,
    }


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def resolve_model(local_only: bool = True) -> Path:
    code = (
        "from huggingface_hub import snapshot_download; "
        f"print(snapshot_download({MODEL_ID!r}, revision={MODEL_REVISION!r}, "
        f"local_files_only={local_only!r}))"
    )
    out = run_checked([str(PYTHON), "-c", code])
    path = Path(out.splitlines()[-1]).resolve()
    if path.name != MODEL_REVISION:
        raise GateError(f"resolved model snapshot is not the frozen revision: {path}")
    required = [path / "config.json", path / "model.safetensors.index.json"] + [
        path / f"model-{i:05d}-of-00007.safetensors" for i in range(1, 8)
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise GateError(f"model snapshot incomplete: {missing}")
    return path


def model_artifact_manifest(path: Path) -> list[dict[str, Any]]:
    result = []
    for file_path in sorted(p for p in path.iterdir() if p.is_file()):
        result.append({"name": file_path.name, **file_identity(file_path)})
    return result


def fetch_model() -> Path:
    code = (
        "from huggingface_hub import snapshot_download; "
        f"print(snapshot_download({MODEL_ID!r}, revision={MODEL_REVISION!r}))"
    )
    run_checked([str(PYTHON), "-c", code])
    return resolve_model(local_only=True)


def load_artifacts() -> dict[str, Any]:
    if not ARTIFACTS.is_file():
        raise GateError(f"missing frozen artifact manifest: {ARTIFACTS}")
    return json.loads(ARTIFACTS.read_text())


def verify_python_artifacts() -> dict[str, Any]:
    frozen = load_artifacts()
    env = controlled_environment(cuda_visible_devices="")
    code = r'''
import importlib, importlib.metadata, json, pathlib
names = ["lmcache", "lmcache.lmcache_native", "lmcache.cuda_ops",
         "lmcache.integration.vllm.vllm_v1_adapter",
         "lmcache.integration.vllm.lmcache_connector_v1",
         "lmcache.v1.storage_backend.local_disk_backend", "vllm",
         "vllm.distributed.kv_transfer.kv_connector.factory",
         "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"]
mods={}
for name in names:
    m=importlib.import_module(name); p=pathlib.Path(m.__file__).resolve()
    s=p.stat()
    mods[name]={"path":str(p),"bytes":s.st_size,"device":s.st_dev,"inode":s.st_ino,
                "mtime_ns":s.st_mtime_ns,"ctime_ns":s.st_ctime_ns}
print(json.dumps({"lmcache_version":importlib.metadata.version("lmcache"),
                  "vllm_version":importlib.metadata.version("vllm"),"modules":mods},sort_keys=True))
'''
    out = run_checked([str(PYTHON), "-c", code], env=env)
    observed = json.loads(out.splitlines()[-1])
    if observed["lmcache_version"] != EXPECTED_LMCACHE_VERSION:
        raise GateError(f"unexpected LMCache version: {observed['lmcache_version']}")
    if observed["vllm_version"] != EXPECTED_VLLM_VERSION:
        raise GateError(f"unexpected vLLM version: {observed['vllm_version']}")
    expected_paths = frozen["runtime_import_paths"]
    actual_paths = {name: item["path"] for name, item in observed["modules"].items()}
    if actual_paths != expected_paths:
        raise GateError("runtime import paths differ from artifacts-current.json")
    wheel = HERE / frozen["lmcache_wheel"]["relative_path"]
    vllm_wheel = HERE / frozen["vllm_wheel"]["relative_path"]
    freeze = HERE / frozen["environment_freeze"]["relative_path"]
    for artifact in (wheel, vllm_wheel, freeze):
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise GateError(f"required artifact is missing or empty: {artifact}")
    actual_lines = sorted(
        "lmcache==0.5.4" if line.startswith("lmcache @ ") else line
        for line in run_checked(
            [str(UV), "pip", "freeze", "--python", str(PYTHON)], env=env
        ).splitlines()
    )
    if actual_lines != freeze.read_text().splitlines():
        raise GateError("installed dependency set differs from current-requirements.txt")
    run_checked([str(UV), "pip", "check", "--python", str(PYTHON)], env=env)
    return observed


def controlled_environment(cuda_visible_devices: str = "0") -> dict[str, str]:
    """Return a fixed launch environment; never inherit caller runtime hooks."""
    return {
        "PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin",
        "HOME": "/home/yunwei37",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "CUDA_HOME": "/usr/local/cuda-12.9",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "HF_HOME": "/home/yunwei37/.cache/huggingface",
        "XDG_CACHE_HOME": "/home/yunwei37/.cache",
        "PYTHONNOUSERSITE": "1",
    }


def existing_storage_anchor(target: Path) -> tuple[Path, Path]:
    resolved = target.expanduser().resolve(strict=False)
    anchor = resolved
    while not anchor.exists():
        if anchor.parent == anchor:
            raise GateError(f"no existing ancestor for storage target: {target}")
        anchor = anchor.parent
    if not anchor.is_dir():
        raise GateError(f"storage target ancestor is not a directory: {anchor}")
    return resolved, anchor


def storage_metadata(target: Path) -> dict[str, Any]:
    resolved, anchor = existing_storage_anchor(target)
    stat = os.statvfs(anchor)
    mount = run_checked(["findmnt", "-J", "-T", str(anchor), "-o", "SOURCE,FSTYPE,TARGET"])
    drive = run_checked(["lsblk", "-J", "-o", "NAME,PATH,MODEL,SERIAL,REV,ROTA,TYPE,MOUNTPOINTS"])
    temperatures = []
    for p in sorted(Path("/sys/class/nvme").glob("nvme*/device/hwmon/hwmon*/temp*_input")):
        try:
            temperatures.append({"path": str(p), "millidegrees_c": int(p.read_text().strip())})
        except (OSError, ValueError):
            pass
    return {
        "requested_path": str(target), "resolved_path": str(resolved),
        "existing_anchor": str(anchor),
        "mount": json.loads(mount), "block_devices": json.loads(drive),
        "free_bytes": stat.f_bavail * stat.f_frsize, "block_size": stat.f_bsize,
        "nvme_temperatures": temperatures,
    }


def admission(port: int, require_model: bool = True, storage_path: Path = HERE / "raw") -> dict[str, Any]:
    errors = []
    manifest: dict[str, Any] = {"timestamp_ns": time.time_ns()}
    try:
        manifest["lmcache_source"] = git_clean_at(LMCACHE_REPO, LMCACHE_COMMIT)
        manifest["runtime_imports"] = verify_python_artifacts()
    except Exception as exc:
        errors.append(f"software artifacts: {exc}")
    try:
        dataset = json.loads(DATASET.read_text())
        if not isinstance(dataset, list) or len(dataset) <= max(ROW_STARTS):
            raise GateError("dataset structure cannot satisfy the frozen row starts")
        load_prompts(PROMPTS)
        schedule = json.loads(SCHEDULE.read_text())
        validate_schedule(schedule)
        manifest["workload_artifacts"] = {
            "dataset": file_identity(DATASET),
            "prompts": file_identity(PROMPTS),
            "schedule": file_identity(SCHEDULE),
        }
    except Exception as exc:
        errors.append(f"public workload artifacts: {exc}")
    try:
        state = gpu_state()
        manifest["gpu"] = state
        if state["driver"] != EXPECTED_DRIVER:
            errors.append(f"driver: expected {EXPECTED_DRIVER}, found {state['driver']}")
        if state["compute_apps"]:
            errors.append(f"GPU has foreign compute processes: {state['compute_apps']}")
        if state["memory_used_mib"] > 256:
            errors.append(f"GPU residual memory is {state['memory_used_mib']} MiB (>256 MiB)")
    except Exception as exc:
        errors.append(f"GPU state: {exc}")
    try:
        storage = storage_metadata(storage_path)
        manifest["storage"] = storage
        entry = storage["mount"]["filesystems"][0]
        if Path(entry.get("source", "")).resolve() != Path(EXPECTED_MOUNT_SOURCE).resolve() or entry.get("fstype") != "ext4":
            errors.append(f"requested output/cache path is not {EXPECTED_MOUNT_SOURCE} ext4: {entry}")
        if storage["free_bytes"] < 100 * 1024**3:
            errors.append(f"less than 100 GiB free on cache filesystem: {storage['free_bytes']}")
    except Exception as exc:
        errors.append(f"cache filesystem: {exc}")
    if not port_is_free(port):
        errors.append(f"port {port} already has a listener")
    if not Path("/usr/bin/strace").is_file():
        errors.append("/usr/bin/strace is required for O_DIRECT preflight")
    if require_model:
        try:
            manifest["model_path"] = str(resolve_model(local_only=True))
            manifest["model_revision"] = MODEL_REVISION
            manifest["model_artifacts"] = model_artifact_manifest(Path(manifest["model_path"]))
        except Exception as exc:
            errors.append(f"model: {exc}")
    if errors:
        raise GateError("admission refused:\n- " + "\n- ".join(errors))
    return manifest


def tokenizer_path() -> Path:
    code = (
        "from huggingface_hub import snapshot_download; "
        f"print(snapshot_download({MODEL_ID!r}, revision={MODEL_REVISION!r}, "
        "allow_patterns=['*.json','*.txt','LICENSE','README.md']))"
    )
    return Path(run_checked([str(PYTHON), "-c", code]).splitlines()[-1]).resolve()


def prepare_prompts(output: Path) -> dict[str, Any]:
    tok_path = tokenizer_path()
    code = f'''
import json
from pathlib import Path
from transformers import AutoTokenizer
dataset = json.loads(Path({str(DATASET)!r}).read_text())
tok = AutoTokenizer.from_pretrained({str(tok_path)!r}, local_files_only=True)
prefixes = []
for index, start in enumerate({ROW_STARTS!r}):
    pieces, row_ids, cursor = [], [], start
    while True:
        row = dataset[cursor % len(dataset)]
        row_ids.append(row.get("id", str(cursor)))
        for turn in row.get("conversations", []):
            pieces.append(f"[{{turn.get('from', 'unknown')}}] {{turn.get('value', '')}}")
        ids = tok.encode("\\n".join(pieces), add_special_tokens=False)
        if len(ids) >= {PREFIX_TOKENS}:
            prefix_ids = ids[:{PREFIX_TOKENS}]
            text = tok.decode(prefix_ids, skip_special_tokens=False)
            if tok.encode(text, add_special_tokens=False) != prefix_ids:
                raise RuntimeError("tokenizer decode/encode is not stable")
            cold_suffix = f"\\n\\nCold suffix {{index}}: summarize the material in one sentence."
            warm_suffix = f"\\n\\nWarm suffix {{index}}: state the central topic in one sentence."
            cold_text = text + cold_suffix
            warm_text = text + warm_suffix
            cold_ids = tok.encode(cold_text, add_special_tokens=False)
            warm_ids = tok.encode(warm_text, add_special_tokens=False)
            lcp = 0
            for a, b in zip(cold_ids, warm_ids):
                if a != b: break
                lcp += 1
            expected_hit = lcp - lcp % {CHUNK_TOKENS}
            if expected_hit != {PREFIX_TOKENS}:
                raise RuntimeError(f"expected 1536 aligned hit tokens, got {{expected_hit}}")
            prefixes.append({{
                "index": index, "start_row": start, "row_ids": row_ids,
                "prefix_text": text, "cold_suffix": cold_suffix, "warm_suffix": warm_suffix,
                "prefix_token_ids": prefix_ids, "cold_token_ids": cold_ids,
                "warm_token_ids": warm_ids, "prefix_tokens": len(prefix_ids),
                "cold_tokens": len(cold_ids), "warm_tokens": len(warm_ids),
                "lcp_tokens": lcp, "expected_hit_tokens": expected_hit,
                "expected_store_tokens": len(cold_ids) - len(cold_ids) % {CHUNK_TOKENS},
            }})
            break
        cursor += 1
result = {{"schema": 3, "model": {MODEL_ID!r}, "model_revision": {MODEL_REVISION!r},
          "dataset": "workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
          "prefix_tokens": {PREFIX_TOKENS}, "chunk_tokens": {CHUNK_TOKENS},
          "row_starts": {ROW_STARTS!r}, "prefixes": prefixes}}
print(json.dumps(result, ensure_ascii=False))
'''
    result = json.loads(run_checked([str(PYTHON), "-c", code]).splitlines()[-1])
    atomic_write_json(output, result)
    return result


def load_prompts(path: Path) -> dict[str, Any]:
    prompts = json.loads(path.read_text())
    if (
        prompts.get("schema") != 3
        or prompts.get("model") != MODEL_ID
        or prompts.get("model_revision") != MODEL_REVISION
        or prompts.get("dataset")
        != "workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
        or prompts.get("prefix_tokens") != PREFIX_TOKENS
        or prompts.get("chunk_tokens") != CHUNK_TOKENS
        or prompts.get("row_starts") != ROW_STARTS
        or len(prompts.get("prefixes", [])) != PREFIXES
    ):
        raise GateError("prompt artifact does not match protocol")
    for i, item in enumerate(prompts["prefixes"]):
        prefix_ids = item.get("prefix_token_ids")
        cold_ids = item.get("cold_token_ids")
        warm_ids = item.get("warm_token_ids")
        if (
            item.get("index") != i
            or item.get("start_row") != ROW_STARTS[i]
            or not isinstance(item.get("prefix_text"), str)
            or not isinstance(prefix_ids, list)
            or not isinstance(cold_ids, list)
            or not isinstance(warm_ids, list)
            or len(prefix_ids) != PREFIX_TOKENS
            or item.get("prefix_tokens") != len(prefix_ids)
            or item.get("cold_tokens") != len(cold_ids)
            or item.get("warm_tokens") != len(warm_ids)
            or item.get("expected_hit_tokens") != PREFIX_TOKENS
        ):
            raise GateError(f"invalid expected hit metadata for prefix {i}")
        lcp = next((j for j, pair in enumerate(zip(cold_ids, warm_ids)) if pair[0] != pair[1]),
                   min(len(cold_ids), len(warm_ids)))
        if item.get("lcp_tokens") != lcp or lcp - lcp % CHUNK_TOKENS != PREFIX_TOKENS:
            raise GateError(f"invalid exact common-prefix structure for prefix {i}")
        if item.get("expected_store_tokens") != PREFIX_TOKENS:
            raise GateError(f"invalid expected store metadata for prefix {i}")
    return prompts


def server_environment(config: str, cache_dir: Path) -> dict[str, str]:
    env = controlled_environment()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
               VLLM_USE_DEEP_GEMM="0")
    if config == "lmcache_cpu":
        env.update(LMCACHE_CHUNK_SIZE=str(CHUNK_TOKENS), LMCACHE_LOCAL_CPU="True",
                   LMCACHE_MAX_LOCAL_CPU_SIZE="8.0", LMCACHE_SAVE_UNFULL_CHUNK="False",
                   LMCACHE_USE_LAYERWISE="False")
    elif config == "lmcache_disk":
        env.update(LMCACHE_CHUNK_SIZE=str(CHUNK_TOKENS), LMCACHE_LOCAL_CPU="False",
                   LMCACHE_MAX_LOCAL_CPU_SIZE="2.0", LMCACHE_LOCAL_DISK="file://" + str(cache_dir),
                   LMCACHE_MAX_LOCAL_DISK_SIZE="16.0", LMCACHE_SAVE_UNFULL_CHUNK="False",
                   LMCACHE_USE_LAYERWISE="False", LMCACHE_EXTRA_CONFIG=canonical({"use_odirect": True}))
    return env


def server_argv(config: str, model_path: Path, port: int | str) -> list[str]:
    argv = [str(VLLM), "serve", str(model_path), "--served-model-name", MODEL_ID,
            "--enforce-eager", "--max-model-len", "4096", "--gpu-memory-utilization", "0.98",
            "--max-num-seqs", "1", "--no-enable-prefix-caching", "--port", str(port)]
    if config != "recompute":
        argv.extend(["--kv-transfer-config", canonical(
            {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        )])
    return argv


def start_server(config: str, model_path: Path, cache_dir: Path, port: int, log_path: Path, trace_dir: Path | None = None):
    argv = server_argv(config, model_path, port)
    launch = list(argv)
    if trace_dir is not None:
        trace_dir = trace_dir.resolve()
        trace_dir.mkdir(parents=True, exist_ok=False)
        launch = ["/usr/bin/strace", "-ff", "-qq", "-s", "4096", "-e", "trace=open,openat",
                  "-o", str(trace_dir / "open.trace")] + launch
    log_file = log_path.open("x")
    proc = subprocess.Popen(launch, cwd=VLLM_WORKLOAD, env=server_environment(config, cache_dir),
                            stdout=log_file, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    return proc, log_file, argv, launch


def wait_ready(proc: subprocess.Popen, port: int, log_path: Path, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-5000:] if log_path.exists() else ""
            raise GateError(f"server exited {proc.returncode} before ready\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise GateError("server readiness timeout")


def stop_owned_server(proc: subprocess.Popen, log_file) -> None:
    try:
        if proc.poll() is None:
            pgid = os.getpgid(proc.pid)
            if pgid != proc.pid:
                raise GateError(f"refusing cleanup: process group changed ({pgid} != {proc.pid})")
            os.killpg(pgid, signal.SIGINT)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=20)
    finally:
        log_file.close()


def wait_gpu_idle(timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = gpu_state()
        if not state["compute_apps"] and state["memory_used_mib"] <= 256:
            return
        time.sleep(2)
    raise GateError(f"GPU did not return to idle after owned server exit: {gpu_state()}")


def streamed_completion(port: int, token_ids: list[int], request_id: str) -> dict[str, Any]:
    payload = json.dumps({"model": MODEL_ID, "prompt": token_ids, "max_tokens": OUTPUT_TOKENS,
                          "temperature": 0, "seed": 0, "ignore_eos": True, "stream": True,
                          "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json", "X-Request-Id": request_id}, method="POST")
    start = time.perf_counter_ns()
    first = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    status = None
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[6:])
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    piece = choice.get("text") or ""
                    if piece and first is None:
                        first = time.perf_counter_ns()
                    text_parts.append(piece)
    except urllib.error.HTTPError as exc:
        raise GateError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    end = time.perf_counter_ns()
    output = "".join(text_parts)
    if status != 200 or first is None or not output:
        raise GateError(f"invalid response: status={status}, first={first}, bytes={len(output)}")
    if int(usage.get("completion_tokens", -1)) != OUTPUT_TOKENS:
        raise GateError(f"expected {OUTPUT_TOKENS} output tokens, got {usage}")
    if int(usage.get("prompt_tokens", -1)) != len(token_ids):
        raise GateError(f"server prompt-token count differs from frozen token array: {usage}")
    return {"request_header": request_id, "engine_request_id": f"cmpl-{request_id}-0",
            "input_tokens": len(token_ids),
            "status": status, "ttft_ms": (first - start) / 1e6, "e2e_ms": (end - start) / 1e6,
            "usage": usage, "text": output}


def request_log_values(log: str, request_id: str) -> dict[str, Any]:
    q = re.escape(request_id)
    hits = [None if value == "None" else int(value) for value in re.findall(
        rf"Reqid:\s*{q},[^\n]*LMCache hit tokens:\s*(None|\d+)", log)]
    stores = [(int(a), int(b)) for a, b in re.findall(
        rf"\[req_id={q}\] Stored\s+(\d+) out of total\s+(\d+) tokens", log)]
    retrieved = [int(x) for x in re.findall(rf"\[req_id={q}\] Retrieved\s+(\d+) out of", log)]
    return {"hits": hits, "stores": stores, "retrieved": retrieved}


def disk_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("*.pt"))


def sync_and_verify_disk(cache_dir: Path, expected_files: int, timeout: float = 180) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = None
    stable = 0
    while time.monotonic() < deadline:
        files = disk_files(cache_dir)
        stats = [p.stat() for p in files]
        state = (len(files), tuple((s.st_size, s.st_blocks * 512) for s in stats))
        if len(files) > expected_files:
            raise GateError(f"unexpected extra disk chunks: {len(files)} > {expected_files}")
        if len(files) == expected_files and all(
            size == KV_CHUNK_BYTES and allocated >= size for size, allocated in state[1]
        ):
            stable = stable + 1 if state == last else 0
            if stable >= 2:
                for p in files:
                    fd = os.open(p, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                dfd = os.open(cache_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
                final = disk_files(cache_dir)
                final_stats = [p.stat() for p in final]
                sizes = [s.st_size for s in final_stats]
                if (
                    len(final) != expected_files
                    or any(x != KV_CHUNK_BYTES for x in sizes)
                    or any(s.st_blocks * 512 < s.st_size for s in final_stats)
                ):
                    raise GateError("disk chunk set changed during durability sync")
                return {"files": len(final), "bytes": sum(sizes), "per_file_bytes": sorted(set(sizes)),
                        "durability": "fsync(each file) + fsync(directory)"}
        else:
            stable = 0
        last = state
        time.sleep(1)
    raise GateError(f"disk persistence timeout: expected {expected_files} x {KV_CHUNK_BYTES} bytes, got {last}")


def wait_for_cold_store(config: str, log_path: Path, cache_dir: Path, engine_request_id: str,
                        prefix_index: int, expected_store_tokens: int) -> dict[str, Any]:
    if config == "recompute":
        return {"files": 0, "bytes": 0, "durability": "not applicable"}
    deadline = time.monotonic() + 180
    values: dict[str, Any] = {}
    while time.monotonic() < deadline:
        log = log_path.read_text(errors="replace")
        values = request_log_values(log, engine_request_id)
        if values["stores"] and all(a == expected_store_tokens for a, _ in values["stores"]):
            break
        time.sleep(1)
    else:
        raise GateError(f"request-scoped store evidence missing for {engine_request_id}: {values}")
    if not values["hits"] or any(x not in (0, None) for x in values["hits"]):
        raise GateError(f"cold request had external hits: {engine_request_id} {values['hits']}")
    if config == "lmcache_disk":
        state = sync_and_verify_disk(cache_dir, (prefix_index + 1) * CHUNKS_PER_PREFIX)
    else:
        state = {"files": 0, "bytes": 0, "durability": "synchronous LocalCPUBackend insertion"}
    return {**state, "request_log": values}


def validate_odirect(trace_dir: Path, cache_dir: Path) -> dict[str, Any]:
    relevant = []
    for path in sorted(trace_dir.glob("open.trace*")):
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            match = re.search(r'open(?:at)?\([^\n]*?"([^"]+\.pt)"[^\n]*\)\s+=\s+(-?\d+)', line)
            if match:
                relevant.append({"file": path.name, "line": line_no, "text": line,
                                 "cache_path": match.group(1), "result_fd": int(match.group(2))})
    if not relevant:
        raise GateError("strace observed no .pt opens")
    buffered = [x for x in relevant if "O_DIRECT" not in x["text"]]
    if buffered:
        raise GateError(f".pt files opened without O_DIRECT: {buffered[:8]}")
    failed = [x for x in relevant if x["result_fd"] < 0]
    if failed:
        raise GateError(f"unsuccessful O_DIRECT .pt opens: {failed[:8]}")
    cache_root = cache_dir.resolve()
    escaped = [x for x in relevant
               if not Path(x["cache_path"]).resolve(strict=False).is_relative_to(cache_root)]
    if escaped:
        raise GateError(f"O_DIRECT .pt path escaped current cache directory {cache_root}: {escaped[:8]}")
    writes = [x for x in relevant if "O_WRONLY" in x["text"] or "O_RDWR" in x["text"]]
    reads = [x for x in relevant if "O_RDONLY" in x["text"] or "O_RDWR" in x["text"]]
    expected = PREFIXES * CHUNKS_PER_PREFIX
    write_paths = {str(Path(x["cache_path"]).resolve(strict=False)) for x in writes}
    read_paths = {str(Path(x["cache_path"]).resolve(strict=False)) for x in reads}
    if len(write_paths) != expected or len(read_paths) != expected or write_paths != read_paths:
        raise GateError(
            "incomplete O_DIRECT path coverage: "
            f"unique_writes={len(write_paths)}, unique_reads={len(read_paths)}, "
            f"same_set={write_paths == read_paths}, expected={expected}"
        )
    evidence = {"all_pt_opens_have_odirect": True, "pt_open_count": len(relevant),
                "write_open_count": len(writes), "read_open_count": len(reads),
                "all_pt_opens_succeeded": True, "all_paths_under_cache_dir": True,
                "cache_dir": str(cache_root),
                "unique_write_paths": sorted(write_paths), "unique_read_paths": sorted(read_paths),
                "expected_unique_paths_each_direction": expected,
                "trace_files": {p.name: file_identity(p) for p in sorted(trace_dir.glob("open.trace*"))}}
    atomic_write_json(trace_dir / "odirect-evidence.json", evidence)
    return evidence


def validate_log(config: str, log: str, observations: list[dict[str, Any]], cache_dir: Path) -> dict[str, Any]:
    fatal = [pattern for pattern in FATAL_LOG_PATTERNS if re.search(pattern, log, re.I)]
    if fatal:
        raise GateError(f"fatal/fallback evidence in server log: {fatal}")
    prefix_rate = [float(x) for x in re.findall(r"Prefix cache hit rate:\s*([0-9.]+)%", log)]
    if any(x != 0 for x in prefix_rate):
        raise GateError(f"native vLLM prefix cache engaged: {prefix_rate}")
    request_evidence = {}
    if config == "recompute":
        if "LMCache initialized" in log or "LMCache hit tokens:" in log:
            raise GateError("recompute control unexpectedly engaged LMCache")
    else:
        init = re.findall(r"LMCache initialized[^\n]*version\s+([^, ]+), vllm version\s+([^, ]+)", log)
        if not init or any(a != EXPECTED_LMCACHE_VERSION or b != EXPECTED_VLLM_VERSION for a, b in init):
            raise GateError(f"LMCache/vLLM initialization evidence mismatch: {init}")
        for item in observations:
            expected = item["expected_hit_tokens"]
            cold_id = item["cold"]["engine_request_id"]
            warm_id = item["warm"]["engine_request_id"]
            cold = request_log_values(log, cold_id)
            warm = request_log_values(log, warm_id)
            if not cold["hits"] or any(x not in (0, None) for x in cold["hits"]):
                raise GateError(f"cold hit gate failed for {cold_id}: {cold}")
            if (not warm["hits"] or any(value != expected for value in warm["hits"])
                    or not warm["retrieved"] or any(value != expected for value in warm["retrieved"])):
                raise GateError(f"exact warm hit/retrieval gate failed for {warm_id}: expected {expected}, got {warm}")
            request_evidence[warm_id] = warm
        if config == "lmcache_disk":
            for needle in ("LocalDiskBackend", "Using O_DIRECT for disk I/O: True"):
                if needle not in log:
                    raise GateError(f"disk engagement evidence missing: {needle}")
            if re.search(r"['\"]local_cpu['\"]\s*:\s*True", log):
                raise GateError("disk-only mode unexpectedly enabled local CPU retention")
            final = sync_and_verify_disk(cache_dir, PREFIXES * CHUNKS_PER_PREFIX)
            if final["bytes"] != EXPECTED_DISK_BYTES:
                raise GateError(f"disk footprint mismatch: {final}")
    return {"request_evidence": request_evidence, "native_prefix_rates": prefix_rate}


def run_config(config: str, run_dir: Path, prompts: dict[str, Any], port: int,
               model_path: Path, trace: bool = False) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = (run_dir / "cache").resolve()
    cache_dir.mkdir()
    log_path = run_dir / "server.log"
    trace_dir = run_dir / "strace" if trace else None
    proc, log_file, argv, launch = start_server(config, model_path, cache_dir, port, log_path, trace_dir)
    observations: list[dict[str, Any]] = []
    try:
        wait_ready(proc, port, log_path)
        for item in prompts["prefixes"]:
            index = item["index"]
            cold = streamed_completion(port, item["cold_token_ids"], f"lmc-p{index}-cold")
            store = wait_for_cold_store(config, log_path, cache_dir, cold["engine_request_id"], index,
                                        item["expected_store_tokens"])
            observations.append({"prefix_index": index,
                                 "expected_hit_tokens": item["expected_hit_tokens"], "cold": cold,
                                 "store_state": store})
        warm_start = time.perf_counter_ns()
        for item, observation in zip(prompts["prefixes"], observations, strict=True):
            index = item["index"]
            observation["warm"] = streamed_completion(port, item["warm_token_ids"], f"lmc-p{index}-warm")
        warm_end = time.perf_counter_ns()
    finally:
        try:
            stop_owned_server(proc, log_file)
        finally:
            wait_gpu_idle()
    log = log_path.read_text(errors="replace")
    engagement = validate_log(config, log, observations, cache_dir)
    odirect = validate_odirect(trace_dir, cache_dir) if trace_dir is not None else None
    warm_elapsed_s = (warm_end - warm_start) / 1e9
    warm_tokens = sum(int(x["warm"]["usage"]["completion_tokens"]) for x in observations)
    result = {
        "schema": 2, "config": config, "command": argv, "launch_command": launch,
        "environment": server_environment(config, cache_dir),
        "warm_phase": {"sequential": True, "requests": PREFIXES, "output_tokens": warm_tokens,
                       "elapsed_s": warm_elapsed_s, "requests_per_s": PREFIXES / warm_elapsed_s,
                       "output_tokens_per_s": warm_tokens / warm_elapsed_s,
                       "excludes": ["server startup", "cold population", "persistence barriers", "shutdown"]},
        "observations": observations, "engagement": engagement, "odirect": odirect,
        "cache_footprint": {"files": len(disk_files(cache_dir)),
                            "bytes": sum(p.stat().st_size for p in disk_files(cache_dir))},
        "server_log": file_identity(log_path),
    }
    atomic_write_json(run_dir / "result.json", result)
    return result


def output_texts(result: dict[str, Any]) -> dict[str, str]:
    return {f"{item['prefix_index']}:{phase}": item[phase]["text"]
            for item in result["observations"] for phase in ("cold", "warm")}


def validate_schedule(schedule: dict[str, Any]) -> None:
    attempts = schedule.get("attempts", [])
    if (
        schedule.get("order_seed") != ORDER_SEED
        or schedule.get("target_valid_blocks") != TARGET_BLOCKS
        or schedule.get("maximum_attempts") != MAX_ATTEMPTS
        or len(attempts) != MAX_ATTEMPTS
        or [item.get("attempt") for item in attempts] != list(range(MAX_ATTEMPTS))
        or any(sorted(item.get("order", [])) != sorted(CONFIGS) for item in attempts)
    ):
        raise GateError("precomputed schedule does not match protocol")


def protocol_manifest(admit: dict[str, Any]) -> dict[str, Any]:
    schedule = json.loads(SCHEDULE.read_text())
    validate_schedule(schedule)
    placeholder_model = Path("{MODEL_PATH}")
    placeholder_cache = Path("{CACHE_DIR}")
    return {
        **admit,
        "protocol": {"target_valid_blocks": TARGET_BLOCKS, "maximum_attempts": MAX_ATTEMPTS,
                     "prefixes": PREFIXES, "prefix_tokens": PREFIX_TOKENS, "chunk_tokens": CHUNK_TOKENS,
                     "output_tokens": OUTPUT_TOKENS, "order_seed": ORDER_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
                     "configs": CONFIGS, "disk_expected_files": PREFIXES * CHUNKS_PER_PREFIX,
                     "disk_expected_bytes": EXPECTED_DISK_BYTES,
                     "kv_footprint_formula": "48 layers * 2(K,V) * 4 KV heads * 128 head_dim * 2 bf16 bytes",
                     "cpu_tier_limit_gib": 8, "disk_staging_pool_gib": 2, "disk_tier_limit_gib": 16},
        "artifact_files": {
            "artifacts": file_identity(ARTIFACTS),
            "schedule": file_identity(SCHEDULE),
            "prompts": file_identity(PROMPTS),
        },
        "protocol_sources": {"runner": file_identity(RUNNER), "plan": file_identity(PLAN)},
        "server_configs": {
            config: {"argv": server_argv(config, placeholder_model, "{PORT}"),
                     "environment": server_environment(config, placeholder_cache)}
            for config in CONFIGS
        },
        "preflight_trace_prefix": ["/usr/bin/strace", "-ff", "-qq", "-s", "4096",
                                   "-e", "trace=open,openat", "-o", "{TRACE_PREFIX}"],
    }


def comparable_manifest(value: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    result.pop("timestamp_ns", None)
    result.get("storage", {}).pop("free_bytes", None)
    result.get("storage", {}).pop("nvme_temperatures", None)
    return result


def create_or_verify_root(output: Path, manifest: dict[str, Any], resume: bool) -> None:
    if output.exists():
        if not resume:
            raise GateError(f"output exists; pass --resume after inspecting it: {output}")
        old_path = output / "manifest.json"
        if not old_path.is_file():
            raise GateError("resume root has no manifest.json")
        if canonical(comparable_manifest(json.loads(old_path.read_text()))) != canonical(comparable_manifest(manifest)):
            raise GateError("resume manifest mismatch")
    else:
        output.mkdir(parents=True)
        atomic_write_json(output / "manifest.json", manifest)


def run_preflight(output: Path, prompts_path: Path, port: int) -> None:
    admit = admission(port, require_model=True, storage_path=output)
    manifest = protocol_manifest(admit)
    if output.exists():
        raise GateError(f"preflight output already exists: {output}")
    output.mkdir(parents=True)
    atomic_write_json(output / "manifest.json", manifest)
    result = run_config("lmcache_disk", output / "lmcache_disk", load_prompts(prompts_path), port,
                        Path(admit["model_path"]), trace=True)
    atomic_write_json(output / "PREFLIGHT-PASS.json",
                      {"manifest": file_identity(output / "manifest.json"),
                       "files": {
                           "lmcache_disk/result.json": file_identity(output / "lmcache_disk" / "result.json"),
                           "lmcache_disk/strace/odirect-evidence.json": file_identity(
                               output / "lmcache_disk" / "strace" / "odirect-evidence.json"
                           ),
                       },
                       "odirect": result["odirect"]})


def run_smoke(output: Path, prompts_path: Path, port: int) -> None:
    admit = admission(port, require_model=True, storage_path=output)
    manifest = protocol_manifest(admit)
    if output.exists():
        raise GateError(f"smoke output already exists: {output}")
    output.mkdir(parents=True)
    atomic_write_json(output / "manifest.json", manifest)
    prompts = load_prompts(prompts_path)
    reference = None
    for config in CONFIGS:
        result = run_config(config, output / config, prompts, port, Path(admit["model_path"]))
        texts = output_texts(result)
        if reference is None:
            reference = texts
        elif texts != reference:
            raise GateError(f"exact output text differs in isolated smoke: {config}")
    atomic_write_json(output / "reference-output-texts.json", reference)
    atomic_write_json(
        output / "SMOKE-PASS.json",
        {
            "manifest": file_identity(output / "manifest.json"),
            "files": {
                "reference-output-texts.json": file_identity(output / "reference-output-texts.json"),
                **{f"{config}/result.json": file_identity(output / config / "result.json")
                   for config in CONFIGS},
            },
        },
    )


def verify_gate_artifact(path: Path, marker: str, current: dict[str, Any]) -> None:
    marker_path = path / marker
    manifest_path = path / "manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise GateError(f"missing gate artifact {marker} under {path}")
    marker_data = json.loads(marker_path.read_text())
    if marker_data.get("manifest") != file_identity(manifest_path):
        raise GateError(f"gate marker manifest identity mismatch: {path}")
    required_files = (
        {"lmcache_disk/result.json", "lmcache_disk/strace/odirect-evidence.json"}
        if marker == "PREFLIGHT-PASS.json"
        else {"reference-output-texts.json", *(f"{config}/result.json" for config in CONFIGS)}
    )
    files = marker_data.get("files")
    if not isinstance(files, dict) or set(files) != required_files:
        raise GateError(f"gate marker required file set mismatch: {path}: {files}")
    for relative, expected in files.items():
        target = path / relative
        if not target.is_file() or file_identity(target) != expected:
            raise GateError(f"gate marker file identity mismatch: {target}")
    if marker == "PREFLIGHT-PASS.json":
        evidence_path = path / "lmcache_disk" / "strace" / "odirect-evidence.json"
        evidence = json.loads(evidence_path.read_text())
        if marker_data.get("odirect") != evidence:
            raise GateError(f"preflight marker O_DIRECT evidence mismatch: {path}")
        trace_dir = evidence_path.parent
        trace_files = evidence.get("trace_files")
        actual_traces = {p.name: file_identity(p) for p in sorted(trace_dir.glob("open.trace*"))}
        if not isinstance(trace_files, dict) or trace_files != actual_traces:
            raise GateError(f"preflight raw trace file identity mismatch: {trace_dir}")
    old = json.loads(manifest_path.read_text())
    for key in ("protocol", "protocol_sources", "server_configs", "preflight_trace_prefix",
                "artifact_files",
                "model_revision", "model_artifacts", "runtime_imports"):
        if old.get(key) != current.get(key):
            raise GateError(f"gate artifact mismatch for {key}: {path}")


def validate_complete_attempt(
    attempt_dir: Path,
    scheduled: dict[str, Any],
    reference_output_texts: dict[str, str],
) -> None:
    marker_path = attempt_dir / "COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    if marker.get("attempt") != scheduled["attempt"] or marker.get("order") != scheduled["order"]:
        raise GateError(f"complete-attempt schedule mismatch: {attempt_dir}")
    files = marker.get("result_files")
    if not isinstance(files, dict) or set(files) != set(CONFIGS):
        raise GateError(f"complete-attempt result file set is invalid: {attempt_dir}")
    for config in CONFIGS:
        result_path = attempt_dir / config / "result.json"
        if not result_path.is_file() or file_identity(result_path) != files[config]:
            raise GateError(f"complete-attempt result file identity mismatch: {result_path}")
        result = json.loads(result_path.read_text())
        if result.get("config") != config or output_texts(result) != reference_output_texts:
            raise GateError(f"complete-attempt semantic mismatch: {result_path}")


def run_experiment(output: Path, prompts_path: Path, port: int, preflight: Path, smoke: Path,
                   approval: Path, resume: bool) -> None:
    approval_text = approval.read_text(errors="replace") if approval.is_file() else ""
    if "FINAL DECISION: APPROVE" not in approval_text:
        raise GateError(f"full run lacks final independent approval marker in {approval}")
    admit = admission(port, require_model=True, storage_path=output)
    manifest = protocol_manifest(admit)
    verify_gate_artifact(preflight, "PREFLIGHT-PASS.json", manifest)
    verify_gate_artifact(smoke, "SMOKE-PASS.json", manifest)
    create_or_verify_root(output, manifest, resume)
    prompts = load_prompts(prompts_path)
    reference = json.loads((smoke / "reference-output-texts.json").read_text())
    schedule = json.loads(SCHEDULE.read_text())["attempts"]
    scheduled_by_name = {f"attempt-{item['attempt']:02d}": item for item in schedule}
    valid = []
    for attempt_dir in sorted(p for p in output.glob("attempt-*") if (p / "COMPLETE.json").is_file()):
        if attempt_dir.name not in scheduled_by_name:
            raise GateError(f"complete attempt is outside the frozen schedule: {attempt_dir}")
        validate_complete_attempt(attempt_dir, scheduled_by_name[attempt_dir.name], reference)
        valid.append(attempt_dir)
    if len(valid) > TARGET_BLOCKS:
        raise GateError(f"output already has too many complete attempts: {len(valid)}")
    run_complete = output / "RUN-COMPLETE.json"
    if run_complete.is_file():
        marker = json.loads(run_complete.read_text())
        names = sorted(p.name for p in valid)
        if len(valid) == TARGET_BLOCKS and marker.get("valid_attempts") == names:
            return
        raise GateError("existing RUN-COMPLETE.json does not match validated attempts")
    for item in schedule:
        if len(valid) >= TARGET_BLOCKS:
            break
        attempt = item["attempt"]
        attempt_dir = output / f"attempt-{attempt:02d}"
        complete = attempt_dir / "COMPLETE.json"
        invalid = attempt_dir / "INVALID.json"
        if complete.is_file() or invalid.is_file():
            continue
        if attempt_dir.exists():
            atomic_write_json(invalid, {"attempt": attempt, "technical_invalid": True,
                                        "reason": "partial attempt found during exact resume"})
            continue
        attempt_dir.mkdir()
        atomic_write_json(attempt_dir / "order.json", item)
        try:
            for config in item["order"]:
                result = run_config(config, attempt_dir / config, prompts, port, Path(admit["model_path"]))
                if output_texts(result) != reference:
                    raise GateError(f"exact output text drift in attempt {attempt}, config {config}")
            atomic_write_json(complete, {"attempt": attempt, "order": item["order"],
                                         "result_files": {
                                             config: file_identity(attempt_dir / config / "result.json")
                                             for config in CONFIGS
                                         }})
            valid.append(attempt_dir)
        except Exception as exc:
            atomic_write_json(invalid, {"attempt": attempt, "technical_invalid": True,
                                        "reason": f"{type(exc).__name__}: {exc}", "decision_blind": True})
    if len(valid) != TARGET_BLOCKS:
        raise GateError(f"attempt cap reached with {len(valid)} valid blocks; required {TARGET_BLOCKS}")
    atomic_write_json(output / "RUN-COMPLETE.json",
                      {"valid_attempts": sorted(p.name for p in valid), "valid_blocks": len(valid),
                       "maximum_attempts": MAX_ATTEMPTS})


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_ci(values: list[float], seed: int, draws: int = 10000) -> tuple[float, float, float]:
    import random
    rng = random.Random(seed)
    estimates = [statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)]
    return statistics.median(values), percentile(estimates, 0.025), percentile(estimates, 0.975)


def analyze(root: Path) -> dict[str, Any]:
    attempts = sorted(p for p in root.glob("attempt-*") if (p / "COMPLETE.json").is_file())
    if len(attempts) != TARGET_BLOCKS:
        raise GateError(f"expected {TARGET_BLOCKS} complete attempts, found {len(attempts)}")
    run_marker_path = root / "RUN-COMPLETE.json"
    if not run_marker_path.is_file():
        raise GateError("analysis requires RUN-COMPLETE.json")
    run_marker = json.loads(run_marker_path.read_text())
    if run_marker.get("valid_attempts") != [p.name for p in attempts]:
        raise GateError("RUN-COMPLETE.json does not match complete attempts")
    rows = []
    for attempt in attempts:
        complete = json.loads((attempt / "COMPLETE.json").read_text())
        row: dict[str, Any] = {"attempt": attempt.name}
        for config in CONFIGS:
            result_path = attempt / config / "result.json"
            if file_identity(result_path) != complete.get("result_files", {}).get(config):
                raise GateError(f"analysis result file identity mismatch: {result_path}")
            result = json.loads(result_path.read_text())
            if result.get("schema") != 2 or result.get("config") != config:
                raise GateError(f"analysis result schema/config mismatch: {result_path}")
            warm = [float(x["warm"]["ttft_ms"]) for x in result["observations"]]
            row[config] = {"warm_ttft_median_ms": statistics.median(warm),
                           "warm_ttft_p95_ms": percentile(warm, 0.95), "warm_ttft_max_ms": max(warm),
                           "sequential_warm_requests_per_s": float(result["warm_phase"]["requests_per_s"])}
        rows.append(row)
    disk_recompute_latency = [r["lmcache_disk"]["warm_ttft_median_ms"] - r["recompute"]["warm_ttft_median_ms"] for r in rows]
    disk_cpu_latency = [r["lmcache_disk"]["warm_ttft_median_ms"] - r["lmcache_cpu"]["warm_ttft_median_ms"] for r in rows]
    disk_recompute_rate = [r["lmcache_disk"]["sequential_warm_requests_per_s"]
                           / r["recompute"]["sequential_warm_requests_per_s"] - 1 for r in rows]
    latency_ci = bootstrap_ci(disk_recompute_latency, BOOTSTRAP_SEED)
    cpu_ci = bootstrap_ci(disk_cpu_latency, BOOTSTRAP_SEED + 1)
    rate_ci = bootstrap_ci(disk_recompute_rate, BOOTSTRAP_SEED + 2)
    if latency_ci[2] < 0 and rate_ci[1] >= -0.05:
        decision = "beneficial"
    elif latency_ci[2] < 0 and rate_ci[1] < -0.05:
        decision = "latency-throughput tradeoff"
    elif latency_ci[1] >= 0:
        decision = "not beneficial"
    else:
        decision = "inconclusive"
    result = {
        "complete_blocks": len(attempts),
        "estimand_scope": "repeated complete-block variability for this fixed model, eight prefixes, and SSD",
        "rows": rows,
        "paired_percentile_bootstrap": {"draws": 10000,
            "disk_minus_recompute_warm_ttft_ms": {"median": latency_ci[0], "ci95": latency_ci[1:]},
            "disk_minus_cpu_warm_ttft_ms": {"median": cpu_ci[0], "ci95": cpu_ci[1:]},
            "disk_vs_recompute_sequential_warm_rate_relative": {"median": rate_ci[0], "ci95": rate_ci[1:]}},
        "decision": decision,
        "claim_scope": "standalone LMCache storage-tier characterization; not pooled with submitted gpubpf results",
    }
    atomic_write_json(root / "analysis.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_admit = sub.add_parser("admission")
    p_admit.add_argument("--port", type=int, default=18080)
    p_admit.add_argument("--storage-root", type=Path, default=HERE / "raw")
    p_prepare = sub.add_parser("prepare-prompts")
    p_prepare.add_argument("--output", type=Path, default=PROMPTS)
    sub.add_parser("fetch-model")
    for name in ("preflight", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--prompts", type=Path, default=PROMPTS)
        p.add_argument("--port", type=int, default=18080)
    p_run = sub.add_parser("run")
    p_run.add_argument("--output", type=Path, required=True)
    p_run.add_argument("--prompts", type=Path, default=PROMPTS)
    p_run.add_argument("--port", type=int, default=18080)
    p_run.add_argument("--preflight", type=Path, required=True)
    p_run.add_argument("--smoke", type=Path, required=True)
    p_run.add_argument("--approval", type=Path, default=HERE / "plan-review-v2.md")
    p_run.add_argument("--resume", action="store_true")
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "admission":
            print(json.dumps(admission(args.port, storage_path=args.storage_root), indent=2))
        elif args.command == "prepare-prompts":
            print(json.dumps(prepare_prompts(args.output), indent=2, ensure_ascii=False))
        elif args.command == "fetch-model":
            print(fetch_model())
        elif args.command == "preflight":
            run_preflight(args.output, args.prompts, args.port)
        elif args.command == "smoke":
            run_smoke(args.output, args.prompts, args.port)
        elif args.command == "run":
            run_experiment(args.output, args.prompts, args.port, args.preflight, args.smoke, args.approval, args.resume)
        elif args.command == "analyze":
            print(json.dumps(analyze(args.root), indent=2))
    except GateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
