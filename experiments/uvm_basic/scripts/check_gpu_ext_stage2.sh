#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"
OUT="${RESULTS_DIR}/gpu_ext_stage2_preflight.json"
DOC="${UVM_BASIC_DIR}/docs/STAGE2_PREFLIGHT.md"
CUSTOM_UVM="${GPU_EXT_CUSTOM_UVM:-/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
CUSTOM_SYMVERS="${GPU_EXT_CUSTOM_SYMVERS:-/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module/kernel-open/Module.symvers}"

python3 "${UVM_BASIC_DIR}/analysis/audit_uvm_only_compat.py" \
    --experiment-dir "${UVM_BASIC_DIR}" --custom-uvm "${CUSTOM_UVM}" \
    --custom-symvers "${CUSTOM_SYMVERS}" >/dev/null

python3 - "${GPU_EXT_ROOT}" "${CUSTOM_UVM}" "${OUT}" "${DOC}" \
    "${RESULTS_DIR}/uvm_only_compatibility.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
custom = Path(sys.argv[2])
output = Path(sys.argv[3])
document = Path(sys.argv[4])
compatibility = json.loads(Path(sys.argv[5]).read_text())
names = [
    "prefetch_trace",
    "chunk_trace",
    "prefetch_none",
    "prefetch_always_max",
    "prefetch_adaptive_sequential",
]

def command(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.returncode, result.stdout.strip()

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

binaries = {}
for name in names:
    path = root / "extension" / name
    file_rc, file_text = command("file", str(path)) if path.exists() else (127, "MISSING")
    ldd_rc, ldd_text = command("ldd", str(path)) if path.exists() else (127, "MISSING")
    binaries[name] = {
        "path": str(path),
        "present": path.is_file(),
        "executable": os.access(path, os.X_OK),
        "sha256": sha256(path),
        "file_exit": file_rc,
        "file": file_text,
        "ldd_exit": ldd_rc,
        "ldd": ldd_text.splitlines(),
        "dependencies_complete": ldd_rc == 0 and "not found" not in ldd_text,
    }

_, loaded_path = command("modinfo", "-n", "nvidia_uvm")
_, loaded_version = command("modinfo", "-F", "version", "nvidia_uvm")
loaded_srcversion_path = Path("/sys/module/nvidia_uvm/srcversion")
loaded_srcversion = (loaded_srcversion_path.read_text().strip()
                     if loaded_srcversion_path.exists() else "UNAVAILABLE")
_, loaded_depends = command("modinfo", "-F", "depends", "nvidia_uvm")
if custom.is_file():
    _, custom_version = command("modinfo", "-F", "version", str(custom))
    _, custom_vermagic = command("modinfo", "-F", "vermagic", str(custom))
    _, custom_srcversion = command("modinfo", "-F", "srcversion", str(custom))
    _, custom_depends = command("modinfo", "-F", "depends", str(custom))
else:
    custom_version = custom_vermagic = custom_srcversion = custom_depends = "UNAVAILABLE"

try:
    kallsyms = Path("/proc/kallsyms").read_text(errors="replace")
except OSError:
    kallsyms = ""
hook_names = sorted({fields[2] for line in kallsyms.splitlines()
                     if len(fields := line.split()) >= 3
                     and ("uvm_bpf_call_gpu_page_prefetch" in fields[2]
                          or "gpu_mem_ops" in fields[2])})
hook_visible = "uvm_bpf_call_gpu_page_prefetch" in hook_names
all_binaries = all(item["present"] and item["executable"] and item["ldd_exit"] == 0
                   and item["dependencies_complete"]
                   for item in binaries.values())
identity_match = (
    custom.is_file()
    and custom_version == loaded_version
    and custom_vermagic.startswith(os.uname().release + " ")
)
custom_binary_loaded = bool(custom_srcversion and custom_srcversion == loaded_srcversion and hook_visible)
distribution_loaded = bool(Path("/sys/module/nvidia_uvm").exists() and not custom_binary_loaded)
if not all_binaries:
    status = "BLOCKED_EXTENSION_BINARIES"
elif custom_binary_loaded:
    status = "READY_FOR_ROOT_TRACE_ATTACH"
else:
    status = "READY_FOR_MANUAL_GPU_EXT_STAGE2"

data = {
    "evidence_class": "GPU_EXT_STAGE2_PREFLIGHT",
    "status": status,
    "binaries": binaries,
    "all_binaries_ready": all_binaries,
    "loaded_module": {
        "path": loaded_path,
        "version": loaded_version,
        "srcversion": loaded_srcversion,
        "depends": loaded_depends,
        "distribution_module_loaded": distribution_loaded,
    },
    "custom_module": {
        "path": str(custom),
        "present": custom.is_file(),
        "sha256": sha256(custom),
        "version": custom_version,
        "vermagic": custom_vermagic,
        "srcversion": custom_srcversion,
        "depends": custom_depends,
        "version_kernel_match": identity_match,
    },
    "gpu_ext_prefetch_hook_visible": hook_visible,
    "custom_hook_symbols_visible": hook_visible,
    "custom_hook_symbols": hook_names,
    "custom_binary_loaded": custom_binary_loaded,
    "bpf_attached": False,
    "root_operation_executed": False,
    "uvm_only_compatibility": {
        "status": compatibility["status"],
        "recommendation": compatibility["recommendation"],
        "evidence": str(Path(sys.argv[5])),
    },
}
output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
lines = [
    "# gpu_ext Stage 2 Preflight", "",
    f"Status: `{status}`.", "",
    f"- Distribution `nvidia_uvm` loaded: `{distribution_loaded}`",
    f"- Custom `nvidia_uvm` loaded: `{custom_binary_loaded}`",
    f"- Custom hook symbols visible: `{hook_visible}`",
    f"- Loaded module candidate: `{loaded_path}`",
    f"- Loaded runtime srcversion: `{loaded_srcversion}`",
    f"- Custom module: `{custom}`",
    f"- Custom version/kernel match: `{identity_match}`",
    f"- Loaded/custom dependency: `{loaded_depends}` / `{custom_depends}`",
    f"- All five extension binaries ready: `{all_binaries}`", "",
    f"- UVM-only static ABI status: `{compatibility['status']}`",
    f"- Module-switch recommendation: `{compatibility['recommendation']}`", "",
    "No root operation, module switch, BPF attach, or policy execution was performed by this preflight.", "",
    "The privileged `bpftool prog show` and `bpftool struct_ops list` checks are provided only in `scripts/SAFE_GPU_EXT_COMMANDS.sh`.", "",
]
document.write_text("\n".join(lines))
print(json.dumps({key: data[key] for key in (
    "status", "all_binaries_ready", "gpu_ext_prefetch_hook_visible",
    "custom_binary_loaded", "bpf_attached", "root_operation_executed")}, indent=2))
PY
