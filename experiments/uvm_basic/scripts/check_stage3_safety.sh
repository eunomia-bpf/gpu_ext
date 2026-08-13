#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"
OUTPUT="${RESULTS_DIR}/stage3/safety.json"
mkdir -p "$(dirname "${OUTPUT}")"

python3 - "${UVM_BASIC_DIR}" "${OUTPUT}" <<'PY'
import json, os, re, stat, sys
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
safe = root / "scripts" / "SAFE_STAGE3_COMMANDS.sh"
runtime = [path for path in (root / "scripts").glob("*stage3*.sh")
           if path != safe and path.name != "check_stage3_safety.sh"]
runtime += [root / "scripts" / name for name in (
    "run_trace_semantics.sh", "run_trace_overhead.sh", "run_cpu_first_touch_diagnosis.sh",
    "run_array_migration_diagnosis.sh", "run_oversub_sweep.sh", "check_oversub_safety.sh")]
runtime = sorted({path for path in runtime if path.exists()})
forbidden_runtime = re.compile(r"\b(?:sudo|rmmod|insmod|modprobe)\b|make\s+modules_install|pkill\s+-f")
runtime_hits = []
for path in runtime:
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        code = line.split("#", 1)[0]
        if forbidden_runtime.search(code): runtime_hits.append(f"{path.name}:{number}:{code.strip()}")
safe_text = safe.read_text(errors="replace")
safe_forbidden = []
for pattern in (r"make\s+modules_install", r"/lib/modules/.+\bcp\b", r"pkill\s+-f"):
    if re.search(pattern, safe_text): safe_forbidden.append(pattern)
checks = {
    "root_commands_only_in_safe_script": not runtime_hits,
    "no_modules_install": "modules_install" not in safe_text,
    "no_system_module_copy": not safe_forbidden,
    "safe_script_non_executable": not bool(safe.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
    "no_fuzzy_pkill": "pkill -f" not in safe_text,
    "no_default_move_head_policy": "eviction_lfu" not in (root / "scripts" / "run_oversub_sweep.sh").read_text(),
    "absolute_ratio_ceiling_present": "1.25" in (root / "scripts" / "run_oversub_sweep.sh").read_text(),
}
data = {"evidence_class": "STATIC_STAGE3_SAFETY", "checks": checks,
        "runtime_hits": runtime_hits, "safe_forbidden": safe_forbidden,
        "root_operation_executed": False,
        "status": "PASS_STAGE3_STATIC_SAFETY" if all(checks.values()) else "FAIL_STAGE3_STATIC_SAFETY"}
output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
if data["status"].startswith("FAIL"): raise SystemExit(1)
PY
