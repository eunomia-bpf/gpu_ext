#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"
TARGET="${STAGE4_TARGET_EFFECTIVE}"
FALLBACK_USED="${STAGE4_FALLBACK_USED:-0}"
POLICIES=(custom_no_policy prefetch_none prefetch_always_max prefetch_adaptive_sequential)
python3 - "${STAGE4_RESULTS}/calibration_status.json" "${TARGET}" <<'PY'
import json, sys
try: data = json.load(open(sys.argv[1]))
except (OSError, json.JSONDecodeError): raise SystemExit("Stage 4A calibration evidence is missing")
if data.get("status") != "PASS_STAGE4A_PHYSICAL_GUARD_CALIBRATION":
    raise SystemExit("Stage 4A calibration has not passed")
if data.get("target_effective_capacity") != sys.argv[2]:
    raise SystemExit("Stage 4A calibration target does not match this matrix")
PY

run_case() {
    local policy="$1" kind="$2" ratio="$3" index="$4"
    bash "$(dirname "$0")/run_stage4_case.sh" --experiment prefetch_matrix_stage4 \
        --policy "${policy}" --kind "${kind}" --ratio "${ratio}" --index "${index}" \
        --target-effective "${TARGET}" --guard-bytes "${STAGE4_GUARD_DEVICE_BYTES}"
}

for ratio in 0.95 1.05 1.10; do
    for policy in "${POLICIES[@]}"; do
        for index in 1 2 3; do
            set +e
            run_case "${policy}" timing "${ratio}" "${index}"
            rc=$?
            set -e
            if ((rc != 0)); then
                latest_exit="$(stage4_latest_case_exit "${STAGE4_RESULTS}/prefetch_matrix_stage4/${policy}/${ratio}" 2>/dev/null || true)"
                if [[ "${policy}" == prefetch_none && "${TARGET}" == 8G && "${FALLBACK_USED}" == 0 &&
                      "${latest_exit}" == 124 ]]; then
                    printf 'prefetch_none timed out at 8G; stop and schedule one 6G maintenance window.\n' \
                        >"${STAGE4_RESULTS}/PREFETCH_NONE_8G_TIMEOUT.txt"
                    exit "${rc}"
                fi
                if [[ "${policy}" == prefetch_none && "${TARGET}" == 6G && "${latest_exit}" == 124 ]]; then
                    printf 'PREFETCH_NONE_UNBOUNDED_EVEN_AT_REDUCED_CAPACITY\n' \
                        >"${STAGE4_RESULTS}/PREFETCH_NONE_UNBOUNDED_EVEN_AT_REDUCED_CAPACITY"
                fi
                exit "${rc}"
            fi
        done
        run_case "${policy}" trace "${ratio}" 1
        if [[ "${ratio}" == 1.10 ]]; then
            run_case "${policy}" nsys "${ratio}" 1
        fi
    done
done

python3 "${UVM_BASIC_DIR}/analysis/summarize_stage4.py" --results "${STAGE4_RESULTS}"
python3 - "${STAGE4_RESULTS}" "${TARGET}" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root, target = Path(sys.argv[1]), sys.argv[2]
counts = Counter()
safe = True
for path in root.glob("prefetch_matrix_stage4/*/*/*/manifest.json"):
    data = json.loads(path.read_text())
    if data.get("evidence_class") != "GPU_EXT_STAGE4_RUN": continue
    rows = []
    for line in (path.parent / "program.jsonl").read_text(errors="replace").splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    cap = next((row for row in rows if row.get("phase") == "capacity_manifest"), {})
    expected = (8 if target == "8G" else 6) << 30
    if abs(int(cap.get("effective_gpu_capacity_bytes", 0)) - expected) > (256 << 20): continue
    key = (str(data.get("policy")), str(data.get("ratio")), str(data.get("run_kind")))
    if data.get("exit_code") == 0 and data.get("correct") and data.get("struct_ops_detached") and data.get("xid_delta") == 0:
        counts[key] += 1
    else: safe = False
policies = ("custom_no_policy", "prefetch_none", "prefetch_always_max", "prefetch_adaptive_sequential")
ratios = ("0.95", "1.05", "1.10")
complete = safe and all(counts[(p, r, "timing")] >= 3 and counts[(p, r, "trace")] >= 1
                        for p in policies for r in ratios)
data = {"evidence_class": "GPU_EXT_STAGE4_PREFETCH_MATRIX", "target_effective_capacity": target,
        "counts": {"|".join(k): v for k, v in sorted(counts.items())},
        "status": "PASS_STAGE4_REDUCED_CAPACITY_PREFETCH_MATRIX" if complete else "PARTIAL_STAGE4_PREFETCH_MATRIX"}
(root / "prefetch_matrix_status.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
if not complete: raise SystemExit(1)
PY
