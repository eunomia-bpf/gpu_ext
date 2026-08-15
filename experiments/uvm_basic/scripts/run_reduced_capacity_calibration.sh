#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"
TARGET="${STAGE4_TARGET_EFFECTIVE}"

for ratio in 0.95 1.05 1.10; do
    for index in 1 2; do
        bash "$(dirname "$0")/run_stage4_case.sh" --experiment reduced_capacity \
            --policy custom_no_policy --kind timing --ratio "${ratio}" --index "${index}" \
            --target-effective "${TARGET}" --safety-headroom "${STAGE4_SAFETY_HEADROOM}"
    done
    bash "$(dirname "$0")/run_stage4_case.sh" --experiment reduced_capacity \
        --policy custom_no_policy --kind trace --ratio "${ratio}" --index 1 \
        --target-effective "${TARGET}" --safety-headroom "${STAGE4_SAFETY_HEADROOM}"
done

python3 "${UVM_BASIC_DIR}/analysis/summarize_stage4.py" --results "${STAGE4_RESULTS}" \
    --output "${STAGE4_RESULTS}/calibration_summary.csv"
python3 - "${STAGE4_RESULTS}/calibration_summary.csv" \
    "${STAGE4_RESULTS}/calibration_status.json" "${TARGET}" <<'PY'
import csv, json, sys
from pathlib import Path
rows = list(csv.DictReader(open(sys.argv[1])))
counts = {}
for row in rows:
    if (row.get("experiment") == "reduced_capacity"
            and row.get("policy") == "custom_no_policy"
            and row.get("run_kind") == "trace"):
        counts[row["ratio"]] = int(row.get("selected_eviction_count") or 0)
checks = {
    "all_ratios_present": all(ratio in counts for ratio in ("0.95", "1.05", "1.10")),
    "095_has_none_or_little_eviction": counts.get("0.95", 10**18) <= max(128, counts.get("1.05", 0) // 20),
    "105_has_eviction": counts.get("1.05", 0) > 0,
    "110_exceeds_105": counts.get("1.10", 0) > counts.get("1.05", 0),
}
data = {"evidence_class": "GPU_EXT_STAGE4_CALIBRATION", "target_effective_capacity": sys.argv[3],
        "selected_evictions": counts,
        "checks": checks, "status": "PASS_STAGE4_REDUCED_CAPACITY_CALIBRATION"
        if all(checks.values()) else "FAILED_STAGE4_REDUCED_CAPACITY_CALIBRATION"}
Path(sys.argv[2]).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
if not all(checks.values()): raise SystemExit(1)
PY
