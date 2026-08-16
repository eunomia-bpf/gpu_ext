#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"
TARGET="${STAGE4_TARGET_EFFECTIVE}"
GUARD="${STAGE4_GUARD_DEVICE_BYTES}"

# Preserve the failed mathematical-headroom summary before replacing the canonical output.
if [[ -f "${STAGE4_RESULTS}/calibration_summary.csv" &&
      ! -f "${STAGE4_RESULTS}/legacy_mathematical_headroom_calibration_summary.csv" ]]; then
    cp "${STAGE4_RESULTS}/calibration_summary.csv" \
        "${STAGE4_RESULTS}/legacy_mathematical_headroom_calibration_summary.csv"
fi
if [[ -f "${STAGE4_RESULTS}/calibration_status.json" &&
      ! -f "${STAGE4_RESULTS}/legacy_mathematical_headroom_calibration_status.json" ]]; then
    cp "${STAGE4_RESULTS}/calibration_status.json" \
        "${STAGE4_RESULTS}/legacy_mathematical_headroom_calibration_status.json"
fi

for ratio in 0.95 1.05 1.10; do
    for index in 1 2; do
        bash "$(dirname "$0")/run_stage4_case.sh" --experiment physical_guard_calibration \
            --policy custom_no_policy --kind timing --ratio "${ratio}" --index "${index}" \
            --target-effective "${TARGET}" --guard-bytes "${GUARD}"
    done
    bash "$(dirname "$0")/run_stage4_case.sh" --experiment physical_guard_calibration \
        --policy custom_no_policy --kind trace --ratio "${ratio}" --index 1 \
        --target-effective "${TARGET}" --guard-bytes "${GUARD}"
done

python3 "${UVM_BASIC_DIR}/analysis/summarize_stage4.py" --results "${STAGE4_RESULTS}" \
    --output "${STAGE4_RESULTS}/calibration_summary.csv"
python3 - "${STAGE4_RESULTS}/calibration_summary.csv" \
    "${STAGE4_RESULTS}/calibration_status.json" "${TARGET}" "${GUARD}" <<'PY'
import csv, json, sys
from pathlib import Path
rows = list(csv.DictReader(open(sys.argv[1])))
evidence = {}
def number(value):
    try: return float(value)
    except (TypeError, ValueError): return None
for row in rows:
    if (row.get("experiment") == "physical_guard_calibration"
            and row.get("policy") == "custom_no_policy"
            and row.get("capacity_model") == "PHYSICALLY_RESERVED_GUARD_MODEL"):
        ratio = row["ratio"]
        item = evidence.setdefault(ratio, {})
        item[row["run_kind"]] = {
            "runs": int(row.get("runs") or 0),
            "correctness_pass_rate": number(row.get("correctness_pass_rate")),
            "all_detached": row.get("all_detached") == "True",
            "xid_delta": int(row.get("xid_delta") or 0),
            "actual_ratio": number(row.get("actual_working_set_ratio_mean")),
            "effective_capacity_bytes": number(row.get("effective_capacity_bytes_mean")),
            "capacity_target_relative_error": number(
                row.get("capacity_target_relative_error_max")),
            "working_set_ratio_error": number(row.get("working_set_ratio_error_max")),
            "selected_evictions": int(row.get("selected_eviction_count") or 0),
            "same_block_refaults": number(row.get("same_block_refault_count")),
            "refaulted_bytes": number(row.get("refaulted_bytes")),
        }
trace = {ratio: values.get("trace", {}) for ratio, values in evidence.items()}
counts = {ratio: int(values.get("selected_evictions", 0)) for ratio, values in trace.items()}
refaults = {ratio: values.get("same_block_refaults") for ratio, values in trace.items()}
ratio_checks = {
    ratio: all((kind.get("working_set_ratio_error") is not None
                and kind["working_set_ratio_error"] <= 0.01)
               for kind in values.values())
    for ratio, values in evidence.items()
}
capacity_checks = {
    ratio: all((kind.get("capacity_target_relative_error") is not None
                and kind["capacity_target_relative_error"] <= 0.02)
               for kind in values.values())
    for ratio, values in evidence.items()
}
run_checks = {
    ratio: all(kind.get("correctness_pass_rate") == 1.0
               and kind.get("all_detached")
               and kind.get("xid_delta") == 0 for kind in values.values())
    for ratio, values in evidence.items()
}
checks = {
    "all_ratios_present": all(ratio in evidence for ratio in ("0.95", "1.05", "1.10")),
    "all_timing_and_trace_runs_present": all(
        evidence.get(ratio, {}).get("timing", {}).get("runs") == 2
        and evidence.get(ratio, {}).get("trace", {}).get("runs") == 1
        for ratio in ("0.95", "1.05", "1.10")),
    "capacity_target_within_2_percent": all(capacity_checks.values()),
    "working_set_ratios_within_0_01": all(ratio_checks.values()),
    "correctness_cleanup_and_xid": all(run_checks.values()),
    "095_has_none_or_little_eviction": counts.get("0.95", 10**18) <= max(128, counts.get("1.05", 0) // 20),
    "095_has_none_or_little_refault": (
        refaults.get("0.95") is None and counts.get("0.95", 0) == 0
    ) or (
        refaults.get("0.95") is not None
        and refaults["0.95"] <= max(128, (refaults.get("1.05") or 0) // 20)
    ),
    "105_has_eviction": counts.get("1.05", 0) > 0,
    "110_not_below_105": counts.get("1.10", 0) >= counts.get("1.05", 0) > 0,
    "110_refault_not_below_105": refaults.get("1.05") is not None
        and refaults.get("1.10") is not None
        and refaults["1.10"] >= refaults["1.05"],
}
passed = all(checks.values())
data = {
    "evidence_class": "PHYSICALLY_RESERVED_GUARD_MODEL",
    "target_effective_capacity": sys.argv[3],
    "physical_guard": sys.argv[4],
    "ratios": evidence,
    "selected_evictions": counts,
    "same_block_refaults": refaults,
    "checks": checks,
    "status": "PASS_STAGE4A_PHYSICAL_GUARD_CALIBRATION" if passed
              else "FAILED_STAGE4A_PRESSURE_GATE_WITH_PHYSICAL_GUARD",
}
Path(sys.argv[2]).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
if not passed: raise SystemExit(1)
PY
