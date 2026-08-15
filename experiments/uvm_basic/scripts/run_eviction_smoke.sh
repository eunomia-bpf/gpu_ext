#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"
AUDIT="${STAGE4_RESULTS}/eviction_policy_audit.json"
python3 - "${STAGE4_RESULTS}/prefetch_matrix_status.json" <<'PY'
import json,sys
try: data=json.load(open(sys.argv[1]))
except (OSError,json.JSONDecodeError): raise SystemExit("Stage 4B matrix evidence is missing")
if data.get("status") != "PASS_STAGE4_REDUCED_CAPACITY_PREFETCH_MATRIX":
    raise SystemExit("Stage 4B matrix has not passed")
PY
python3 "${UVM_BASIC_DIR}/analysis/audit_eviction_policies.py" \
    --extension-dir "${EXTENSION_DIR}" --json "${AUDIT}" \
    --markdown "${UVM_BASIC_DIR}/docs/EVICTION_POLICY_SAFETY_AUDIT_STAGE4.md"

mapfile -t POLICIES < <(python3 - "${AUDIT}" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1]))["policies"]:
    if item["suitable_for_initial_pressure_test"]:
        print(item["policy"])
PY
)
: >"${STAGE4_RESULTS}/approved_for_stage4d.txt"

for policy in "${POLICIES[@]}"; do
    [[ -n "${policy}" ]] || continue
    # The first smoke is deliberately non-oversubscribed and small.
    STAGE3_RESULTS="${STAGE4_RESULTS}" GPU_EXT_RUN_EVIDENCE_CLASS=GPU_EXT_STAGE4_SMOKE \
        bash "$(dirname "$0")/run_stage3_case.sh" --experiment trace_semantics \
        --policy "${policy}" --kind timing --ratio 64M --index 1
    STAGE3_RESULTS="${STAGE4_RESULTS}" GPU_EXT_RUN_EVIDENCE_CLASS=GPU_EXT_STAGE4_SMOKE \
        bash "$(dirname "$0")/run_stage3_case.sh" --experiment trace_semantics \
        --policy "${policy}" --kind trace --ratio 64M --index 1
    bash "$(dirname "$0")/run_stage4_case.sh" --experiment joint_stage4 \
        --policy "${policy}" --kind timing --ratio 0.95 --index 1 \
        --target-effective "${STAGE4_TARGET_EFFECTIVE}" \
        --safety-headroom "${STAGE4_SAFETY_HEADROOM}"
    printf '%s\n' "${policy}" >>"${STAGE4_RESULTS}/approved_for_stage4d.txt"
done
