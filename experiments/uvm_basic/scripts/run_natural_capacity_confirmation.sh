#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"
[[ -s "${STAGE4_RESULTS}/approved_for_stage4d.txt" ]] || {
    echo "Stage 4D approval evidence is missing." >&2
    exit 2
}
python3 - "${STAGE4_RESULTS}/joint_matrix_status.json" <<'PY'
import json,sys
try: data=json.load(open(sys.argv[1]))
except (OSError,json.JSONDecodeError): raise SystemExit("Stage 4D result is missing")
if data.get("status") != "PASS_STAGE4_JOINT_POLICY_MATRIX":
    raise SystemExit("Stage 4D has not passed")
PY
if [[ -n "${STAGE4_NATURAL_POLICIES:-}" ]]; then
    IFS=, read -r -a POLICIES <<<"${STAGE4_NATURAL_POLICIES}"
else
    POLICIES=(custom_no_policy prefetch_always_max)
    while IFS= read -r policy; do
        [[ -n "${policy}" ]] && POLICIES+=("${policy}")
    done < <(sort -u "${STAGE4_RESULTS}/approved_for_stage4d.txt" | head -n2)
fi
for policy in "${POLICIES[@]}"; do
    [[ "${policy}" != prefetch_none ]] || { echo "prefetch_none is forbidden in natural confirmation." >&2; exit 2; }
    for index in 1 2; do
        bash "$(dirname "$0")/run_stage4_case.sh" --experiment natural_stage4 \
            --policy "${policy}" --kind timing --ratio 1.05 --index "${index}"
    done
    bash "$(dirname "$0")/run_stage4_case.sh" --experiment natural_stage4 \
        --policy "${policy}" --kind trace --ratio 1.05 --index 1
done
