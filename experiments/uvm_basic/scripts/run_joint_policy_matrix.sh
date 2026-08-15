#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"
APPROVED="${STAGE4_RESULTS}/approved_for_stage4d.txt"
[[ -s "${APPROVED}" ]] || { echo "No Stage 4C-approved policy list." >&2; exit 2; }
mapfile -t APPROVED_POLICIES < <(sort -u "${APPROVED}")
POLICIES=(custom_no_policy prefetch_always_max)
for candidate in eviction_fifo prefetch_always_max_cycle_moe prefetch_cooperative; do
    printf '%s\n' "${APPROVED_POLICIES[@]}" | grep -qx "${candidate}" && POLICIES+=("${candidate}") || true
done

for ratio in 1.05 1.10; do
    for policy in "${POLICIES[@]}"; do
        for index in 1 2 3; do
            bash "$(dirname "$0")/run_stage4_case.sh" --experiment joint_stage4 \
                --policy "${policy}" --kind timing --ratio "${ratio}" --index "${index}" \
                --target-effective "${STAGE4_TARGET_EFFECTIVE}" \
                --safety-headroom "${STAGE4_SAFETY_HEADROOM}"
        done
        bash "$(dirname "$0")/run_stage4_case.sh" --experiment joint_stage4 \
            --policy "${policy}" --kind trace --ratio "${ratio}" --index 1 \
            --target-effective "${STAGE4_TARGET_EFFECTIVE}" \
            --safety-headroom "${STAGE4_SAFETY_HEADROOM}"
        if [[ "${ratio}" == 1.10 ]]; then
            bash "$(dirname "$0")/run_stage4_case.sh" --experiment joint_stage4 \
                --policy "${policy}" --kind nsys --ratio "${ratio}" --index 1 \
                --target-effective "${STAGE4_TARGET_EFFECTIVE}" \
                --safety-headroom "${STAGE4_SAFETY_HEADROOM}"
        fi
    done
done

python3 "${UVM_BASIC_DIR}/analysis/summarize_stage4.py" --results "${STAGE4_RESULTS}"
python3 - "${STAGE4_RESULTS}" "${POLICIES[@]}" <<'PY'
import json,sys
from collections import Counter
from pathlib import Path
root, policies = Path(sys.argv[1]), sys.argv[2:]
counts=Counter(); safe=True
for path in root.glob("joint_stage4/*/*/*/manifest.json"):
    data=json.loads(path.read_text())
    key=(str(data.get("policy")),str(data.get("ratio")),str(data.get("run_kind")))
    if data.get("exit_code")==0 and data.get("correct") and data.get("struct_ops_detached") and data.get("xid_delta")==0:
        counts[key]+=1
    else: safe=False
joint=[p for p in policies if p not in {"custom_no_policy","prefetch_always_max"}]
complete=safe and bool(joint) and all(counts[(p,r,"timing")]>=3 and counts[(p,r,"trace")]>=1
                                     for p in policies for r in ("1.05","1.10"))
data={"evidence_class":"GPU_EXT_STAGE4_JOINT_MATRIX","policies":policies,
      "counts":{"|".join(k):v for k,v in sorted(counts.items())},
      "status":"PASS_STAGE4_JOINT_POLICY_MATRIX" if complete else "PARTIAL_STAGE4_JOINT_POLICY_MATRIX"}
(root/"joint_matrix_status.json").write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
print(json.dumps(data,indent=2,sort_keys=True))
if not complete: raise SystemExit(1)
PY
