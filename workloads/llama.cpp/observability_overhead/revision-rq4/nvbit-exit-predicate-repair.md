# NVBit exit predicate repair — 2026-09-03

Status: source corrected and sm_120 adapter rebuilt. No post-repair GPU result
yet. This is a bug in our custom adapter, not evidence of an NVBit core defect.

The adapter inserted `observe_exit` before each SASS `EXIT`, without passing
the instruction's guard predicate. That does not distinguish a taken exit
from a predicated-off exit instruction. The official NVBit 1.8
`tools/instr_count/instr_count.cu:188` passes
`nvbit_add_call_arg_guard_pred_val` when excluding predicated-off instructions;
its injected function uses that predicate when counting active threads. The
API is declared in `core/nvbit.h:548` in the retained local release.

Our adapter now passes the same guard as the first argument and returns before
either event emission or histogram increment when it is false. The existing
PTX retprobe explicitly preserves the predicate of `ret`/`exit`. No workload,
target symbol, histogram capacity, output oracle or validity check was weakened.

The old preflight selected 220 launches and instrumented three SASS exits.
It reported 901,120 NVBit histogram events versus 720,896 BPF events. The BPF
raw histogram has counts 220 at IDs 0–1023, 44 at 1024–2047 and 22 at
2048–22527. These are observations, not proof that either old total was the
correct number of actual exits. The guard omission is a concrete candidate
for NVBit overcounting; the causal contribution requires the new same-workload
diagnostic. We retract any presumption that the difference establishes BPF
loss alone. The separate BPF event-channel capacity/drain defects remain.

Build: CUDA 12.9, g++-13, sm_120; both edited translation units and the shared
library rebuild successfully on CPUs 18–19, without a GPU workload. The
[complete build log](../../../../docs/experiment/revision-safety/table1-nvbit-exit-guard-575-01/build.log)
is retained. The previous binary is recoverable at
`/tmp/gpubpf-nvbit-pre-guard-erQCD4/observability.so`; earlier raw results are
unchanged and their source version remains in Git.

Next: a fresh untimed baseline/NVBit-histogram/BPF-histogram diagnostic with
exact application output, full array readback, matched counts, current binary
inventory and owned cleanup. Equal aggregate counts alone will not establish
all per-launch/coordinate semantics or finish the seven-arm Table 1 study.
