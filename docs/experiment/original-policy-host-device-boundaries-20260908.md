# Original policies across host and device

2026-09-08. Local original-paper PDF inspection; no GPU runs or paper edits.

The user asks for faithful original-algorithm host/device BPF ports before
inventing new policies. Having a CUDA workload is not itself device-policy
execution. Conversely, a host-only policy port does not cover all device-side
control logic in the original system.

| Existing system | Original-paper evidence | Current port boundary / next candidate |
| --- | --- | --- |
| XSched | §6.2, Figs. 7–8: host sets a per-queue deactivation flag; GPU-entry guardian checks it, records aborted commands and exits; host clears the flag and relaunches aborted commands. §6.3 has TSG-based and queue/trap-based Level-3 mechanisms. | Current RTX 5090 result is Level-1 HPF with the original XSched actuator. A same-rule host-BPF plus device-BPF guardian is a concrete missing extension. It must preserve queue/command identity and abort/replay behavior; entry abort is not arbitrary running-kernel preemption. sm_120 support remains implementation work. |
| Hummingbird | §4.2 automatically transforms device PTX: split-kernel offset parameters are added to original block IDs. §4.3 host runtime schedules and consolidates split kernels. | Current idle-policy result executes host uBPF, with a shared executor. A device-BPF block-ID mapping paired with the unchanged host splitting rule is a plausible faithful extension, not evidence that it is already implemented. It requires a hook that changes the consumed logical block ID, not merely an observation hook. |
| GPreempt | §3.1 uses driver timeslices. §3.2 also schedules a device preemption kernel that waits until the computing kernel is ready; host notification uses GDRCopy in the original system. | Current host-mapped comparison shares the original blocking-kernel actuator and replaces host decisions. Device BPF could implement the original readiness decision, but persistent waiting should remain in the trusted executor rather than an unbounded BPF loop. The host-mapped/GDRCopy transport distinction remains. |
| POD-Attention | §4.1, Figs. 8–9: a CTA leader reads SM ID, obtains atomic tickets, chooses prefill/decode, and publishes assignment to its CTA. | The existing 250-cell experiment already executes the original selector as actual device BPF. Host launch/parameter preparation is present, but is not automatically host BPF. Any host-BPF extension must identify an actual original host policy to port; do not add a dummy callback to claim both sides. |
| FineMoE / MoE-Infinity / Expert Buffering | FineMoE §4, MoE-Infinity trace/cache design, and Expert Buffering describe GPU expert execution/activation and CPU–GPU expert-cache management. | These establish cross-tier execution, not by themselves that each cache selector originally runs on GPU. Inspect source placement before choosing a same-algorithm split. Preserve original activation counts, history, thresholds and cache semantics rather than introducing a new feedback rule under the reproduction label. |

## Source PDFs and current reports

- XSched: ../paper-material/ref-paper/xsched_osdi25.pdf;
  ../../workloads/xsched/README.md.
- Hummingbird: ../reference/2026-hu-hummingbird-v2.pdf;
  ../../workloads/hummingbird/results-575-20260903.md.
- GPreempt: ../paper-material/ref-paper/gpreempt_atc25.pdf;
  ../../workloads/gpreempt/results-575-host-mapped-20260903.md.
- POD: ../reference/2025-pod-attention.pdf;
  ../../workloads/pod-attention/results-575-20260903.md.
- FineMoE: ../reference/2026-yu-finemoe.pdf;
  ../../workloads/finemoe/results-performance.md.
- MoE-Infinity: ../paper-material/policy-expressibility-papers/20-2024-moe-infinity.pdf.
- Expert Buffering: ../paper/asplos-27-rebuttal/ref/moe-offloading-2303.06182.pdf.

Priority is same-algorithm expressibility and matched original/native versus
BPF performance. New algorithms, if later selected, retain separate labels and
all original numbers. SASS is a possible injection backend, not a prerequisite
for every PTX-capable case. This note selects implementation candidates; it does
not claim new completed host/device ports or performance gains.

## Active implementation assignments

After Qwen Next ended with a terminal HTTP 524 error without a source patch,
root assigned two independent local sessions: Qwen 27B implements the original
XSched Level-2 guardian under workloads/xsched; GLM implements Hummingbird's
original device block-coordinate mapping under workloads/hummingbird. A third
GLM session continues disk UVM in the existing kernel worktree. Qwen Next's
actual service failures explain the temporary use of two GLM sessions.

All three are source/build tasks; GPU runs remain root-coordinated. Preserve
old native/BPF results and frozen builds. No fourth session, new paper, new
adaptive algorithm, or paper edit is part of these assignments. XSched must
preserve the real command-start/abort/replay protocol, not independently abort
arbitrary threads; Hummingbird's BPF coordinates must actually be consumed by
the original kernel. Native controls share their interfaces and actuators.

### XSched source refinement: replay is per CTA in the first command

The original `platforms/cuda/hal/inject/inject.cu` is more precise than Fig. 7:
`check_preempt` lets the CTA leader set a block exit flag and, for the first
preempted command, a block restore flag. A trusted block fence/barrier makes
the CTA exit uniformly. `restore_exec` skips already-completed blocks and
clears flags for blocks being resumed. `InstrumentManager::Reactivate` clears
the header, while `Launch` selects the resume entry for the first preempted
command and the ordinary guardian for subsequent commands in the replay log.

Thus the first command may have partially completed CTAs. A faithful port
must retain its per-CTA restore mask; blindly restarting that whole command
would change semantics. The bounded BPF decision can remain separate from
the trusted barrier/exit glue. Root forwarded this source refinement to the
implementation session; it is not a claim that the new port has run.
