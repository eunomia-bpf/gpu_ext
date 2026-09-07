# Targeted policy/mechanism exposition repair

The completed LMCache, device-tool and SASS campaigns are not rerun.
This follow-up implements the Aug 14 shepherd's explicit attribution request.

The active abstract calls 0.7--3.3% the overhead of reproduced policies;
the introduction additionally generalizes matching within a few percent to
seven systems. The body instead reports operation-specific costs, storage
variability and finite-buffer/readback boundaries. Neither sentence is a
general overhead bound. The agent-policy paragraph also claims that no
existing system implements these policies and that frameworks cannot express
them, stronger than the scoped expressibility evidence supports.

## Bounded edit plan and ownership

- Local OpenCode Qwen 27B: only `docs/paper/tex-revision/main.tex` abstract
  result sentences and `tex-revision/tex/intro.tex` evaluation summary.
  Preserve the existing paragraph order, historical headline numbers,
  citations and contributions. Explicitly attribute the gains to policy;
  describe matching as scoped policy ports, not universal whole-system
  equivalence or a uniform overhead bound. Keep Chinese comments aligned.
- Local OpenCode GLM: only the mechanism-cost and agent-discovered-policy
  paragraphs of `tex-revision/tex/eval.tex`. Preserve measured numbers;
  distinguish the available cross-layer hooks from inability to implement
  similar algorithms in another system. Incorporate the already measured
  LMCache variance as a counterexample to a universal cost claim.
- Root: review those bounded changes against the existing results, build
  the draft, record the outcome, commit and push. No chapter reorganization,
  new research, numeric substitution or extra experiment is planned.
- Local OpenCode Qwen Next: only `tex-revision/tex/design.tex`'s existing
  storage/CXL and system-wide-policy paragraphs. Connect the measured
  storage-admission adapter to the evaluation without claiming a complete
  storage-residency driver state machine, and add the originally promised
  proposed tenant-attachment/namespacing/budget/permission requirements.
  These are explanations, not new implemented-isolation claims.

The opening's background/problem/system paragraphs keep their current roles.
Only the results-to-contribution link changes: policy gains motivate the
mechanism, while matched ports characterize its workload-dependent cost.
The abstract and introduction must agree with that distinction. This is
an authorized continuous-revision edit, not a new standalone reorganization.

## Published opening repair

Paper `2d0441d` updates the abstract, introduction and conclusion. Historical
1.76x/4.8x/2x results remain; policy gains are separated from scoped
mechanism-cost examples, without a uniform 0.7--3.3% bound or a claim of
whole-system equivalence. Root tightened the local Qwen draft's attribution
and aligned the conclusion. Two pdflatex passes succeed (17 pages;
`/var/tmp/policy-attribution-paper-20260907.log`). Body/design edits remain
with the other two local workers; Qwen Next is automatically retrying a
provider error, not stopped for silence.

The now-free Qwen 27B slot implements a separate result-plot update from
existing Table 1 and five-pair GPU-array data. It writes new figure/source
files under `tex-revision/img/results-raw/revision`, preserving the old PDF.
The figure must retain negative launchlate overhead samples, label the
independent campaigns, and keep final bulk readback outside the prefill
metric. No new GPU cells or performance filtering is requested.

Paper `3cddba8` additionally closes the missing explicit trampoline-scaling
explanation: reducing duplicate work within a warp does not make total
instrumentation cost independent of block count. The paragraph points to
the already measured record-stream/buffer variants and introduces no new
measurement claim. Two pdflatex passes succeed (17 pages,
`/var/tmp/trampoline-scope-paper-20260907.log`). The three local body/design/
figure tasks remain active; no quiet session was stopped or restarted.

## Published body and design repair

Paper `46220f2` integrates local OpenCode GLM's mechanism-cost and
agent-policy paragraphs and the previously omitted Hummingbird background
throughput cost. The LMCache variation is explicitly not a tight estimate
of BPF overhead; five matched budget blocks retain both adverse pairs.
The 51.011% read-p99 reduction accompanies a 15.315% **increase in write
throughput**, not an increase in write latency. No new measurements were run.

Qwen Next's design-only task ended with an actual provider HTTP 524 error
without writing its file. Root completed the two small prose additions:
the measured storage-admission adapter is not a complete storage-residency
state machine or evidence of NVMe-to-GPU P2P, and proposed independent-tenant
requirements are not implemented isolation guarantees. Two pdflatex passes
succeed (17 pages, `/var/tmp/revision-body-scope-20260907.log`).

Qwen 27B's figure session previously reached an actual model `finish=length`
termination and was resumed in the same session, not interrupted for silence.
Its new figure/source/data files are now present. Root's visual inspection
found a clipped axis label and requested a short label plus caption; this
local session remains active. The existing figure and every original result
remain intact. Figure integration is not yet marked complete.
