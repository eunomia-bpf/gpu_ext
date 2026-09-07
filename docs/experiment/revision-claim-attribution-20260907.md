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

## Published device figure

Paper `a60ffb0` integrates the local Qwen 27B plot, source and full plotted
data. The old P40 points and all ten original RTX 5090 pairs remain alongside
the independent five-pair GPU-array result. The axis preserves negative
launchlate samples. Root inspected the compiled page 15: labels are visible,
and the caption separates each campaign's baseline, min--max range and the
final 10.38 ms readback outside prefill. The old PDF remains unchanged.
Two pdflatex passes succeed (17 pages, 4,538,515 bytes;
`/var/tmp/revision-device-array-figure-20260907.log`). The conclusion and
references now begin on page 15, so this build is not a final page-budget
approval. A 1.848 pt final-page vertical overflow also remains recorded in
the build log, not treated as a missing-performance-data gate.

Qwen's caption/reproduction note is finishing separately. Local OpenCode GLM
session `ses_f84aa39ecffeeFydNz5D0teAMc` (runner handle 37822) is preparing
one unsubmitted revision-response draft from the existing commitments and
results. It owns only `asplos-27-rebuttal/revision-response-draft-20260907.md`.
No external conference action or new GPU measurement is authorized by this
draft task. Original historical agent transcripts remain missing.

Paper `d3d8fa8` publishes the finished figure caption/reproduction note.
Each measurement uses its own campaign baseline, including the two separate
campaigns in the RTX 5090 panel. Regeneration writes to a fresh temporary
directory, so the documented command does not require deleting tracked
figures. Root made only a small wording correction after local Qwen's edits.

## Current prose task

Local Qwen Next session `ses_f84a453d1ffe8uXt0OcXXxqYUb` (runner 66942)
owns only the matched-policy subsection in `tex-revision/tex/eval.tex`,
excluding its figures/captions and the following RQ2 subsection. Its task is
to remove repeated explanation while preserving every number, comparison,
citation and necessary scope qualifier. Root will inspect and build actual
edits before publication; assigning the task is not evidence of page reduction.
GLM's unsubmitted response draft remains a separate live task. Qwen 27B's
already-published figure task is retrying a provider response; no session was
stopped for silence and no fourth local session was opened.

The figure runner 61325 subsequently exited normally with a final completion
message. Its freed slot now runs Qwen 27B session
`ses_f849e2c96ffezAXflkG6Y5ompc` (runner 35509), owning only
`tex-revision/tex/intro.tex`: remove repeated wording without changing paragraph
order, numerical claims, citations or the scoped policy/mechanism attribution.
This is separate from Qwen Next's matched-policy subsection and GLM's reply
draft. Three local sessions remain the maximum; Qwen Next's live provider
retries are not treated as a terminal failure or grounds to restart it.

## Published introduction checkpoint

The introduction worker's first call ended with actual `finish=length` and
no edits; root resumed the same session (runner 47773), without imposing a
timeout. Paper `822f78c` publishes its subsequent bounded edits: repeated
requirements/async explanations are shortened, while all numerical results
and citations remain. The historical 59-policy study's zero observed kernel
panics are deployment experience, not proof or a guarantee of safety.
Excluding comment lines, the whitespace word count changes from 1,044 to
1,010; no larger reduction is claimed. Two pdflatex passes succeed at
17 pages (conclusion page 15; `/var/tmp/revision-intro-tightening-20260907.log`).
The page-budget item remains open.

GLM's unsubmitted response draft has landed. Root requested corrections to
XSched arm labels, the Fig. 13/device-figure commit attribution, policy/whole-
system distinctions and the separation of actual open deliverables from
future extensions. It is still being revised and is not yet published as a
finished response. Qwen Next's matched-policy tightening remains live.
