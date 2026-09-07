# Fresh Fig. 13 runtime continuation

Completed: all 20 cells ran in `results/fig13_fast_20260907_005958`, published
with the [performance report](results-performance-575-20260907.md) in
`10994d21`. Original core, saved GDS UVM, both services and GDS loader were
restored. Root used logged ordinary shell commands and the existing runner;
the proposed reusable lifecycle script below was not used. Do not rerun the
completed matrix when that script or the reusable analyzer finishes.

The Qwen Next lifecycle session subsequently returned `Error: <none>` and
its process exited; it produced no script. It was not stopped for silence or
by a root-imposed timeout. This does not affect the completed logged manual
execution. The remaining work is the reusable analysis and expanded plot,
not another module swap or performance campaign.

## Retained preparation record

The old `results/fig13_fast_20260905_155839` contains startup failures, not
completed performance cells. Source `ad2e5748` fixes uvmbench's required
`--output=PATH` argument, removes broad process/struct-ops cleanup, defaults
to no tenant timeout, and rotates all four arms modulo four. Local Qwen 27B
implemented the change; root reviewed it, compiled the Python entry, and
built the existing memory/scheduling policy tools. Historical files remain.

The currently loaded core lacks `nv_gpu_sched_ops`; the built candidate
`/home/yunwei37/workspace/gpu/gpu_ext-kernel-575-gds/kernel-open/nvidia.ko`
from driver `a2b40efd` contains the scheduler interface. The current GDS UVM
must remain the saved 61,945,872-byte module at
`/var/tmp/gds-restore-before-stale-20260907.AWgdmi/nvidia-uvm.ko`, not the
larger stale-state candidate or the stock packaged UVM.

The plan was to temporarily load the candidate core, run five rotated four-arm blocks
(baseline, memory only, scheduling only, combined), restore the installed
stock core and saved GDS UVM, then restore the previously active GDM,
persistence service and owned GDS loader. Current device clients are only
the GDM greeter and persistence service, with no active user graphical
session. This pre-run description is superseded by the completed record above.

Local OpenCode session `fig13-scoped-core-lifecycle`, using Qwen Next, owns
the bounded lifecycle script and may not execute it. Root will review it for
future use, not rerun this matrix. OpenCode is not given a short timeout and will not be stopped for
silence. Module replacement uses ordinary unload, never force-unload or
broad process termination; restoration errors must remain explicit. The
experiment retains all outputs without extra correctness or timing gates.

This does not repeat MoE, XSched, GPreempt, LMCache, Table 1 or the completed
stale-state campaign. The current LMCache improvement is published in
`a451db6a`; stale-state phase analysis in `1ea66808` and paper integration
in `57e9937` are complete.
