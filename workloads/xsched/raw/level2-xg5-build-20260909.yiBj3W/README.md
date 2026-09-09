# XG5 device-consumed argument probe: component build

2026-09-09 05:47:43 PDT. The root compiled the local GLM's current trampoline
snapshot with the existing CUDA 12.9 pipeline: nvcc PTX, existing BPF/trampoline
merge, then ptxas -astoolspatch for sm_120. All commands exited zero; the
local cubin is 22,768 bytes. Both shared experiment leases covered the build.
No GPU workload, driver reload, or installed-tool replacement occurred.

The probe records CTA-0/thread-0's consumed argument fields in the existing
argument slot's unused four-byte tail. The emitted PTX retains the global
atomic ordinal increment and the volatile write at offset 28. It does not
change the decision or the protocol bytes at offsets 0–27.

This is temporary diagnostic code for the existing missing/repeated-body
problem, not an optimization or a performance result. The mark stores reduced
bit-width fields and the most recent entry in each slot; it is not a complete
trace of every CTA or every overwrite. The matching host drain, tool link and
real diagnostic run are still pending at this build checkpoint.

build.sh records the exact commands. source/ preserves the compiled snapshot;
xsched_guardian.ptx is the unchanged reused BPF export. Generated PTX is kept
as build inspection output. The cubin remains local and is not required in Git.
The old XG4 binary/results and all formal performance data remain unchanged.

Follow-up at 05:53:52 PDT: link-tool.sh embeds this cubin with the existing
fatbinary/carrier pipeline and links the unchanged tool object and NVBit
library. It exits zero and produces a 3,153,248-byte xsched_guard_tool.so
in this raw directory. The installed tool remains untouched. The host drain
and real diagnostic execution are still pending; this is not runtime evidence.
