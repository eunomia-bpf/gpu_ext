# Review of the proposed prefetch diagnostic patch

2026-09-03: root and an independent source reviewer compared the proposal to
the actual 575 driver. No policy/validator/actuator semantic change was found.
The original branch conditions, translations and returned region remain in
place; the additions are a local scalar context, two void/const-pointer
diagnostic calls, and a counter immediately after each real native-loop body
entry's `get_range` call.

Both modified C files are already compiled by the existing UVM Kbuild. The
SELECTED event follows the actual initial-effect choice. FINISHED is at the
single return path after BYPASS, ITERATE or native execution. The proposal
copies the initial region result before native translation may overwrite the
local result variable. Root's read-only `git apply --check --whitespace=error-all`
against the current sibling `gpu_ext-kernel-575` source exits 0. The patch was
**not applied**.

For action 99 with attempted legal region `(0,0)`, current source semantics
predict region result APPLY and initial effect NATIVE. APPLY describes region
validity, not acceptance of action 99. `native_iterations` counts loop-body
executions, not every `get_range` call inside traversal macros. Output is the
selected region, not a final prefetch hint, filtered compute mask, completed
DMA or measured PCIe traffic.

This is only a reviewed source proposal. There has been no driver compilation,
new symbol/BTF inspection, fentry admission, SELECTED/FINISHED observation,
target readback, module reload or reboot. The two hooks and local counter add
functional-diagnostic overhead and must not be used for performance cells.
Follow-up review found that the proposed identity tokens were raw stack and
bitmap-tree addresses. Omitting them from reports would not make that tracing
interface safe, so the proposal now removes both fields and their assignments,
along with fixed constants and an unused page index. The observer must pair one
outstanding SELECTED/FINISHED frame per full `pid_tgid` and reject nesting or
unmatched phases. The actual Q2 transition test remains open despite this
successful source review.
