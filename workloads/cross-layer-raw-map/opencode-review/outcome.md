# OpenCode source-readiness review

OpenCode 1.18.27 session `ses_f9564442cffeyglBSc8ydrCX6L` reviewed the
explicit attachments under model `opencode/ling-3.0-flash-fin-free`.  Both
turns used `snapshot:false`, `share:"disabled"`,
`permission:{"*":"deny"}`, and disabled `write`, `edit`, `bash`, `webfetch`,
and `task`.  The structured event output contained no tool-call events.

The first turn audited all seven properties and found each satisfied.  A
same-session second turn restated `READY`.  After the main implementation
hardened the formal gate to revalidate the preflight's raw files rather than
trusting its manifest, a third same-session turn reviewed the changed protocol
and mutation test and returned `READY`.  A fourth same-session turn reviewed
the final signal-to-cleanup delta and returned the final `READY` verdict
for that change.  A fifth same-session turn reviewed the final CLI dry-run
test delta and returned the final `READY` verdict reproduced verbatim in
[final-review.md](final-review.md).

This is an independent source/CPU-readiness review only.  It did not run the
GPU, load a BPF handler, exercise either map, or produce preflight/formal
evidence.
