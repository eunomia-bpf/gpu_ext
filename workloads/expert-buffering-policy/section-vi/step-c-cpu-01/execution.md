# Step C bounded CPU validation attempt 01

2026-09-03. Coordinator-authorized after the HB timing window ended.
Only CPU 17; no GPU, CUDA import, OpenCode or Git operation. The accepted
stage02 offloader, State, Step B and original FineMoE remain unchanged.

Commands (each has a 30-second deadline and five-second forced-kill grace):

```sh
timeout --kill-after=5s 30s taskset -c 17 make -C workloads/expert-buffering-policy/section-vi -f shadow.mk -j1 test-shadow
timeout --kill-after=5s 30s taskset -c 17 /usr/bin/python3 -B workloads/expert-buffering-policy/section-vi/test_correctness.py
```

Both commands exited 0 on the first attempt. The new shadow library built,
then six bridge tests passed (0.034 s), including five actual host-uBPF JIT
decisions with native post-checks. Eight controller tests passed (0.011 s):
all three synthetic arm records, preflight/timed separation, inherited-shadow
removal, exact token and engagement rejection cases, randomized block shape,
changed/incomplete preflight rejection, and source AST parsing including the
new analyzer. The analyzer has not processed a real EB campaign yet.

Complete merged stdout/stderr is in `shadow.log` and `correctness.log`;
`execution.json` records exact commands, UTC start/end and exit status. No
failed attempt was replaced. These CPU tests are not live GPU correctness or
Section VI performance evidence; the three preflight and 15 timed cells remain.
