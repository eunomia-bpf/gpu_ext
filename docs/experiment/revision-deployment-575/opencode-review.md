# OpenCode read-only evidence review

- OpenCode version: 1.18.27
- Model: opencode/ling-3.0-flash-fin-free
- Session: ses_f9577f689ffeuJftYeoEvhzK60
- Tool policy: all permissions denied; write, edit, bash, webfetch, and task disabled

All eight check points hold:

1. **Runnable CPU evidence is cleanly separated** — RESULTS.md has distinct sections for CPU lifecycle tests, source audit, and open-module comparison; future-GPU work is explicitly marked "not run in this audit."

2. **Lifecycle checks are fail-closed** — `run_cpu_lifecycle.sh` validates CUDA attach is OFF, agent links no CUDA libraries, no inherited state, and the summary awk flags any repetition where `target_alive`, `ipc_socket`, `bpftime_used`, or `command_status` deviate. The Yama `ptrace_scope=1` and target `PR_SET_PTRACER_ANY` opt-in caveat is stated in RESULTS.md and `run-environment.tsv`.

3. **Frida-mediated ptrace is not misrepresented** — `semantic-checks.tsv` confirms "CLI delegates running-process injection to Frida" and "no direct ptrace call in product CLI or agent." RESULTS.md explicitly says this verifies "an actual ptrace lifecycle inside Frida" and "does not show that an unmodified process under a restrictive ptrace policy is attachable."

4. **CPU milliseconds are not GPU performance** — RESULTS.md states "These small measurements are lifecycle checks, not performance results" and explicitly removes 273 ms as "Unsupported by retained samples."

5. **NVBit is not called a working SASS prototype** — Results call it "an NVBit API experiment or baseline, not evidence of a working SASS gpubpf prototype"; device clock reads are commented out (no-op); future protocol gates 1, 2, 4 explicitly fail.

6. **Physical-line scope is explicit** — Results say "This is not a semantic source-line metric" and the TSV provides both `local_physical_lines` and `local_nonblank_lines`.

7. **Module symbols are artifact-only** — Results state "This is artifact/symbol inspection only: the audit did not load modules or exercise GPU hardware."

8. **Future protocol has correctness gates** — `future-gpu-protocol.md` requires all gates hold per trial, "Any missing gate makes the trial a failure, not a censored latency," and "do not silently omit timeouts."

VERDICT: PASS
