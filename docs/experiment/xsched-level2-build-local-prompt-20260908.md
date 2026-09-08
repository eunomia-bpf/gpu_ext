# XSched component-build local implementation prompt

Session: `ses_f7eaa07b1ffe5B4GsoMLSMvmGT`, local OpenCode/direct Qwen 27B.
Started only after the disk-UVM implementation session reported terminal
completion and its status became idle. The existing GLM source owner was
notified of the disjoint sibling-directory ownership before dispatch.
This records an implementation assignment, not a completed build or result.

## Exact prompt

Implement a bounded XSched Level-2 component build entry, not a new experiment or algorithm.

You are one of at most three LOCAL OpenCode sessions. Main source owner GLM continues writing workloads/xsched/level2/ host/HAL/native/device code. You exclusively own NEW sibling workloads/xsched/level2-build/ (Makefile and short README, and only a strictly needed local link helper). Do not modify level2/, deps/, existing build/results, bpftime, driver, paper or other worktrees. Root commits/pushes/builds/measures.

Read workspace AGENTS.md, gpu_ext/AGENTS.md and workloads/xsched/AGENTS.md. No file/content hashes/checksums/fingerprints. Never introduce gates/audit frameworks. Do not run GPU work, driver reload, nvcc/clang compilation, or large CPU builds: another task owns the GPU and CPU build windows. Source inspection and make -n for your targets are allowed. No artificial short timeout. Use bash and read; apply local edits ONLY with installed /home/yunwei37/.codex/tmp/arg0/codex-arg029oUiU/apply_patch via bash. Native apply_patch is not exposed to this Qwen provider; do not loop on invalid tool calls. Do not delegate.

Relevant real source:
- workloads/xsched/level2/xsched_guardian_abi.h
- bpf/xsched_guardian.bpf.c
- tramp/xsched_guard_tramp.cu and tramp/xg_bpf_abi.h
- tool/xsched_guard_tool.cu and tool/merge_ptx.sh
- native/check_preempt_port.cu, native/restore_exec_port.cu, native/probe.cu, native/ldc_patcher.cpp (may evolve under GLM).
- Read-only authoritative working build example workloads/sass-kretprobe/Makefile and its exporter/linking scripts.
- Read-only bpftime exporter checkout /home/yunwei37/workspace/gpu/bpftime-sass-existing-application; existing tools can be discovered by rg --files and source/README.
- NVBit dependency path should come from existing sass-kretprobe Makefile, not be invented.

Deliver a minimal component Makefile using explicit component targets:
1. real BPF ELF to device-callable PTX exported xsched_guardian, with correct symbol/context ABI (derive exporter CLI from local source).
2. shared trusted trampoline PTX merged with BPF PTX using existing level2/tool/merge_ptx.sh, assembled sm_120; native-C vs BPF selection uses SAME actuator glue, no host decision fallback.
3. NVBit xg guard shared library linking the component as the working SASS example does.
4. native probe/stub/ldc_patcher component targets if current source supports them; DO NOT invent completed original cuXtra/HAL integration. Missing HAL/replay integration remains GLM's work and is explicitly named in README.
All outputs under your sibling directory's .output/ (not frozen source/output). Default make builds components only and never runs GPU experiments. Build commands should permit installed CUDA12.9 root and explicit dependency overrides.

Keep paper-scale performance protocol out of this small build task. README says SOURCE PREPARATION / UNBUILT until root actually builds, with the exact compile commands and remaining source integration dependencies. No fabricated output or benchmark numbers. Finish with changed-file inventory and commands.
