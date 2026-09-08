# First real Level-2 component build, 2026-09-08

Command: `make -j2 all`, under the two shared experiment leases. It ended
with exit 2 at BPF-to-PTX export: the existing GPU verifier classifies five
snapshot-dependent branch conditions as lane-varying (instructions 6, 9,
14, 17, 20). The 48-byte-context exporter itself compiled and linked.

`components.log` retains the complete first build. This is not a completed
Level-2 build or execution result. The next local-Qwen edit preserves the
same decision rule as straight-line unsigned arithmetic, without weakening
the verifier or asserting that arbitrary snapshot values are uniform.
Frozen Level-1 artifacts, dependencies and results are unchanged.

The independent native cubins and LDC patcher built. First array generation
failed because the three kernel symbols were C++-mangled while the patcher
expects C names. Root added `extern "C"` to the three declarations; rebuilding
then reaches the next actual issue: the probe has five collected LDC samples,
not the assumed six. Both generation logs are retained. Local GLM owns that
native parser/probe repair separately from Qwen's branchless BPF decision.
No Level-2 execution or performance is claimed from these component builds.
