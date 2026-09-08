# Level-2 component builds, 2026-09-08

Latest: the branchless-barrier device components build successfully. Details
and artifact inventory follow the preserved first failure below.

## First real component build

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
## Completed device component build

`make -j2 bpf-ptx guardian-cubin guard-tool` now exits zero. It ran under both
shared resource locks after the Table 1 campaign ended, without overlapping
performance measurement. The complete log is
`branchless-barrier-components.log`.

Local Qwen's equivalent branchless decision uses full-word select masks and
empty register compiler barriers to prevent clang from reconstructing
conditional branches. The real compiler pipeline reports 49 BPF instruction
words, accepted by the unchanged 48-byte-context GPU verifier. PTX assembly
and library linking complete for sm_120.

Local outputs (not committed):

| File under `.output/` | Bytes |
|---|---:|
| `xsched_guardian.bpf.o` | 6864 |
| `ptx/xsched_guardian.ptx` | 1293 |
| `xg_guardian.cubin` | 20192 |
| `xsched_guard_tool.so` | 3153080 |

The exact current sizes are an artifact inventory, not performance results.
The shared NVBit library also links successfully. No native/BPF Level-2 GPU
comparison has run. The separate native LDC generator and HAL integration
remain unfinished; frozen Level-1 measurements are unchanged.

The subsequent native parser checkpoint also builds with `make -j2 native`
(exit zero, `native-width-parser-components.log`). It retains LDC/LDCU
instruction form and 4/8-byte consumer width, and adds parameter-window
metadata parsing. Those metadata helpers are not yet wired into the old
encoding solver/main flow; a compiled parser is not a generated native
guardian or a successful end-to-end run. Root added the standard `<cctype>`
include required by the new character-classification calls.

The next padded probe also builds (exit zero,
`native-padded-probe-build.log`). Its actual disassembly is retained as
`native-padded-probe.sass`: parameter-window metadata is `0x380/0x1520`,
and a real `LDC.64` consumes `c[0x0][0x1898]`. This exposes a high-offset
instruction for the pending encoding adaptation. It is a compiler specimen,
not a GPU performance or successful guardian-injection result; main/solver
integration remains to be done.

Earlier rejected compilations remain in `components.log`,
`branchless-components.log` and `branchless-fullmask-components.log`.
The first arithmetic draft used a one-bit select mask; root corrected it to
an all-bits mask. The ordinary full-mask expression still let clang emit
lane-varying branches; Qwen then added compiler barriers. These failures are
not deleted or presented as successful builds.
