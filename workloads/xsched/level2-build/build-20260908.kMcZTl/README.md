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
