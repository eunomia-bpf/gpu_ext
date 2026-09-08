# Native guardian conditional exit retained by the compiler

Root mechanically integrated the build-only trailing-store snippet proposed
in local GLM session `ses_f7d9ef4edffeGDLDitPzpt2050`'s terminal response.
The unchanged decision body now has a store after its conditional exit, so
nvcc cannot merge both paths into an unconditional end-of-kernel EXIT.

Actual CUDA 12.9 / g++13 sm_120 compilation succeeds. `check.sass` shows:

- 0x240: barrier;
- 0x250: block-exit flag load;
- 0x260: predicate computation;
- 0x270: predicated EXIT;
- 0x280–0x2a0: build-only marker address/materialization/store;
- 0x2b0: final standalone-kernel EXIT.

The eventual extracted guardian prefix must stop at 0x280, excluding the
marker sequence and final EXIT. The compiler materializes the marker value
using HFMA2, not a MOV containing its source literal: searching disassembly
for that literal is not an appropriate extraction assumption. No marker
kernel or incomplete prefix was launched.

This resolves conditional-exit retention only. Restore CALL.REL.NOINC
handling, prefix extraction and the full native Level-2 route remain
unfinished. The previous GLM response ended naturally at its output length
limit. A fresh GLM context, `ses_f7d4e2503ffesJKDZlvxdw0z0A`, receives the
actual compiled inputs and those remaining tasks; the old session is retained.

Build command, from `workloads/xsched/level2-build`, under both shared leases:

```sh
make NVCC="/usr/local/cuda-12.9/bin/nvcc -ccbin /usr/bin/g++-13" BUILD=.output/native-exit-retention-20260908.NAeniq .output/native-exit-retention-20260908.NAeniq/native/check_preempt_port.cubin
```

`build.log` records compiler commands, `check.sass` the nvdisasm output.
The 12616-byte cubin remains a local build artifact, not a Git payload.
No GPU measurement, driver change, clock calibration or old-cell replay.
