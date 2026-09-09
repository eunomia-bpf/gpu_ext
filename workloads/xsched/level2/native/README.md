# Native sm_120 guardian-stream extraction

The two stub kernels in this directory provide the guardian prefix and the
resume prefix that `GuardianSM120` copies into instruction memory:

- `check_preempt_port.cu` compiles to the check stub. It reads the
  preemption buffer (`c[0x0][0x1880]`) and the kernel arguments
  (`kernel_idx` at `+16`), and it retains a conditional exit behind the
  build-only `xg_check_exit_marker` store. The marker keeps the exit
  predicated after optimization; it writes an otherwise unused device
  marker and does not affect the extracted stream.
- `restore_exec_port.cu` compiles to the resume stub. It reloads the
  instrumented kernel entry point (`c[0x0][0x1888]`). The 0x1500-byte
  by-value pad parameter forces both stubs' parameter regions into the
  debugger window at `c[0x0][0x1880]..`, so no LDC immediate rewriting is
  needed. The extractor verifies the stub's compiled transfer tail
  (`HFMA2 R21=0`, `MOV R20` call-end preload, `CALL.REL.NOINC R2
  0xfffffffc`) on the actual instruction words, then rewrites the last
  three slots of the emitted resume stream to the transfer mechanism
  byte-identical in both runtime-proven upstream resume stubs (sm_70 and
  sm_86): `LDC R21, c[0x0][0x188c]`, `LDC R20, c[0x0][0x1888]`,
  `RET.ABS.NODEC R20 0x0`. The shared old-arch encodings are a candidate,
  not proof of sm_120 support on their own; the passing native run
  (raw `level2-native-retabs-20260909.oJDCbU`) is the evidence.

`ldc_patcher.cpp` consumes the two compiled cubins plus `nvdisasm` and emits
`xg_sm120_guardian_arrays.h`:

```
xg_ldc_patcher CHECK.cubin RESTORE.cubin NVDISASM OUT.h
```

For the current build layout (run from `level2-build`):

```
../level2/native/ldc_patcher \
    .output/<BUILD>/native/check_preempt_port.cubin \
    .output/<BUILD>/native/restore_exec_port.cubin \
    /usr/local/cuda-12.9/bin/nvdisasm \
    .output/<BUILD>/native/xg_sm120_guardian_arrays.h
```

The Makefile `ldc-patch` recipe passes these four arguments (the obsolete
probe cubin argument is gone; `probe.cu` is kept only as the historical
device of proof for the LDC immediate analysis).

## What the patcher verifies before emitting

- Both `.text.*` sections are 16-byte instruction streams and carry no main
  `.rela.text` relocations (the HAL copies the bytes raw; nothing could
  resolve a load-time relocation afterwards).
- The padded parameter ABI is intact: window at offset 0x380/0x1510-0x1518,
  ord 0 pad of 0x1500 bytes, and each real parameter lands exactly on its
  debugger slot (check: `preempt_buf` 0x1880, `_kernel_idx` 0x1890; restore:
  entry point 0x1888).
- The in-window consumers (decoded from `nvdisasm -c`) match the pinned
  prototype: check consumes {0x1880, 0x1890}, restore consumes {0x1880,
  0x1888}, all `LDC`/`LDCU` with width 8. Anything else fails loudly.
- The check blob cuts at `EIATTR_EXIT_INSTR_OFFSETS[0] + 16`, the retained
  predicated `@P0 EXIT` (encoding `0x...094d`). The marker store and the
  final standalone `EXIT` (`0x...794d`) are excluded; the extracted prefix
  falls through into the original kernel text that the HAL appends behind
  it. If the marker were removed from the stub, nvcc would collapse both
  exits into one unconditional `EXIT` and extraction fails (`exit_n != 2`).
- The restore blob cuts at the final `EXIT` and keeps, in order: the
  `@!P0 EXIT` (`0x...894d`), the entry-point load, and the rewritten
  transfer tail (`LDC R21/R20` from `0x188c/0x1888`, `RET.ABS.NODEC
  R20 0x0`). The compiled call tail - `HFMA2 R21=0`, `MOV R20` (call end),
  `CALL.REL.NOINC R2 0xfffffffc` - is still verified word-by-word before
  the rewrite (it pins the compiler-observed shape); the emitted stream
  replaces it. The transfer target is the full absolute entry address
  loaded from the debugger arguments; the transferred-to instrumented
  image ends with the original kernel `EXIT`, so nothing returns into the
  copied prefix. First run with this form (raw
  `level2-native-retabs-20260909.oJDCbU`) completed exit 0 with no repeat
  of the previous CUDA 700.

## Header contract

`xg_sm120_guardian_arrays.h` defines
`XG_SM120_CHECK_PREEMPT_BYTES` (currently 0x280) and
`XG_SM120_RESTORE_EXEC_BYTES` (currently 0x150) and the two
`static const unsigned long long` arrays. Each SASS instruction is two
`ull` entries: the 8-byte encoding word followed by its 8-byte
scheduling/control word, byte-identical to the cubin `.text` redaction the
HAL memcpy's. `static_assert`s pin `sizeof(array)` to the macros.

## Remaining unsupported forms (loud failures, not silent wrong output)

- Any change in consumer layout, parameter ABI, exit structure, or the
  pinned call/return-site encodings aborts extraction instead of emitting a
  wrong blob. Opcode constants (`0x094d`/`0x894d`/`0x794d`, `0x7344`,
  `0x7802`, `0x7431`) are pinned to the CUDA 12.9 (ptxas V12.9.86)
  specimens and must be re-derived for a newer toolkit.
- The REPLACEMENT transfer words (`0x00062300ff157b82`,
  `0x00062200ff147b82`, `0x0000000014007950` with control words
  `0x000fc00000000800` x2 and `0x001fea0003e00000`) are copied verbatim
  from the sm_70 and sm_86 upstream resume streams (the two archs carry
  identical words). sm_120 does not emit them itself; they are a shared
  old-arch candidate that the run validated under CUDA 12.9 / driver
  575.57.08.
- The compiled CALL `callinfo` immediate stays `0xfffffffc` exactly as
  compiled and the return site (`R20 = 0x150`) lies past the blob end; in
  the EMITTED stream both are replaced together with the call, so they
  survive only inside the cubin verification path, never in what the HAL
  copies.
- The marker literal is `HFMA2`-encoded; there is no token-search fallback
  for it (EIATTR exit offsets are the cut mechanism).
- Single-kernel cubins only; `.nv.merc.*` mirrored sections and
  `.nv.constant0` are not consumed; non-LDC/LDCU in-window consumers fail
  loudly.
