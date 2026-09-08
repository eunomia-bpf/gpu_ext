# Native sm_120 guardian-stream extraction

The two stub kernels in this directory provide the guardian prefix and the
resume prefix that `GuardianSM120` copies into instruction memory:

- `check_preempt_port.cu` compiles to the check stub. It reads the
  preemption buffer (`c[0x0][0x1880]`) and the kernel arguments
  (`kernel_idx` at `+16`), and it retains a conditional exit behind the
  build-only `xg_check_exit_marker` store. The marker keeps the exit
  predicated after optimization; it writes to an otherwise unused local
  buffer and does not affect the extracted stream.
- `restore_exec_port.cu` compiles to the resume stub. It reloads the
  instrumented kernel entry point (`c[0x0][0x1888]`) and transfers through
  an intact register-target `CALL.REL.NOINC R2`. The 0x1500-byte by-value
  pad parameter forces both stubs' parameter regions into the debugger
  window at `c[0x0][0x1880]..`, so no LDC immediate rewriting is needed.

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
  `@!P0 EXIT` (`0x...894d`), the R20/R21 return-site preload (`R20 = call
  end offset`, `R21 = 0`), and the preserved `CALL.REL.NOINC R2 0xfffffffc`.
  The call target is the full absolute entry address loaded from the
  debugger arguments; the called instrumented image ends with the original
  kernel `EXIT`, so the call never returns into the copied prefix.

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
- The CALL `callinfo` immediate stays `0xfffffffc` exactly as compiled and
  the return site (`R20 = 0x150`) lies past the blob end; both are benign
  only because the callee never returns into the prefix.
- The marker literal is `HFMA2`-encoded; there is no token-search fallback
  for it (EIATTR exit offsets are the cut mechanism).
- Single-kernel cubins only; `.nv.merc.*` mirrored sections and
  `.nv.constant0` are not consumed; non-LDC/LDCU in-window consumers fail
  loudly.
