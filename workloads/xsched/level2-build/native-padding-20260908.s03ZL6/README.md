# Native sm_120 parameter placement: compiled

Both local-GLM-authored padded CUDA specimens compile successfully with
CUDA 12.9 and g++13. The compiler directly emits the required debugger-region
loads, eliminating the need to infer and rewrite LDC immediate bitfields:

- check: LDC.64 at `c[0x0][0x1880]`; LDC.64 and LDCU.64 at `0x1890`.
- restore: LDC.64 at `0x1880` and `0x1888`.

The unused leading by-value parameter occupies 0x1500 bytes; the real
arguments follow the compiler's 0x380 parameter-region base. These are
extracted-prefix compiler specimens, not normal ABI-launched kernels.

Build command (from `workloads/xsched/level2-build`):

```sh
flock /tmp/gpubpf-revision-gpu0.lock flock /tmp/gpubpf-revision-struct-ops.lock make -j2 NVCC="/usr/local/cuda-12.9/bin/nvcc -ccbin /usr/bin/g++-13" CXX=g++-13 BUILD=.output/native-padding-20260908.s03ZL6 .output/native-padding-20260908.s03ZL6/native/check_preempt_port.cubin .output/native-padding-20260908.s03ZL6/native/restore_exec_port.cubin
```

`build.log` retains compiler commands; `check.sass` and `restore.sass` are
actual nvdisasm output. Untracked cubins are 11568/11288 bytes respectively.
No GPU launch, driver change, or performance measurement occurred.

## Still unfinished

The check specimen still ends with unconditional EXIT at 0x250: the compiler
removed the final conditional flag read at the end of the global function.
This cannot yet be concatenated as a working guardian prefix. Restore emits
`CALL.REL.NOINC R2` at 0x140; the old indirect-call parser is not sufficient.
Conditional-exit retention, restore transfer handling and final prefix-array
extraction remain assigned to GLM. Parameter placement is solved; the whole
native Level-2 execution path is not claimed complete.
