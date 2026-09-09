# Native CUDA 700 follow-up: instruction-copy ordering

Read-only binary inspection on 2026-09-08, after the failed padding candidate
`bd38f2bc`; no new GPU cell or source change is associated with this note.

The installed native HAL is
`../../level2-build/.output/hal-native-20260908.i8na51/install/lib/libhalcuda.so`.
Its CMake link command uses the isolated source tree's
`3rdparty/cuxtra/lib/libcuxtra_linux_x86_64.a`.

`cuXtraInstrMemcpyHtoD` calls `cuxtra::cuda::ObjMemcpyHtoD`. In the latter's
successful path, the installed binary calls `CudaDriver::StreamSynchronize`
on the supplied stream at addresses `0x582d4` and `0x58362`: first after a
tools-memory copy and before host `memcpy`, then after the second
tools-memory copy and before freeing staging memory and returning.
The corresponding archive call offsets are `0x1da4` and `0x1e32`.

To inspect the relevant installed instructions from the repository root:

```sh
objdump -dC --start-address=0x582ad --stop-address=0x58380 \
  workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/install/lib/libhalcuda.so
```

These addresses describe this build, not a stable ABI. The observation
contradicts the specific hypothesis that this wrapper simply queues its
instruction copies and returns without waiting. Adding another identical
stream wait after these copies is therefore not a supported repair for
that hypothesized omission. This does not establish debugger-parameter
delivery, instruction-cache behavior, correct relocated SASS, or the cause
of CUDA 700. It also does not exclude other ordering defects.

Root sent this evidence to the existing GLM native-repair session
`ses_f7d4e2503ffesJKDZlvxdw0z0A` in message
`msg_083cf6591001mEcpRwBLnR6UdH`; implementation remains in progress.
The successful original-entry control and failed redirected-entry cells
remain unchanged and must not be repeated merely to reproduce this finding.
