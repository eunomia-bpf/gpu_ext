# Real-KV disk/UVM restore helper: build handoff

2026-09-09 PDT. This records a completed helper build, not a new primitive
measurement, end-to-end serving run, or performance result.

The local Qwen 27B session `ses_f80c496aeffeBh330rKhrRwPgX` implemented
`../../gds-control/diskuvm_fault.cu`. Its C entry
`diskuvm_fault_read(size_t va, size_t size, int device)` launches full-range
volatile GPU reads and waits for that traversal to complete. It is intended
to trigger the existing disk-backed UVM GPU-promotion mechanism on actual
completed LMCache KV chunks. That mechanism uses CPU staging; this does not
implement NVMe-to-GPU P2P or automatic pressure-driven offload.

Root's first build failed because `uint32_t` had no defining include.
Root added only `<cstdint>` and repeated the build. The failed and successful
outputs are retained as `build.log` and `build-fixed.log`; the latter is
empty because compilation emitted no diagnostics. Both build commands were
covered by the GPU and struct-ops leases; neither executed GPU kernels or
changed loaded drivers.

Successful command, from repository root:

```
/usr/local/cuda-12.9/bin/nvcc -shared -arch=sm_120 -Xcompiler -fPIC \
  -o /tmp/lmcache-diskuvm-serving-build-20260909.7dZdC8/libdiskuvm_fault.so \
  workloads/lmcache-disk/gds-control/diskuvm_fault.cu -lcudart
```

The command exits zero. The resulting shared object is 26,648 bytes and
exports `diskuvm_fault_read`, as recorded in `lifecycle-fixed.log`. The
2,915-byte CUDA source is committed; the generated binary stays outside
Git at the path above, ready for `LMCACHE_DISK_UVM_FAULT_LIB` when the
serving connection is ready.

Remaining integration work is still owned by the same local session:
capture the KV size before completion/refcount release, connect the backing
wrapper after admission, forward the opt-in through the real runner, and
report actual restore versus fallback use. The exact-size Python CUDA
allocator also landed but is still part of that uncommitted integration.
No new timing or completed-LMCache claim follows from this build. Previous
primitive and serving measurements remain unchanged and are not rerun.
