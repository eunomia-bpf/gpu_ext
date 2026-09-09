# Native launch rejection: relay buffer-size control

2026-09-09 PDT, RTX 5090, NVIDIA 575.57.08. This is a runtime diagnostic,
not a performance measurement or a completed Level-2 reproduction.

The unchanged installed native candidate from
`../level2-native-wrapped-meta-20260908.NmlhJN/` was used. Both wrapped
metadata extension and original-entry control remained enabled. The same
single-BE shape as `../level2-wrapped-relay-entry-control-20260909.kv7jHZ/`
was retained: four queues, 50 kernels per queue, 9,511,106 iterations,
340 blocks and 256 threads. That preceding control fails at its first
queued launch with CUDA 701 when the full relay buffer size is passed.

## Engaged control

`control-driver.gdb` stops at the shim's first launch, obtains a handle to
the already-loaded real driver with `dlopen` using `RTLD_NOLOAD | RTLD_NOW`,
resolves its `cuLaunchKernel` with `dlsym`, then places an instruction-address
breakpoint at that driver's actual function entry. On this x86-64 ABI the
eleventh argument (`extra`) is at entry stack offset 40 bytes. For the
relay buffer form, the script changes only its declared size from `0x1520`
to `0x20`, the original compute kernel's parameter size. It does not rebuild
the HAL, undo the metadata extension, or change the original-entry setting.

The completed run is `debug-driver.log`, with `lifecycle-driver.log` and
`xserver-driver.log`. It starts at 00:34:09 and finishes at 00:34:22 PDT.
Both GPU and struct-ops leases cover the run; a dedicated `xserver HPF 50000`
is stopped afterward. The command is the existing service-mismatch workload
with arguments `be 1 4 50 9511106 340 256 1 0` and the existing `GO` input.

Observed:

- 200 actual relay-size modifications and 200 queued launch returns of zero.
- The worker exits normally and emits 200 completed service records, each
  with 340 completed blocks; its existing output check reports 17,408,000
  values validated.
- All four queues report zero suspend/resume operations, as expected for
  this single-BE control without a higher-priority competitor.
- GPU utilization is 0% with 1 MiB used at the final observation (P0
  immediately after execution). No driver reload or recovery is needed.

The change from rejected launch to successful execution with only the
declared relay size shortened establishes that the large relay declaration
is causally relevant to this candidate's launch failure. It does not yet
identify the precise driver's internal limit or the required metadata fix.
Other retained resource adjustments do not independently prevent this
original-entry workload from running with the shorter buffer.

**This is not a production repair:** shortening the buffer omits the
guardian argument block at relative offset `0x1500`. Original entry remains
selected; no guardian/resume behavior is demonstrated. The raw timings
include debugger interference and are not added to a performance table.
GLM received the result in `msg_0851785a3001jxHgpH1x5xDzn6` and continues
implementing full, accepted argument delivery. No old result is replaced.

## Retained setup failures

`control.gdb` / `debug.log` target the shim's same-named function and make
zero buffer-size modifications; the worker still fails. They do not test
the intended intervention. `control-frame.gdb` / `debug-frame.log` try a
driver-frame offset at that shim breakpoint and encounter an inaccessible
address before performing the intervention. Both root-authored debugger
setup failures and their lifecycle/server logs are retained separately.
Only the driver-resolved control above supports the conclusion.

## Implementation follow-up (not another run)

GLM subsequently added an opt-in KPARAM descriptor-growth candidate. Root
identified a required companion change in the current native HAL:
`platforms/cuda/hal/src/common/cuda_command.cpp`, constructor lines 81–96,
queries `cuXtraGetParamInfo` and copies each reported size from the original
application argument pointer. If the driver's descriptor query exposes the
expanded final argument, that copy would read beyond the original 8-byte
argument before the relay is constructed. The metadata pass also reaches
the timer kernel, whose ordinary launch does not use the relay.

The candidate therefore needs to retain the original host argument layout
and use it when constructing a padded launch buffer, while preserving
non-relay launches. Root sent these concrete source findings in
`msg_0851bf163001zYyGA2ChlwT6iI`. The widened-descriptor candidate has not
been run; this paragraph reports a source-level dependency, not a measured
failure or a verified fix. The successful diagnostic above remains unchanged.
