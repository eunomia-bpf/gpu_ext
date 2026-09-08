# Field-major full-record layout, 2026-09-08

This new layout experiment changes only the organization of the GPU-local
record payload: adjacent threads store each u64 field eight bytes apart,
rather than storing 80-byte records. The ten original fields, per-thread
timestamp, 524288 slots, 256-record capacity per slot, 32 banks, and
335675408-byte bank size remain. The old AoS and ring measurements are not
replaced. Source is main `e6b049bc`; runtime remains bpftime `886b4ca`.

Local Qwen supplied the header, writer and collector patches. Root applied
the already-generated collector patch and repaired one remaining renamed
type reference, then built a separate staging copy at
`/home/yunwei37/workspace/gpu/bpftime-auto-warp/example/gpu/full-record-soa-20260908.fpd5QL`.
The persistent Makefile option is separate follow-up work; this build used
the existing Makefile with explicit compiler flags:

```sh
make -C /home/yunwei37/workspace/gpu/bpftime-auto-warp/example/gpu/full-record-soa-20260908.fpd5QL \
  -j2 CUDA_HOME=/usr/local/cuda-12.9 OUTPUT=.output-soa \
  CLANG='clang -DFRDB_SOA_LAYOUT' CC='cc -DFRDB_SOA_LAYOUT'
```

That command returned zero (`build-retry.log`). The first attempt instead
overrode `CFLAGS`; that override propagated into the recursive bpftool
build and removed its include flags, causing missing `bpf/hashmap.h`.
The failed build log remains as `build.log`; no GPU run used that build.
The compiler's unused linker-input warning is retained. Generated object,
skeleton, executable and caches remain local, not in Git.

`execute.py` invokes the existing pp512/tg0 TinyLlama runner once, using
AUTO_WARP_EXECUTION=0 and the SoA collector copied locally as `kernelretsnoop`.
Root holds both shared experiment locks during build and execution. There
is no clock-precision prerequisite or extra validation campaign. The
collector's existing counts and post-run copy time are retained with the
throughput result. One cell alone is not a paired overhead estimate.

## First prefill completed

Client and collector both return zero. Prefill throughput is
**27677.720835 token/s**. The collector reports 23068672 committed records,
524288 active slots, zero overflow/out-of-range events, and 23068672
nonzero timestamps. The 10741613056-byte bulk drain takes 1499138759 ns
(1499.139 ms), outside prefill timing. GPU is idle after normal cleanup.
The existing warning-mode verifier emits map-pointer warnings; this is
performance evidence, not a claim of strict admission.

The preceding AoS five-block campaign's median was 23267.228804 token/s;
that is historical context, not this cell's paired control. Retain this
successful SoA cell as block 1 when adding its fresh baseline and AoS
controls, then finish the remaining rotating blocks. Do not rerun this
completed cell or the older 15-cell ring/AoS campaign.

## Persistent build option

The local model's Makefile now supports `LAYOUT=soa` with `.output-soa/`
objects and `full-record-device-buffer-soa`, and default `LAYOUT=aos` with
`.output/` objects and `full-record-device-buffer`. Both variants build
successfully in the staging directory. The SoA target links the already
compiled SoA objects (`build-make-soa.log`).

The first default invocation reported nothing to build because the earlier
manual SoA build had used the default executable name and the Makefile's
`.SECONDARY` behavior did not rebuild missing intermediates. It did not
build an AoS executable. Explicitly requesting
`.output/full-record-device-buffer.o full-record-device-buffer` subsequently
compiled and linked the default layout successfully; both logs are retained
(`build-make-aos.log`, `build-make-aos-explicit.log`). Fresh builds do not
inherit that manual-build executable. The measured SoA executable copied
into this raw directory was not overwritten or used for another run.
