# Full-record GPU-local buffer: first component build

The local Qwen implementation now has a shared value layout, BPF writer and
host collector. Root compiled the BPF object, generated its libbpf skeleton,
and compiled/linked the collector; all three commands exit zero. No CUDA
workload has run with this implementation yet. This is not a performance
result or an update to Table 1.

The writer retains all ten u64 fields per thread, with eight banks of
65536 thread slots and 256 records per slot. Each bank value is 1342701584
bytes. The collector drains eight whole values after client completion,
reuses one host destination allocation, and reports copy time separately.
This checkpoint uses thread-major record indexing; record-major placement
for adjacent-thread writes is the next local-model edit, not measured here.

Builds use clang `-g -O2 -target bpf -D__TARGET_ARCH_x86`, the current
bpftime-auto-warp vmlinux header, and the existing libbpf headers/archive
from `../table1-original-ring-encoded-20260908.iVdbES/gpubpf_tool_build/kernelretsnoop/.output/`.
The existing bootstrap bpftool generated the skeleton. Host linking uses
`cc -g -Wall`, CUDA 12.9 headers, `-lcuda -lelf -lz`. BPF and host compilation
hold both shared experiment leases; neither changes the loaded driver.

`bpf-build.log` is empty (successful compilation). `collector-build.log`
retains four PRIu64 versus ULL-constant formatting warnings; their casts
are queued to the source owner. The BPF object is 12976 bytes, and the
collector executable is 1549288 bytes. Generated objects, skeleton and
executable remain local build artifacts, excluded from the commit.

Remaining: finish the local-model Makefile/README and layout edit, build
that final source, then run the existing pp512 throughput comparison.
No clock calibration or additional audit campaign is introduced.
