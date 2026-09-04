# Device-side Table 1 compact-coordinate readiness (2026-09-04)

## Status

Preparation passed; this document is **not a performance result**. The failed
`raw/preflight-575-noncross-clock-02` campaign remains immutable failure
evidence and will not be resumed or reclassified. The repaired design must run
in a fresh `raw/preflight-575-noncross-clock-03` campaign before any full run.

## Repair

The gpubpf and NVBit exit probes now expose the same 32-byte record:
`(global_x, global_y, global_z, timestamp)`, with four 64-bit fields. This is a
global logical-thread coordinate; it does not preserve the original CUDA
block/thread decomposition.

Correctness retains 22,528 logical coordinates and 256 entries per gpubpf
ring. Its exact oracle requires extent 88×256×1, 220 selected launches,
720,896 records, and coordinate multiplicities 1,024@220, 1,024@44, and
20,480@22.

Timing uses 44 entries per coordinate because the fixed benchmark performs 44
selected launches. At pp32 the exact layout is 128×256×1 coordinates and
1,441,792 records. At pp512 it is 2,048×256×1 coordinates and 23,068,672
records, with every coordinate appearing exactly 44 times. With the gpubpf
40-byte aligned ring stride, the pp512 layout occupies 935,329,824 bytes
(892.000031 MiB), below the fixed 1,000 MiB shared-memory budget. Therefore all
44 launches fit even if the collector does not drain concurrently; a 45th
entry would be the first that could find a full per-coordinate ring.

Both collectors now validate record width, nonzero timestamps, exact extent,
coordinate multiplicity, invalid coordinates, segment mismatches, unique
coordinate count, selected launches, and a fail-closed collector gate. The
runner additionally requires equality of every shared observable between the
gpubpf and NVBit paired arms. Each implementation retains its native
transport: per-thread gpubpf rings and NVBit `ChannelDev`.

## Verification completed

- `python3 -B -m unittest -v test_offline.py test_analyze_revision_rq4.py`:
  60 tests passed.
- `python3 -B -m py_compile run_revision_rq4.py analyze_revision_rq4.py
  test_offline.py test_analyze_revision_rq4.py`: passed.
- `patch --dry-run --batch --forward --fuzz=0 -p1` against the pinned
  `bpftime-table1-575/example/gpu/kernelretsnoop` source: passed.
- The first preparation attempt inherited an old binary and `.output` tree
  from the source directory; its apparent success was rejected and is not
  evidence. A second build copied only `Makefile`, `README.md`,
  `kernelretsnoop.c`, `kernelretsnoop.bpf.c`, `vec_add.cu`, and `.gitignore`
  into an empty directory, applied the patch without fuzz, and rebuilt every
  host and BPF artifact against `bpftime-table1-575`. The resulting libbpf
  inspection reports a 32-byte map value. `ldd` reports no unresolved host
  dependency.
- On that fresh binary,
  `kernelretsnoop --self-test-multiplicity-oracle` passed its valid oracle and
  rejected missing-event, swapped-segment, and invalid-geometry cases.
- The same fresh binary rejects a missing ring-entry setting and the oversized
  value 4,294,967,296 with exit status 2 before BPF open/load.
- A clean copied NVBit adapter build completed with CUDA 12.9, NVBit 1.8, and
  `ARCH=sm_120`; `make CXX=g++ test` also passed the launch-clock CPU test.
- Strict read-only OpenCode capacity audit
  `ses_f94a5fc37ffeg4t8L2w4D9Wol0`: pass with the conditions implemented here.
- Strict read-only OpenCode final symmetry audit
  `ses_f948bd452ffegU12dF9ztECv4x`: pass; it found no concrete correctness or
  validity defect in the final runner, analyzer, source, tests, or plans.
- The fixed pp32 and pp512 dry-run matrices passed: respectively five
  correctness plus five timing cells, and five correctness plus fifty timing
  cells. The latter still requires the independent preflight gate.

## Next evidence gate

Run one fresh pp32 preflight block only after both workspace leases are held,
the RTX 5090 is idle, and no unrelated compiler or OpenCode process remains.
Only an independently complete preflight authorizes the disjoint pp512,
10-block full campaign. Build/readiness success alone supports no overhead or
performance claim.
