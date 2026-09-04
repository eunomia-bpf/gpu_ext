# Table 1 phase-capacity build readiness — 2026-09-03

Status: **CPU preparation passes; a new real pp32 preflight is authorized.**
This is dependency/setup evidence only. It contains no GPU execution, real
preflight, throughput measurement, or Table 1 result. The failed campaign
`raw/preflight-575-noncross-clock-01` remains failed and unchanged.

## Source and boundary

- Main experiment source: commit `c167256`, including
  [`kernelretsnoop-phase-capacity.patch`](kernelretsnoop-phase-capacity.patch).
- bpftime source: branch `revision/table1-575`, commit `209d978`. The tracked
  `example/gpu/kernelretsnoop` and `runtime/include` paths were clean. The
  existing `table1-preparation/` untracked directory was neither read as source
  nor changed.
- bpftime build:
  `/home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575`, Debug,
  `BPFTIME_ENABLE_CUDA_ATTACH=ON`. Its syscall-server and agent libraries are
  readable.
- Host: Linux `6.15.11-061511-generic`, GNU Make 4.3, Ubuntu clang 18.1.3,
  Ubuntu GCC 13.3.0, bootstrap bpftool 7.7.0 with libbpf 1.7.

The valid preparation used the new directory
`/tmp/table1-kernelretsnoop-readiness-clean.EzvUZC`. It copied only the
kernelretsnoop source files, so neither `.output` nor a prior binary existed.
The declared patch was applied with `patch --batch --forward --fuzz=0 -p1`;
the command exited 0 and touched only `kernelretsnoop.c`. A reverse dry-run
with `--fuzz=0` also exited 0. The disposable copy then received the same
absolute bpftime include paths, omitted `vec_add`, and selected the same rope
symbol as the runner. No tracked bpftime source was modified.

An earlier disposable copy at
`/tmp/table1-kernelretsnoop-readiness.9N8aU1` accidentally inherited `.output`.
Its successful incremental `make` is not used as clean-build evidence. The
source-only preparation above supersedes it.

## Commands and exit status

All commands below are CPU-only. Inherited bpftime injection variables were
removed from the build and self-test environments.

| Check | Ordinary command | Exit | Outcome |
|---|---|---:|---|
| Strict patch | `patch --batch --forward --fuzz=0 -p1 -i /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/kernelretsnoop-phase-capacity.patch` | 0 | Applied to the fresh copy without fuzz. |
| Reverse dry-run | `patch --dry-run --batch --reverse --fuzz=0 -p1 -i /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/kernelretsnoop-phase-capacity.patch` | 0 | The applied change is exactly reverse-applicable. |
| Source schema | `python3 -B` calling `validate_kernelretsnoop_source_schema()` on the fresh copy | 0 | Required block-dimension ABI, requested-entry input, pre-load map sizing, and reporting markers are present. |
| Clean build | `env -u LD_PRELOAD -u BPFTIME_LOG_OUTPUT -u BPFTIME_SHM_MEMORY_MB -u BPFTIME_MAP_GPU_THREAD_COUNT -u BPFTIME_KERNELRETSNOOP_RING_ENTRIES make -j8` | 0 | Built libbpf, bootstrap bpftool, BPF object, skeleton, host object, and host loader from the fresh tree. |
| Dynamic dependencies | `if ldd kernelretsnoop \| rg -q 'not found'; then exit 1; fi` | 0 | No unresolved host-loader dependency. |
| Oracle self-test | `env -u LD_PRELOAD -u BPFTIME_LOG_OUTPUT -u BPFTIME_SHM_MEMORY_MB -u BPFTIME_MAP_GPU_THREAD_COUNT -u BPFTIME_KERNELRETSNOOP_RING_ENTRIES ./kernelretsnoop --self-test-multiplicity-oracle` | 0 | Exact oracle and all three malformed-case rejection checks passed. |
| Missing capacity | `env -u LD_PRELOAD -u BPFTIME_KERNELRETSNOOP_RING_ENTRIES ./kernelretsnoop` | 2 | Failed before BPF open/load with the positive-`uint32` diagnostic. |
| Oversize capacity | `env -u LD_PRELOAD BPFTIME_KERNELRETSNOOP_RING_ENTRIES=4294967296 ./kernelretsnoop` | 2 | Failed before BPF open/load with the same diagnostic. |
| Offline tests | `python3 -B -m unittest -v test_offline.py test_analyze_revision_rq4.py` | 0 | 58/58 tests passed. |
| pp32 dry-run | `python3 -B run_revision_rq4.py --phase preflight --dry-run --tools kernelretsnoop threadhist --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 --gpu-thread-count 22528` | 0 | One block, five correctness cells, five timing cells. |
| pp512 dry-run | `python3 -B run_revision_rq4.py --phase full --dry-run --tools kernelretsnoop threadhist --preflight-dir /tmp/table1-readiness-passed-preflight-placeholder --output-dir /tmp/table1-readiness-full-placeholder --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 --gpu-thread-count 22528` | 0 | Ten blocks, five correctness cells, fifty timing cells; full still requires a separate independently passing preflight. |

The clean host loader is an x86-64 ELF executable and its generated probe is a
Linux BPF relocatable ELF object. Dynamic dependency inspection reported no
unresolved library. These facts establish buildability, not runtime attachment.

## Frozen geometry and memory budget

The one-off check independently evaluated
`32 + slots * (24 + entries * 88)` and compared every returned layout field
with the runner:

| Phase | Slots | Entries/slot | Launches | Events | Shared bytes | MiB |
|---|---:|---:|---:|---:|---:|---:|
| correctness | 22,528 | 256 | 220 | 720,896 | 508,051,488 | 484.515656 |
| pp32 timing | 32,768 | 16 | 44 | 1,441,792 | 46,923,808 | 44.750031 |
| pp512 timing | 524,288 | 16 | 44 | 23,068,672 | 750,780,448 | 716.000031 |

The fixed 1,000 MiB segment is 1,048,576,000 bytes. The pp512 timing layout
leaves 297,795,552 bytes (283.999969 MiB) of headroom. Sixteen entries are
buffering capacity, not an allowed-loss threshold: any OOB/full/bad-size/other
drop, dirty or pending record, second-drain event, invalid coordinate, wrong
geometry, or gpubpf/NVBit event or launch disagreement still rejects the cell.

The first hand-entered arithmetic assertion exited 1 because its otherwise
unused expected correctness-byte literal was transcribed incorrectly. It made
no source change. The subsequent formula-derived command above exited 0 and
checked all layout fields exactly; this was a readiness-command transcription
error, not experiment output.

The oracle self-test observed exactly 720,896 events, 220 launches, and 22,528
coordinates; coordinate multiplicities were 1,024 at 220, 1,024 at 44, 20,480
at 22, and zero otherwise. Missing-event, swapped-segment, and invalid-geometry
fixtures were all rejected.

## Independent OpenCode review

OpenCode 1.18.27 ran as a fresh process on CPU 18 with model
`opencode/ling-3.0-flash-fin-free`; snapshots and sharing were disabled and all
write, edit, shell, web, and task tools were denied. Session
`ses_f94e657dbffeYf3dtj6TWULgx3` exited 0 with verdict **READY** and no blocker.
It separately checked strict application, pre-load map sizing, fail-closed
capacity parsing, all three geometries, the 1,000 MiB boundary, and the scope of
the evidence. Its authorization is only for the next real pp32 preflight.

Next command: run a fresh two-tool pp32 preflight. Promotion to pp512 full is
still barred until that campaign completes every correctness/timing cell and
the independent analyzer reports `complete=true`.
