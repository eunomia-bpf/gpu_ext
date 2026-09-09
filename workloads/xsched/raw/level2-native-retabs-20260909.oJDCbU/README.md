# XSched native cuXtra sm_120: first completed resume cell

2026-09-09: the captured RET.ABS candidate builds and one native-blob
bring-up cell completes. All six workload processes exit zero, with 400
LC and 800 BE kernel service records. Each of the four BE process logs
contains one type-2 resume launch. The previous CUDA 700 resume failure
does not occur in this cell. This is one successful run, not a completed
randomized baseline/native/BPF performance comparison.

## Change and build

Main checkout base: `07660def`, plus the captured local-model candidate
in [ldc_patcher.cpp](ldc_patcher.cpp). The generator replaces the final
three resume instructions with LDC into R21/R20 from the debugger entry
slot and RET.ABS.NODEC R20, following the upstream sm_70/sm_86 transfer
form. It no longer passes the full entry address to CALL.REL.NOINC R2.
The original guardian/cooperative preemption, parameter-window expansion,
argument-layout registry and native cuXtra actuator remain in use.
This is the original actuator adapted to sm_120, not an untouched upstream
binary and not the separately measured NVBit-actuated policy port.

The first [build attempt](build-patcher.log) used an older guardian cubin
without the retained conditional-exit marker and failed during extraction.
[Its command](build-patcher-old-input.sh) remains. This was an input-path
mistake, not a GPU attempt or evidence against the new transfer.

[build-patcher.sh](build-patcher.sh) then rebuilt both captured current
stub sources with CUDA 12.9 and the generator with g++. It completed at
10:45:48 PDT: guardian prefix 640 bytes and resume prefix 336 bytes,
ending in RET.ABS at offset 0x140. The generated
[array header](xg_sm120_guardian_arrays.h) and [build log](build-stubs.log)
are retained. [build-hal.sh](build-hal.sh) configured the existing isolated
HAL with that new header and completed install at 10:46:58 PDT
([log](build-hal.log)). Binaries and build caches are not published here.
Both builds and the run held the shared GPU and struct-ops leases.

## One-cell observation

[Run command](run.sh), [run log](run.log),
[protocol](cells/protocol.json), [cell result](cells/block-01-native_blob/result.json)
and [summary](cells/summary.json) are retained with all worker logs.

RTX 5090, driver 575.57.08; two LC and four BE processes, four streams
each (24 XQueues), 50 kernels per stream, 9,511,106 recurrence iterations,
340 blocks and 256 threads. The native shim is selected through
LD_LIBRARY_PATH; no NVBit tool or BPF selector is loaded in this cell.

| Metric | Native cuXtra, one cell |
| --- | ---: |
| LC GPU-service p99 | 960.855232 ms |
| LC GPU-service mean | 488.820840 ms |
| BE GPU-service p99 | 1410.813728 ms |
| BE throughput | 10.163525 kernels/s |
| Completed LC / BE service records | 400 / 800 |
| Workload process exits | 6 / 6 zero |

These are GPU-service and host-elapsed measurements, not complete
arrival-to-completion or queueing latency. Do not compare this single
observation with the historical five-block policy-port medians as a paired
effect. No baseline or BPF control was measured in this attempt.
The runner exited zero at 10:49:08 PDT; afterward no compute process
remained and nvidia-smi reported 0% utilization and 1 MiB used.

The earlier [constant-bank attempt](../level2-native-constant0-20260909.oAKEda/README.md)
remains a failed resume attempt. This run resolves its observed bring-up
failure, not the remaining multi-arm comparison or portable full-runtime
reproduction. The absolute paths in these scripts document this execution;
they are not a fresh-machine installer.
