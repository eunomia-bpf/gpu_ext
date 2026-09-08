# Disk-backed UVM same-address restore: five complete runs

Completed 2026-09-08 on RTX 5090 / CUDA 12.9 / NVIDIA 575.57.08 / Linux
6.15.11-061511-generic. Driver source `c8e2831d`, client `df45d70a`.
Five fresh processes each run seven stages on a 256 MiB allocation. All
five return zero; all stages are retained. No manuscript edits were made.

| Stage | Median elapsed ms | Meaning |
|---|---:|---|
| Initial GPU read | 0.336521 | Full read after initialization copies data to GPU |
| Register | 0.172915 | Attach read-only backing to the managed allocation |
| First offload | 278.299030 | Write backing and release in-memory copies, submit through completion |
| CPU restore | 169.819063 | Full host traversal at the original managed VA |
| Second release | 3.310861 | Release restored copies; backing already exists |
| GPU restore | 170.601143 | Full GPU traversal at the original managed VA, launch through stream synchronization |
| Repeated GPU read | 5.407625 | Same traversal after first restore, without another offload |

These are stage medians over five runs, not paired policy improvements.
The fixed stage order is initial GPU read, register, offload, CPU restore,
release, GPU restore, repeated GPU read. GPU allocation/readback setup is
outside the timed launch-and-synchronize span. O_DIRECT is requested and
the actual descriptor flags are recorded (`0140002`); O_DSYNC is not enabled,
so this is not a crash-durability guarantee. Each offload reports all 65,536
pages on disk, pending zero, error zero. `bytes_expected` is the allocation
size, not a measured transport counter. The test reuses one backing file
across fresh processes and does not establish cold-device-cache performance.

## What this completes, and what it does not

The existing driver can attach a real local file to an explicitly registered
sealed managed allocation, offload it, and service later CPU/GPU reads at
the **same virtual address**. Fault recovery uses CPU staging. No cuFile or
NVMe-to-GPU P2P claim follows. The repeated GPU read is much slower than the
initial GPU read; do not assume restored data has the same residency/cache
state or label this a measured HBM-resident bandwidth result. Source explains
why: `uvm_va_block_select_residency` explicitly selects CPU residency when
a sealed on-disk page has no resident copy, so disk hydration happens before
any possible promotion. GPU access succeeding does not itself prove HBM
promotion. The current primitive has no explicit post-hydration GPU promotion.

This closes the first live five-repeat test of the new disk/UVM restore
primitive. It is not yet an end-to-end LMCache KV integration, an automatic
pressure-triggered BPF placement policy, or a FIFO/native/BPF comparison of
this new primitive. Existing LMCache policy/throughput/p99 reports remain
the evidence for those comparisons and are not replaced by this table.

## Records and earlier attempts

Full completed records:
`raw/disk-uvm-restore-20260908.cxYKb4/full-read-retry.4Nw61Y/`.
They include five client logs, the client's original raw files, build log,
compiled GPU traversal, derived `timings.csv`, command wrapper, and lifecycle
log. Run from the workload directory to reproduce the table:

```sh
python3 gds-control/disk-uvm/analyze_results.py \
  raw/disk-uvm-restore-20260908.cxYKb4/full-read-retry.4Nw61Y
```

Earlier records remain in the parent directory. The first process stopped
at the original multiple-UVM-FD refusal, before timing. Local Qwen implemented
range-query-based descriptor selection (`2b80f0ad`). The subsequent five runs
in `fd-selection-retry.e5NtEs/` completed, but GPU timings did not traverse
the full buffer: compiled code eliminated the loop around a volatile local
sink, retaining only one sample load per CTA. Those raw numbers are retained,
not represented as full GPU restoration. Root's one-line volatile **source**
load fix (`df45d70a`) retains the actual load loop; the table above uses only
the subsequent full-read run. CPU/offload stages from the earlier runs are
also retained, without mixing campaigns or selecting favorable samples.

Final lifecycle: all five exits zero, `DISK_RUN_EXIT=0 RESTORATION_OK=1`.
The saved original UVM was restored; GDS/KV loaders 3146830/3146831 both report
`attached`. Large backing data, executables and modules are excluded from Git.
After collection, the unused 268,435,456-byte temporary backing file and its
empty temporary directory were removed. The file is reproducible from the
client; original logs, small outputs and all performance measurements remain.
