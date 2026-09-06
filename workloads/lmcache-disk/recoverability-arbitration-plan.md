# Recoverability-Aware Arbitration Plan — LMCache/UVM-KV + gpubpf

## Reviewer-facing execution decision — 2026-09-06

The paper-facing experiment must exercise LMCache's real local-disk backend;
an unrelated UVM policy run beside LMCache is insufficient. The final matched
campaign uses five arms:

1. `recompute`: no-cache control;
2. `lmcache_cpu`: original LMCache CPU tier;
3. `lmcache_disk_uvm_kv`: original LMCache local-disk tier with default UVM;
4. `lmcache_disk_uvm_kv_native_recovery`: local disk plus the recoverability
   policy implemented as a native selector; and
5. `lmcache_disk_uvm_kv_gpubpf_debt_range`: local disk plus the identical
   decision implemented through gpubpf.

The native and BPF policy arms consume the same per-range tenant, generation,
lifecycle, disk-durability, recovery-cost, and next-use metadata and use the
same bounded victim order. LMCache remains responsible for actual disk store
and restore; the policy selects UVM/HBM residency using the fact that a range
has a durable disk copy and its expected restore cost. A completed disk-store
barrier marks the range durable and inactive, while reuse marks it active.
This makes disk state an input to the resource decision and isolates the cost
of the BPF mechanism from the policy benefit.

Report warm TTFT, request/s, output-token/s, available LMCache disk read/write
bytes or operation totals, and available UVM migration/eviction counters. Run
five rotated complete blocks and retain every performance result. These
observations are recorded, not used as admission gates; do not add clock,
preflight, correctness, or result-filtering gates to this campaign.

Design plan only, no implementation. Replaces the single-largest-allocation durable
bool with a per-range semantic ABI and a bounded eviction order, defines the two
experiments that separate policy from mechanism, adds a compact discard-isolation
matrix (Section 5.3), and reframes the stronger design as verified recovery
contracts over the existing 575 `UvmDiscard` API (Section 6). Built on
[results-575-gpubpf-kv-range-canary-20260906.md](results-575-gpubpf-kv-range-canary-20260906.md)
and `extension/eviction_debt.bpf.c`, `eviction_debt_model.h`, `eviction_debt.c` (loader).

## 1. Why the current durable bool is a mechanism canary, not a semantic signal

Current state: one `kv_pool_range` ARRAY slot holds the largest `uvm_kv_malloc`
`[start, end, tgid]` (largest-wins; a freed pool stays recorded until a larger
allocation replaces it); one global `DEBT_CONFIG_DISK_DURABLE` bit is sampled at
activation and flipped retroactively by the loader's `w` key; per-chunk state is 8
bytes (`owner_pid, debt, accesses, is_kv, disk_durable`).

Why this is only a canary:

- **Phase flag, not provenance.** The bit says "the warm phase happened", not "this
  page is on local NVMe". LMCache O_DIRECT stores are per cached block under a
  16 GiB cap, and the current mechanism cannot map logical LMCache blocks to UVM
  subranges, so the loader has no per-page provenance (the coarse canary records
  "exact LMCache chunk -> UVM page identity unavailable").
- **Coarse coverage.** Largest-wins single range: a pool split across smaller
  allocations is only partially covered (canary: 13 of 624 tracked chunks in-range).
- **Stale generation.** A freed pool stays recorded; a new allocation inside the old
  range is still treated as durable.
- **No lifecycle, tenant, or recovery cost.** No active/inactive/pinned; the debt
  cap is a recency proxy, not a next-use deadline; one tgid; every durable chunk
  looks equally cheap.

The 2026-09-06 canary confirms this: the cross-layer path engages (range capture,
sampling, retroactive marking of exactly the 13 in-range chunks) while `saved=0`,
`evicted=0`, pressure 0 — the policy machinery never acted. Under real pressure the
single bool would protect or evict on range membership alone, too coarse to
attribute any delta to policy.

## 2. Per-range semantic ABI

Two tiers. The **MVP** is a pool-range lifecycle table using only the existing
struct_ops hooks and the two allocator probes. **Full per-block semantics** is out
of MVP scope: a true per-block ABI requires the LMCache connector/UVM-KV plugin to
publish tensor VA spans at save/load/free, driving per-subrange tier/cost records;
the range-wide store barrier is an approximation, not sufficient provenance.

```c
/* BPF_MAP_TYPE_HASH, key u32 range_id, max_entries 16 */
struct debt_range_record {
    u32 tenant;        /* owner tgid of the pool */
    u32 generation;    /* monotone per tenant; bumped on re-allocation */
    u64 start;         /* [start, end) */
    u64 end;
    u8  lifecycle;     /* RANGE_IDLE | RANGE_ACTIVE | RANGE_INACTIVE | RANGE_PINNED */
    u8  backing_tier;  /* TIER_NONE (HBM only) | TIER_DISK_LOCAL */
    u8  recovery_cost; /* RECOVER_CHEAP (local NVMe read) | RECOVER_SLOW | RECOVER_LOSS */
    u8  _pad;
    u64 next_use_deadline_ns; /* 0 = unknown */
    u64 last_update_ns;       /* last loader semantic write */
};

/* Per-chunk state, BPF hash value (chunk_ptr -> state): 16 bytes, four u32, no padding. */
struct debt_chunk_state {
    u32 owner_pid;
    u32 range_id;   /* 0 = no tracked range */
    u32 range_gen;  /* generation snapshot at activation */
    u32 flags;      /* bits 0-7 debt; 8-15 accesses (saturated);
                       16-23 lifecycle: CHUNK_INACTIVE | CHUNK_ACTIVE | CHUNK_PINNED */
};
```

MVP population (control plane in the loader; bounded hot path):

- **Capture.** The existing uprobe/uretprobe pair on `uvm_kv_malloc` stops
  largest-wins: each successful allocation inserts a record with a loader-assigned
  `range_id` and fresh `generation`.
- **Retire.** An entry uprobe on `uvm_kv_free` (`ptr` is an argument; the allocator
  exports the symbol and the plugin binds it as `FREE_FN`) retires the record whose
  `start` matches; the next allocation for that tenant gets a new `generation`.
  The loader also retires records when the owner tgid exits. Bounded by
  `max_entries = 16`.
- **Tier/cost.** After the runner's cold-store barrier the loader writes
  `TIER_DISK_LOCAL` + `RECOVER_CHEAP` into the tenant's range record (all-or-nothing
  per range, like the canary warm flag, but recorded per range and per generation).
- **Deadline/lifecycle/pin.** Loader keys `d <range_id> <deadline_ns>` (monotone
  nondecreasing; runner sets it from the known warm-phase schedule),
  `l <range_id> <state>`, and `p <range_id> 1|0`.
- **Activation/access.** `gpu_block_activate` looks up the range table by
  `(va_start, owner tgid)` and stamps `{range_id, range_gen}` (no live match leaves
  `range_id = 0`, default path); `gpu_block_access` sets `CHUNK_ACTIVE` on observed
  reuse; sub-cap second-chance save is unchanged.

## 3. Bounded verifier-friendly eviction order

All new work fits in the existing unrolled 8-chunk head walk in `gpu_evict_prepare`
plus at most one range-table lookup per chunk; no new kernel hooks for this MVP. No
floats. No reorder requests from `evict_prepare` (container_of-derived pointers are
untrusted to the verifier; saves remain in `gpu_block_access`).

For each walked tracked chunk, in fixed order:

1. **Ignore (stale generation).** `chunk.range_gen` differs from the live record's
   `generation`, or the record is retired/absent: no debt, no victim; count
   `ignored_stale`.
2. **Protect.** Record `lifecycle == RANGE_PINNED`, chunk `CHUNK_ACTIVE`,
   `recovery_cost == RECOVER_LOSS`, or the record's next-use deadline is soon:
   `next_use_deadline_ns != 0 && next_use_deadline_ns <= now + protect_window`:
   no debt, no victim; count `protected`.
3. **Eligible.** Record `backing_tier == TIER_DISK_LOCAL` with chunk
   `CHUNK_INACTIVE`, or default debt-marked chunks at the cap.

Victim selection among eligible chunks: choose the **MIN** of the lexicographic
tuple

```
victim_key = (rec_rank, U64_MAX - deadline_key, last_touch, chunk_ptr)
rec_rank:     RECOVER_CHEAP = 0, RECOVER_SLOW = 1 (RECOVER_LOSS never reaches this
               step; it is protected in step 2)
deadline_key: record next_use_deadline_ns, with 0 (unknown) mapped to U64_MAX
last_touch:   last observed gpu_block_access timestamp (u64 ns)
```

MIN means: lower recovery cost first, then farthest next use first (unknown =
never, evicted before any known later deadline), then oldest touch first, with
`chunk_ptr` as total-order tie-break. All unsigned integer compares; 2 map reads
per chunk x 8 chunks, constant to the verifier. The chosen chunk is dropped from
tracking and left at HEAD for the kernel to evict; counted `victims_disk_local`
when disk-backed, `victims_none_tier` otherwise. The prefetch-suppression pressure
gate is unchanged.

## 4. What gpubpf uniquely contributes (framing)

- **Driver-global arbitration.** The struct_ops callbacks run inside `nvidia_uvm`
  PMM decision points and see UVM pages of every process attached to that GPU; the
  tenant/generation fields make cross-process arbitration explicit instead of
  one-tenant-by-accident.
- **Verified hot-swappable policy.** A verifier-checked BPF program attaches to the
  live driver through struct_ops with bounded maps and loops; swapping policy is
  attach/detach, not a driver or GPU-memory-API change.
- **Recovery contract over an existing driver capability, not a new primitive.**
  The checked-out 575 UVM already implements `UvmDiscard`
  (`kernel-open/nvidia-uvm/uvm.h`): discarded pages are selected for eviction
  *before* pages that require copies, and eviction of discarded pages does not
  copy their contents; with `UVM_DISCARD_FLAGS_UNMAP` (`uvm_types.h`) later
  writes are guaranteed to land and clear the discard status, and reads after a
  write observe the written data. What the driver does *not* provide is a
  policy-controlled entry point: the gpubpf struct_ops surface
  (`uvm_bpf_struct_ops.h`) exposes only prefetch / block-activate / block-access /
  evict-prepare, and the transition validator's PMM request space
  (`nv-gpu-transition-validator.h`) is chunk destination/position only — no
  discard transition. The stronger design is therefore a set of **verified
  recovery contracts** (Section 6): (i) after a durable store, the LMCache
  connector publishes each block's exact UVM range, tenant, generation,
  lifecycle, and restore capability; (ii) a validated asynchronous gpubpf
  transition may request discard only for ranges that are inactive,
  disk-durable, generation-current, and unpinned; (iii) LMCache must restore
  (write) a discarded range before the model reads it; (iv) stale or missing
  metadata falls back to native UVM, and a failed restore falls back to
  ordinary LMCache recovery/recompute.
- **Not claimed here:** KV-reuse prediction (debt/second-chance is a standard
  recency heuristic), tiering (LMCache already tiers to disk; CachedAttention and
  ECHO also tier KV), file-backed GPU memory (UVM migration and GAIA precede it),
  or the discard primitive itself — `UvmDiscard` and `UVM_DISCARD_FLAGS_UNMAP`
  are existing 575 driver APIs. The scope difference is *where and for whom* the
  eviction/recovery decision is made: inside the driver's page lifecycle, across
  tenants, under a hot-swappable verified policy, against an explicit recovery
  contract.

## 5. Experiments

Common: RTX 5090, driver parameter 575.57.08, CUDA 12.9, vLLM 0.27.1, LMCache
v0.5.4 (revision in the cell environment), Qwen3-30B-A3B-FP8, `--enforce-eager`,
`--max-model-len 4096`, `--max-num-seqs 1`, native prefix caching off, port 18080,
`run_uvm_kv_perf.py` (kind `lmcache_uvm_kv_perf`), no retry, no result filtering;
every number preserved per cell.

**Pressure setup (both experiments).** Increasing the prefix count alone cannot
create pressure: vLLM allocates its KV pool once at startup. Pressure requires two
explicit knobs: (a) an **explicit KV pool size** — the `--kv-cache-memory-bytes`
server option exposed by this checked-out vLLM (`vllm/engine/arg_utils.py`),
set above the HBM-fit KV footprint so UVM must migrate pool pages (a
`--max-num-seqs`/context configuration reaching the same oversubscription is an
alternative only); and (b) a **controlled UVM pressure tenant `U`** — a separate
process holding a fixed UVM pool (e.g. 2 GiB) with periodic touches, whose size is
the pressure-level knob. `U` is identical in every arm.

### 5.1 Experiment A — minimal three-arm single-tenant pressure

T1 = the KV workload: the cold phase populates `P` prefixes (each with its store
barrier); the timed warm phase reuses all `P`. A recorded, non-gating preflight
measures pool bytes (plugin `uvm_get_allocated_bytes`, `UVM_KV_PLUGIN_COUNTERS=1`)
against HBM available to KV at the chosen pool cap plus `U`, and fixes `P` and the
pool cap; both are recorded in the campaign params before any arm runs.

| arm | name | what it is |
|---|---|---|
| A0 | `lmcache_disk_uvm_kv` | pool + LMCache disk, no gpubpf (default kernel UVM) |
| A1 | `lmcache_disk_uvm_kv_gpubpf_debt` | current policy (largest-range durable bool) |
| A2 | `lmcache_disk_uvm_kv_gpubpf_debt_range` | MVP per-range ABI + order (Sections 2-3) |

Arms run contemporaneously, rotated complete blocks, identical prompts.

Metrics per cell: warm TTFT median/p95/max ms, req/s, out tok/s, ok/failed; loader
ready/returncode/tracked/KV-range/durable, per-PID `active/used/saved/evicted`,
debt pressure, and new counters `ignored_stale`, `protected`, `victims_disk_local`,
`victims_none_tier`; per warm request the LMCache retrieved tokens (existing
server-log parse) with refault cost as the warm TTFT delta vs A0 (proxy — exact
refault latency needs a UVM migration trace, out of scope); HTTP 200 and exact
retrieved-token count per warm request.

Attribution: **A1 vs A0** = mechanism + coarse semantics (driver-global hooks
acting on the single bool). **A2 vs A1** = policy semantics only (identical
mechanism; per-range ABI and order change). **A2 vs A0** = combined.

Recorded observations (non-gating): loader ready and returncode 0 in A1/A2;
`saved` or `evicted` > 0 in A1 and A2 (pressure actually reached the policy);
server returncode 0 in all arms. The campaign applies no performance filtering;
deltas are reported, not gated. Falsification: if A1 matches A2, semantic
refinement is invisible at this pressure; escalate to B or raise `U`.

### 5.2 Experiment B — co-location (higher value)

The only experiment where driver-global arbitration is load-bearing: T1's HBM
pressure must decide whose pages migrate.

- **T1 (foreground):** Experiment A's workload.
- **T2 (background, semantic):** a second UVM tenant with its own small KV pool,
  fixed-period cyclic access, and a known deadline (`next_use_deadline_ns` =
  period). Fallback: a second small vLLM instance with a periodic short prompt
  stream.
- **U:** the same pressure tenant as in A.

B additionally needs the runner to launch T2 and `U` alongside the T1 server in a
cell (new runner option; arm definitions unchanged).

| arm | name | policy visibility |
|---|---|---|
| B0 | both tenants, no gpubpf | default kernel UVM |
| B1 | `lmcache_disk_uvm_kv_gpubpf_debt` | tracks only T1's tgid range; T2 untracked |
| B2 | `lmcache_disk_uvm_kv_gpubpf_debt_range` | range table for both tenants; per-tenant generation/deadline/pin |

Metrics per cell: T1 warm TTFT median/p95; T2 per-cycle access latency and miss
count (T2 harness records its own page-access times); per-PID
`active/used/saved/evicted` (existing `pid_chunk_count` is per-PID, so tenants are
separated) plus the Section 3 counters; HBM residency split by tenant via per-PID
`current_count` at timed-phase start; T1 deadline-miss rate = fraction of timed
warm requests whose prefix is not HBM-resident at request start (loader per-PID
counts + warm-phase timestamps).

Attribution: **B2 vs B1** = tenant-aware semantic arbitration on the same mechanism
(tenant/generation/deadline fields vs single-tenant blind spot). **B1 vs B0** =
mechanism under co-location (single-tenant coarse policy vs default kernel).
**B2 vs B0** = combined. B vs A with the same T1 configuration = cost of sharing
the GPU, independent of policy choice.

Recorded observations (non-gating): both tenants live through the timed phase in
all arms; B2 shows non-zero tracked counts for both PIDs; T1 `evicted` > 0. No
performance filtering is applied.

### 5.3 Experiment D — discard isolation matrix (stretch arms labeled)

Compact matrix isolating the discard mechanism from the policy that requests
it. D0 and D1 run on the current 575 driver: `UVM_DISCARD` is an existing
ioctl (`uvm_ioctl.h`), so D1 needs connector-side publication of exact block
spans, not a driver change. D2 and D3 require the stretch-level B driver bridge
(Section 6) and are **stretch-only**; the current code implements neither, and
no arm claims first use of discard.

| arm | name | discard trigger |
|---|---|---|
| D0 | `discard_native_uvm` | none (native UVM, LMCache disk pool) |
| D1 | `discard_lmcache_direct` | the LMCache connector itself calls `UVM_DISCARD` with `UVM_DISCARD_FLAGS_UNMAP` on its own inactive, durable blocks after the store barrier |
| D2 | `discard_gpubpf_mechanic` | gpubpf policy requests the *same discard action set* through a validated asynchronous transition (B bridge) |
| D3 | `discard_gpubpf_semantic` | gpubpf semantic policy (Sections 2-3 range ABI: tenant, generation, lifecycle, pin, deadline, restore capability) selects which ranges to discard |

Common: the A/B pressure setup (explicit pool cap plus pressure tenant `U`),
the same `P` prefixes and timed warm reuse; D2/D3 additionally run B's second
tenant T2 for the foreground-SLO metric. A recorded, non-gating preflight for
D1+ probes `UVM_DISCARD` on a scratch pool subrange and confirms the
connector's block spans lie in UVM-managed VA ranges (otherwise
`NV_ERR_INVALID_ADDRESS`).

Metrics per cell:

- Warm TTFT median/p95 and throughput (req/s, out tok/s).
- **GPU-to-CPU eviction bytes avoided** vs D0: total bytes UVM migrated to host
  in the timed phase (per-PID PMM counters plus plugin byte counters); the
  predicted effect of discard is that durable discarded pages never copy to
  host.
- **Discard/refill counts**: `UvmDiscard` invocations and
  applied/rejected transition outcomes (connector-side in D1; driver-side in
  D2/D3), restore writes into discarded ranges (refill), and faults observed
  on discarded ranges (indeterminate-data hazard indicator: must be zero under
  the restore-before-read contract).
- **Foreground SLO under a second tenant** (T2 present): T1 warm TTFT p95 and
  deadline-miss rate, T2 per-cycle miss rate; D3 vs D0/D1 isolates
  tenant-aware discard selection.

Attribution: **D1 vs D0** = discard mechanism alone (no policy). **D2 vs D1**
= the same action, policy-driven: cost and behavior of the validated
asynchronous path. **D3 vs D2** = semantic selection. **D3 vs D0** = combined.
Recorded observations (non-gating): all arms HTTP 200 with exact
retrieved-token counts matching D0 — the UNMAP contract makes any read served
from a discarded range before its restore detectable as wrong content — and
D1/D2 show discard count > 0 with eviction bytes avoided > 0.

## 6. Stretch: UvmDiscard-backed recovery contracts (A / B / C)

Verified source fact in this checked-out 575 UVM (`kernel-open/nvidia-uvm/uvm.h`,
`UvmDiscard`): when a VA range is discarded, read-duplicated copies are
collapsed, *discarded pages are selected for eviction before pages that require
copies during eviction*, and *if discarded memory is evicted, the data in the
pages backing the evicted range is not copied to the eviction destination* and
the discard status is not cleared. With `UVM_DISCARD_FLAGS_UNMAP`
(`uvm_types.h`) the discarded pages are unmapped so that *later writes are
guaranteed to land and remove the discard status*, and reads after a write
observe the written data. Discard may happen concurrently with memory
migrations, so the caller must serialize page accesses and/or migrations
against the discard. The current gpubpf struct_ops interface
(`uvm_bpf_struct_ops.h`) does not expose discard as a policy transition: its
PMM request space is chunk destination/position only
(`nv-gpu-transition-validator.h`). No new fault handler is needed for the
discard itself: the existing UVM fault/migration path plus the
`UVM_DISCARD_FLAGS_UNMAP` write semantics is what restores a range. What is
missing is a validated policy-controlled trigger and precise block-range
publication.

Three levels, in increasing scope:

- **(A) Bounded semantic eviction ordering — the current plan.** Sections 2-3:
  the per-range ABI and verifier-friendly victim order over the existing
  struct_ops hooks. No new driver surface, no discard. This is what Experiments
  A and B (Sections 5.1-5.2) exercise.
- **(B) UvmDiscard-backed prototype — near-term stretch.** A driver bridge in
  the driver-bridge-v1 pattern (driver-owned snapshot, append-only struct_ops
  op, validated transition) through which a validated asynchronous gpubpf
  transition may request discard. The contract:
  1. **Publication.** After a durable store, the LMCache connector publishes
     each block's exact UVM subrange, tenant, generation, lifecycle, and
     restore capability (per-block VA spans at save/load/free; the range-wide
     store barrier of Section 2 is an approximation, not sufficient provenance
     for discard).
  2. **Eligibility.** A transition may request discard only for ranges that are
     `RANGE_INACTIVE` (not active, not pinned), `TIER_DISK_LOCAL`,
     generation-current (record generation matches the live allocation), and
     whose chunk is not `CHUNK_PINNED`. Stale, missing, or absent metadata, or
     a failed snapshot validation, is `NOOP_STALE`/rejected and falls back to
     **native UVM** (ordinary copy-requiring migration).
  3. **Restore-before-read.** LMCache must restore (write) a discarded range
     before the model reads it: a read of an UNMAP-discarded range before the
     restore returns indeterminate data. The connector gates model reads on
     restore completion for the affected blocks.
  4. **Restore failure.** If the restore fails (e.g. the O_DIRECT object is
     gone), the connector falls back to **normal LMCache recovery/recompute**
     and never reads the discarded HBM range; the UVM side needs no special
     path.
  5. **Serialization.** The bridge serializes the `UvmDiscard` call against
     concurrent access/migration of the range, per the `UvmDiscard` caller
     responsibility.
  B is **not implemented in the current code**; D2/D3 (Section 5.3) are
  stretch-only until the bridge and the connector publication land.
- **(C) Direct storage-to-GPU refill — optional future.** Serve the next fault
  on a discarded range by streaming the LMCache O_DIRECT file directly into
  HBM, bypassing the CPU-staged restore. This has no verified path in the 575
  source tree (there is no storage-backed UVM fill today); it is a long-term
  option, out of scope for this plan, and no experiment here claims it.

Claim discipline for this section: `UvmDiscard` and `UVM_DISCARD_FLAGS_UNMAP`
are existing driver capabilities; nothing here claims first use of discard in
serving, and nothing claims the current code implements B or C.

## 7. Literature boundary

Titles and stable publication URLs only. The scope-difference column states only
where gpubpf's decision point differs; no priority or novelty claims.

| Work | Title | Venue / URL | What it establishes | Scope difference for gpubpf |
|---|---|---|---|---|
| KVCache Cache in the Wild | KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider | USENIX ATC '25, <https://www.usenix.org/conference/atc25/presentation/wang-jiahao> | Production-scale KV reuse characterization; workload-aware eviction at the serving-layer cache | Eviction acts on cache blocks inside the engine; UVM page lifecycle unmodified |
| Tiered Memory Beyond Hotness | Tiered Memory Management Beyond Hotness | USENIX OSDI '25, <https://www.usenix.org/conference/osdi25/presentation/liu> | Amortized offcore latency (AOL); SOAR/ALTO allocation and migration regulation for host tiered memory | Governs host DRAM tiers; the tier governed here is UVM-managed GPU HBM |
| CachedAttention | Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention | USENIX ATC '24, <https://www.usenix.org/conference/atc24/presentation/gao-bin-cost> | Hierarchical KV caching across memory/storage media; layer-wise preloading; scheduler-aware fetch/eviction | Hierarchy placement orchestrated by engine and scheduler; page movement uses the standard OS/driver path |
| AsymCache | Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving | arXiv:2606.02964, <https://arxiv.org/abs/2606.02964> | Computation-latency-aware KV residency scoring in the engine | In-engine policy over KV blocks; GPU pages move through the unmodified allocator path |
| mzCache | mzCache: On-Device LLM Memory Management under Multitasking | arXiv:2609.01338 (MobiCom '26), <https://arxiv.org/abs/2609.01338> | On-device LLM memory management under multitasking | On-device single-OS page management; no driver-level verified hot-swappable policy across UVM pages |
| ECHO | ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs | USENIX OSDI '26, <https://www.usenix.org/conference/osdi26/presentation/liu-guangda> | KV offloading with lossless prefetching inside the serving system | Offload/prefetch orchestrated in-engine; driver-global page decision points not involved |
| GAIA | GAIA: An OS Page Cache for Heterogeneous Systems | USENIX ATC '19, <https://www.usenix.org/conference/atc19/presentation/brokhman> | OS page cache spanning CPU and GPU memory; mmap into the GPU address space | File-backed page cache via OS-side integration; not a verifier-checked policy at the driver's UVM decision points |
| cachebpf | Cache is King: Smart Page Eviction with eBPF | arXiv:2502.02750, <https://arxiv.org/abs/2502.02750> | eBPF-customized Linux page-cache eviction without kernel modification | Host page-cache policy in kernel mm; the target here is nvidia_uvm PMM decision points with the per-range semantic ABI as state |
| eBPF-mm | eBPF-mm: Userspace-guided memory management in Linux with eBPF | arXiv:2409.11220, <https://arxiv.org/abs/2409.11220> | eBPF hook in the Linux page-fault path for userspace-guided page-size/promotion policies | CPU fault-path hooks for host pages; the hooks here are the driver's UVM PMM decision points, shared across GPU tenants |

## 8. Claim discipline

- No "first"/"novel"/"state-of-the-art" claim: all literature statements are scope
  differences, and the gpubpf contribution is stated as the arbitration point plus
  the verified hot-swappable policy model.
- The canary results are quoted as mechanism-engagement evidence only; no
  performance win/loss is claimed from non-contemporaneous cells.
- The discard framing is scoped to the existing 575 driver API
  (`UvmDiscard` / `UVM_DISCARD_FLAGS_UNMAP` in `uvm.h` / `uvm_types.h`):
  no "first" claim on the primitive, and no claim that the current tree
  implements B or C.
- Stretch items are labeled as such and excluded from the experiment claims;
  D2/D3 (Section 5.3) are stretch-only until the B bridge lands.
