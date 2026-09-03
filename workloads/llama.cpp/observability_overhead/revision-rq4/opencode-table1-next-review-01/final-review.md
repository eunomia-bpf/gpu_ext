

Review produced from the attached request only. No files, commands, GPU, or network were used. All path-level details not contained in the request remain unverified.

## Scope note
The request names only `workloads/llama.cpp/observability_overhead/revision-rq4` and "the status summaries that directory links." I have no tool access, so those summaries, the gpubpf source, NVBit config, and the exact llama.cpp dispatch path are **unverified**. The two cells below target only the two stated blockers and add no re-run of already-completed formal cells.

## Unverifiable claims (flagged)
- "Exact three-arm histogram correctness closure" — not evidenced in the request; unverified.
- "No admitted seven-arm timing run" — unverified.
- Any statement that baseline/NVBit/BPF currently agree or disagree — not in the request; unverified.
- gpubpf BPF map/ring sizes, the exact launch symbol, NVBit hooks, and RTX 5090 `%globaltimer`↔host mapping availability — all unverified.

---

## Experiment 1 — Lossless exit-observation (repair of BPF exit-event collection)

**Hypothesis.** With lossless BPF exit collection, the per-launch and total kernel-return events observed by BPF equal the ground-truth launch/return set produced by baseline semantics and by NVBit, with zero dropped records.

**Exact controls.**
- Fixed workload cell (same model, prompt, seed, iteration count), single process, exclusive GPU occupancy (pre-check no other GPU work), GPU clocks pinned.
- Four-count reconciliation per arm: baseline count, NVBit launch count, BPF enter count, BPF exit count.
- Per-launch correlation key: `(pid, tid, monotonic sequence)` so each enter pairs to exactly one exit; no implicit merging.
- BPF ring/perf buffer size pinned; `PERF_RECORD_LOST` (or equivalent BPF map-drop counter) read at end of run; probe attach state asserted at start.

**Engagement / correctness gates.**
- G1 Engagement: BPF enter count > 0 AND exit count > 0 (probe actually firing).
- G2 Losslessness: dropped records = 0; enter count == exit count (no orphans) at tolerance 0.
- G3 Per-launch: for every launch index i, exactly one baseline, one NVBit, one BPF enter, one BPF exit; per-launch counts equal across all arms.
- G4 Total: total events equal across baseline, NVBit, BPF, and equal to Σ per-launch.
- G5 No silent drops: any residual difference must be named and attributed (e.g., a launch not yet returned at end-of-run, stream ordering) and explicitly accounted; unexplained difference = 0.

**Failure outcomes.**
- G2 fail ⇒ BPF exit collection is not lossless; ring/pipeline must be fixed; timing barred.
- G3 fail ⇒ pairing bug or dropped/merged events; not admissible.
- G4 fail with G3 passing ⇒ total-accounting bug; fix.
- Exit-before-enter ordering ⇒ exit boundary is wrong; re-derive boundary.

**Minimum condition before performance timing is allowed.** G1–G5 all pass for the exact cell: zero dropped records, exact per-launch and total equality across baseline/NVBit/BPF.

---

## Experiment 2 — Launch-latency clock-domain (repair of host-time vs `%globaltimer` boundary)

**Hypothesis.** The host launch target being timed is the genuine CUDA launch entry (the actual `cudaLaunchKernel`/`cuLaunchKernel` dispatch leaf, not a wrapper/poll/submit helper), and launch latency is valid because both endpoints are expressed in one disclosed common clock domain (or a defensible common interval), so the boundary is fair and not a mixed host-time/`%globaltimer` artifact.

**Exact controls.**
- Host target verification: BPF enter call-stack must show the timed function as the leaf launch call on llama.cpp's dispatch chain; its call count must equal the losslessly reconciled launch count from Experiment 1.
- Clock-domain mapping: a single disclosed mapping between host clock and `%globaltimer` (e.g., calibrated `rdtsc`↔`%globaltimer` offset) with measured offset, residual error, and stated uncertainty; confirm `%globaltimer` availability/semantics on the target arch.
- Symmetric interval: define start (host launch entry) and end (device kernel start / return) identically for every arm; no arm may start or end at a different semantic point.
- Pinned clocks, isolated GPU, single stream, fixed workload, warmup excluded.

**Engagement / correctness gates.**
- G1 Target correctness: timed host function is the genuine launch entry; verified by stack + call-count match to Experiment 1 + no wrapper double-count.
- G2 Domain validity: one disclosed host↔`%globaltimer` mapping with bounded residual error, or the whole interval measured within a single domain; both endpoints in the same disclosed domain.
- G3 Boundary fairness: start/end definitions identical across baseline, gpubpf, and device-observability arms (directly attacks the "unfair/incorrect host-time vs `%globaltimer` boundary" blocker).
- G4 Reconciliation: the set of launches timed equals the losslessly reconciled set from Experiment 1.

**Failure outcomes.**
- G1 fail ⇒ host target is not the real launch function (wrong leaf, includes host overhead); latency invalid; re-target.
- G2 fail ⇒ host time and `%globaltimer` not in a common/disclosed domain with bounded error; measurement unfair/incorrect; mapping must be fixed before timing.
- G3 fail ⇒ asymmetric boundaries across arms ⇒ unfair comparison; barred.
- G4 fail ⇒ timed set ≠ reconciled set; drop.

**Minimum condition before performance timing is allowed.** G1–G4 all pass: real launch target confirmed; single disclosed clock domain (or defensible interval) with bounded calibration error; symmetric boundaries across arms; timed set equals the losslessly reconciled launch set.

---

## Ranking — which repair first
1. **Experiment 1 (lossless exit-observation) first.** It is the losslessness prerequisite and establishes the ground-truth launch/return set and the exact launch/return boundary that Experiment 2's interval must be measured over; the BPF exit pipeline is shared infrastructure for both.
2. **Experiment 2 (clock-domain) second**, proceeding in parallel on the target-verification half (cheap: confirm the leaf launch symbol), but its timing gate stays barred until G1–G4 pass and until Experiment 1's reconciled set is available.

No performance timing cell is admissible until both minimum conditions above are satisfied.
