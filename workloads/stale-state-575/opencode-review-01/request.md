# Read-only source-boundary review request

Review every attached file for Reviewer D's stale cross-layer state / UVM
thrashing experiment. This is deliberately a CPU/source boundary, not a live
GPU implementation or a result claim.

Return exactly one leading verdict line:

- `READY AT DECLARED BOUNDARY`
- `REQUIRED FIXES`

Then give concise evidence for the verdict. Treat as required-fix issues any
case where the attached sources:

1. imply that live native-vs-BPF execution is possible despite the documented
   missing driver-owned shared snapshot/native consumer/diagnostic interface;
2. access a GPU, filesystem path, lease, or process from `dry-run`, or allow
   `live` to pass the missing-interface gate;
3. let the formal 21-cell, three-block analysis proceed without a distinct
   complete seven-cell preflight, exact matrix identities, or fair within-block
   comparisons (native vs BPF at equal delay; delayed vs fresh within one
   implementation);
4. accept synthesized/proxy UVM or policy counters, incomplete phase truth,
   early publication, decision timestamps that cannot join to host truth,
   missing driver effects, numerical mismatches, monitor loss, foreign compute,
   unsafe lifecycle, or incomplete cleanup;
5. fail to retain 0/100/1000 ms decision ages, wrong-phase decisions, faults,
   migrations, prefetch, discard, thrashing, eviction, throughput/timing, and
   complete numerical checking;
6. make a positive-result assumption instead of retaining valid negative rows;
7. contain a source/ABI/logic issue that prevents the CPU tests, monitor build,
   CUDA source build, or future coordinator integration described by the
   boundary.

Evidence already obtained without running a GPU:

- `make test-offline`: C policy assertions plus 13 Python tests passed.
- `make stale_state_workload`: compiled with nvcc under C++17.
- real monotonic CPU relay preflight: all nine samples published no earlier
  than eligibility, with median ages approximately 0 ms, 100.15 ms, and
  1000.16 ms; this is explicitly labeled dependency evidence, not experiment
  evidence.
- both dry-run stages returned plans; `live` returned status 2 before creating
  its requested output path.

Do not request or perform live GPU execution. Do not weaken the declared
boundary merely to produce an executable live command.
