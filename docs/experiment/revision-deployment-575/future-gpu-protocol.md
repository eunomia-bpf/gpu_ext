# Frozen protocol for GPU deployment claims

Status: **not run in this audit**. The current bpftime build has CUDA attach
disabled and the execution environment was intentionally CPU-only. This
protocol must produce new raw observations before any GPU attach-latency or
SASS claim is restored.

## A. PTX device-hook deployment

Use bpftime revision `d6316fa73edaac4fdfe21b89d4470da6cd9b8ae8` and the
`example/gpu/threadscheduling_dynamic_hook` target and loader. Build a separate
CUDA-enabled tree; do not reuse the CPU-only build audited here.

Record, before any trial: host CPU, kernel release, Yama policy, GPU model,
driver version, CUDA toolkit version, selected CUDA device, bpftime source
revision, complete build options, and exact target/loader commands. Use a
dedicated test account or an isolated machine. If the target opts in with
`PR_SET_PTRACER_ANY`, report that fact rather than describing the route as
generally unprivileged.

Run three warm-up pairs, discard them, then run 20 measured pairs. Alternate
the order by pair: preload/attach for odd pairs and attach/preload for even
pairs. Each trial must use a unique shared-memory name, start from no agent in
the target, and use the same vector dimensions, device, power policy, and
background-load condition.

Emit machine-readable monotonic timestamps from one coordinator:

1. `t0`: immediately before invoking the deployment route;
2. `t_agent`: agent IPC ready in the target;
3. `t_policy`: loader confirms that the intended link is installed;
4. `t_event`: loader receives the first valid device event for the trial ID;
5. `t_end`: route command completes or the trial times out.

The historical metric, if retained, must be defined as `t_event - t0`.
Also report `t_agent - t0` separately so injection and device initialization
are not conflated. Preserve every trial row. Report median, min--max, and P95
only after showing the failure count; do not silently omit timeouts.

A trial passes only when all gates hold: route exit status is zero, target
remains alive, agent IPC appears, the expected link is present, at least one
event contains the correct trial ID and legal block/thread coordinates, the
vector result matches the uninstrumented result, and cleanup removes the
trial's processes and shared-memory object. Use a fixed 30-second timeout per
phase. Any missing gate makes the trial a failure, not a censored latency.

This experiment may justify a new measured attach-to-first-device-event
distribution. It must not be used to recreate the unsupported 273 ms value.

## B. SASS implementation gate

Keep SASS portability in discussion until all of the following exist:

1. a product-side backend, not code confined to a benchmark directory;
2. an explicit path from an eBPF input program to SASS instrumentation;
3. a reproducible build tied to a supported NVBit or equivalent release;
4. a positive test whose device event proves the eBPF program executed;
5. a negative control with no instrumentation and an application-output
   equivalence check;
6. at least one complete raw run on named GPU, driver, and CUDA versions.

The current NVBit example fails gates 1, 2, and 4 because it is separate from
the product path and its inserted device functions are active no-ops. Until the
gates pass, call NVBit a preliminary API experiment at most, not a working
SASS prototype.

## C. Cross-vendor portability gate

An AMD or Intel statement remains a design hypothesis until a named backend
builds and executes the same positive/negative device-hook test. Sharing an
abstract host interface or proposing SPIR-V is not by itself port evidence.

