# Queued per-launch callback-delivery repair

## Completed attempt: the placement change does not repair delivery

The queued build completes 07:27:25–07:27:26 PDT after LMCache releases both
leases. The one native_port run starts 07:28:44. Both LC workers exit zero;
all four BE workers exit 2 with missing/repeated output. run.log retains the
runner exception; cells retains each worker record and failure.json.
This failed Level-2 execution supplies no policy performance sample.

In BE1, retained XG5 entries still read slot 1 / kernel index 1 through
entry ordinal 200. After reactivation they retain entry type 2. The first
stream/command reports 680 completed blocks instead of 340, while later
commands report zero. Moving set_at_launch into the callback after insertion
therefore does not resolve the previously observed stale device arguments.
The device probe and its bounded, overwritable observations are unchanged;
this does not establish which argument-delivery or ABI layer is responsible.

The previous tool library is restored by the run.sh exit trap and compares
equal to the saved pre-run copy. After workers exit, GPU is idle (0%, 1 MiB,
P8); this attempt requires no persistence restart or module change. The local
GLM tool session receives these results to continue the concrete repair.
The historical queued-status text below is superseded by this section.

The short-context local Qwen candidate moves the saved per-launch value
into nvbit_set_at_launch inside the pre-launch callback, after instrumentation.
Zero/disarm uses the same order. The external prepare call, existing ABI,
target filtering, decision algorithm and XG5 device probe are unchanged.
This placement is a repair hypothesis, not proven by the API example alone.

The existing GLM session independently produced the same nonzero ordering
change in its original tool file; its zero/disarm call remains before
insertion. Both source versions are retained here. Only the Qwen candidate
is selected for one runtime attempt; equivalent hypotheses do not require
duplicate GPU runs. Neither candidate is a performance result yet.

build-tool.sh reuses the already built XG5 carrier from yiBj3W and the
installed HAL from jgLEyd; it compiles only the new host tool. The current
HAL implementation matches that successful build's saved source. No device
rebuild or BPF exporter rebuild is needed. Both shared leases serialize
the build after the ongoing pXYN4F LMCache campaign. Build and runtime
completion remain pending. No old data or manuscript is changed.
