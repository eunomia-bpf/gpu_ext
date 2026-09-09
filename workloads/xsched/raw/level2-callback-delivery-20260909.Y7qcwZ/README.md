# Queued per-launch callback-delivery repair

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
