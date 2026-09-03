You are an independent read-only systems experiment reviewer. Do not edit
files, run commands, launch GPU work, or use the network.

Review the RTX 5090 Table 1 work under
workloads/llama.cpp/observability_overhead/revision-rq4 and the status summaries
that directory links. Do not propose rerunning already completed formal cells.

The revision promise is a fair RTX 5090 comparison of baseline, gpubpf and
existing device-observability tools. Current evidence reportedly has an exact
three-arm histogram correctness closure, but no admitted seven-arm timing run.
Two blockers remain: lossless BPF exit-event collection for kernel return
observation, and an unfair/incorrect host-time versus `%globaltimer` boundary
for launch latency.

Return a concise, executable design for two distinct correctness experiments:

1. An exit-observation experiment that reconciles per-launch and total events
   between baseline semantics, NVBit and BPF without silently dropping records.
2. A launch-latency experiment that proves the host launch target is the real
   CUDA launch function and compares measurements in a disclosed common clock
   domain or defensible common interval.

For each, state the hypothesis, exact controls, engagement/correctness gates,
failure outcomes, and the minimum condition before performance timing is
allowed. Rank which repair should be implemented first. Flag any claim that
cannot be supported by existing evidence. Cite repository paths where useful;
do not invent results.
