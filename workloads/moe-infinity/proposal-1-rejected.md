# Closed MoE-Infinity proposal 1

The first proposal is permanently closed after the maximum three independent
review rounds. Its final plan SHA-256 was
`f9d4cd1f905671464a7721f7ed6b8bbf9b6db2b39018936fb432acf1cb5cd71c` and
its review record SHA-256 was
`0ef0e1cb1e13d778914f9aebcf2811ee937c812ed26e33b5240178f36a00bc6d`.

Rounds 1 and 2 required repairs to struct_ops composition and ownership,
stream/token evidence, commands and environment, engagement gates, prompt and
tokenizer freezing, stopping rules, canonical requests, golden outputs, and
observable timing semantics. Those repairs were incorporated.

Round 3 rejected the proposal for two remaining paper-facing defects:

1. it called per-request `64 / end-to-end latency` “completion goodput,” which
   conflicts with standard serving-benchmark terminology; and
2. it proposed reading Archer `get_hit_rate()`, whose
   `GetNodeVisitCounts()` implementation mutates `io_state`, then incorrectly
   applied monotonic-delta rules to gauges.

No code or GPU execution was authorized under proposal 1. Proposal 2 is a new
estimand and instrumentation design, not a fourth review of proposal 1.
