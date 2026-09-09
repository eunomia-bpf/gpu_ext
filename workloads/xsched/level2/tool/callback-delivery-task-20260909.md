# Local-model callback-delivery task

2026-09-09 PDT. The preceding Qwen driver task completed normally; the new
short-context session is `ses_f7986d4f8ffeSm3OtJEmhMdi77`, using
`spark-direct-qwen27/qwen3.8-27b-nvfp4` on the existing local OpenCode server.
The two GLM sessions remain live; total active sessions is three. No local
model was stopped or restarted for silence. The candidate has a separate
filename to avoid overwriting GLM's tool source. This is a new recorded task
prompt, not a recovered original-paper agent transcript.

## Exact prompt

Implement one bounded XSched NVBit per-launch parameter-delivery repair, SOURCE ONLY. Workspace /home/yunwei37/workspace/gpu/gpu_ext.
You are a LOCAL Qwen27 OpenCode subagent; root coordinates builds/GPU/commits. No GPU runs, builds, installs, driver reloads, new sessions, tests, paper edits, or hashes/checksums. Use apply_patch for edits. Do not impose short timeouts or stop other sessions.
OWN ONLY these new files:
workloads/xsched/level2/tool/xsched_guard_tool_callback.cu
workloads/xsched/level2/tool/callback-delivery-handoff-20260909.md
Another GLM session owns xsched_guard_tool.cu and isolated HAL source; DO NOT edit those. Copy the current tool source into your candidate with a minimal code delta. No interface redesign or additional diagnostics.

Task: current host sets nvbit_set_at_launch outside the actual NVBit pre-launch callback. XG5 actual runtime shows later launches keep consuming one early slot although host publishes distinct slots. The official NVBit example sets the value INSIDE the callback AFTER instrumentation. Implement that minimal ordering in your candidate using the existing TLS armed_val. Preserve target filtering, counters, disarming for unmanaged launches, actual context/function association, shared host ABI xg_host_prepare, and native/BPF decision_mode. Keep outer prepare behavior if needed as fallback; ensure callback sets the saved nonzero value after instrumenting, then consumes/clears the one-shot TLS marker. No new arbitrary state protocol.
Read:
1. workloads/xsched/level2/tool/xsched_guard_tool.cu (about 220 lines): insert_entry_call calls nvbit_add_call_arg_launch_val64(entry,0); xg_host_prepare stores TLS armed_val then nvbit_set_at_launch(ctx,f,ctx_dev); current callback if armed_val!=0 only clears it, while instrumentation happens later.
2. workloads/llama.cpp/observability_overhead/revision-rq4/deps/nvbit_release_x86_64/tools/mem_trace/mem_trace.cu lines250-280: official enter_kernel_launch instruments then nvbit_set_at_launch inside callback.
3. same NVBit core/nvbit.h lines589-599 and646-657: actual API definitions.
4. workloads/xsched/raw/level2-xg5-runtime-20260909.jgLEyd/README.md (runtime findings, no need reread huge logs unless necessary).
Root already restored the missing InstrumentManager::Xg5Drain via GLM; no HAL edit needed.
Actual evidence: 201 host preparations/callbacks, zero disarms; host slots differ but BE1's device mark remains slot1/kidx1 through ordinal200 and switches permanently to resume type2; BE4 similarlyslot3. Three BE workers missing/repeatedwork; LC finishes. This is a failure repair, NOT performance results. No repeat of completed baselines. Probe itself remains unchanged for a single post-repair diagnostic root will run.

DELIVER the candidate .cu now with the small placement change, then a short handoff with exact differences/remaining concern. Do not spend time reconstructing unrelated native SASS metadata or changing policy logic. Root is running a 15-cell LMCache campaign under both shared locks, so no build or GPU action until root requests it. No git commit/push by you; root does it.
