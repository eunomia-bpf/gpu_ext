# OpenCode endpoint child-PATH fix review

Use `spark-gateway/qwen3.8-27b-nvfp4-200k` in deny-all mode. Review only the
attached lifecycle wrapper and CPU test. Attempt 06 passed endpoint-v1 200/200
and exact rollback but the fixed child failed before any workload because
`cuobjdump` was absent from its PATH.

Return `PASS` or `BLOCKER` plus at most three findings. Check that the repair:

1. replaces inherited PATH with one fixed allowlist containing CUDA 12.9;
2. resolves every required child command to its exact expected executable
   before any module mutation;
3. rejects missing, reordered, prefixed, or otherwise mutated PATH values;
4. preserves the fixed no-shell child, stderr evidence, child gate, and
   unconditional rollback semantics.

Do not execute tools, edit files, or treat pending GPU execution as a defect.
