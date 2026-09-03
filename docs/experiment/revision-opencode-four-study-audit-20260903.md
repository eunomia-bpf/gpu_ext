# OpenCode four-study fidelity follow-up — 2026-09-03

OpenCode session `ses_f98b20c62ffebp6bYJGSwHi35D` returned the complete report
below and exited 0. It used the configured default model with `snapshot=false`
and only read/glob/grep/list permissions. This is an advisory source/raw-record
audit, not an experiment or a substitute for root review. The local full event
stream is `/tmp/gpubpf-opencode-goal-audit-WKbabf/events.jsonl`.

## Root reconciliation and next experiment

- GPreempt and POD already satisfy their scoped measurement objectives. A small
  estimated BPF/C difference does not establish statistical equivalence.
- FineMoE's missing plot was subsequently completed and pushed in `115414f`:
  all five points per arm and completed-payload partitions, not a new run.
- Hummingbird's original 50-cell study remains a complete, valid **negative
  study of the documented conservative port**. It does not causally identify
  the source of its roughly 19–20% background goodput loss.
- The audit's asserted `~1.3 microseconds` launch gap is not supported by the
  cited paper paragraph. The [v2 PDF](../reference/2026-hu-hummingbird-v2.pdf),
  p. 8, says approximately 5 microseconds for synchronization and 6–7 for launch,
  and separately reports **1.3% slowdown** in its own evaluation. None of these
  is a measured timing for this RTX 5090 port.
- Source inspection confirms the difference: `idle_policy.c` waits for actual
  LP completion, then issues another kernel at its tick. The paper describes
  launching as the preceding kernel approaches completion. A **fixed-bound
  completion-fence ablation**, not a favorable-result retry, can test whether
  this stronger fence contributes to the observed loss.
- Prepare an independent depth-1/depth-2 build and fresh outputs; use the same
  native/JIT decision inputs, profile, executor, workloads and full numerical
  oracles. A two-event host outstanding bound is **not** evidence of GPU queue
  depth, simultaneous execution, or the paper's preemption bound. Preserve
  actual issuance/retirement, exactly-once CTA execution, HP admission locking,
  bounded final drain and all failed/negative outcomes. No new GPU cell has run.

This ablation, the Expert Buffering live adapter, and Table 1 diagnostics are
being prepared in separate OpenCode-assisted workstreams. Root alone runs GPU
experiments and publishes reviewed steps. The user has requested commit/push
of each completed step; LMCache remains paused.

## Complete OpenCode report, retained as returned



审计结论（只读；四项原始目标逐项判定）

**1. GPreempt — 已完成。** 两种实现对 native 基线均显著差异：LC p99 C/native −15.31/−11.68/−9.61%，BPF/native −14.82/−11.61/−10.81%；BE 200/连续 C −9.31/−8.95%、BPF −9.47/−8.88%，95%CI 全不含 0。45/45 审计通过、图件齐备，满足“eBPF 与非 eBPF 均显著区别于基线”。BPF≈C 为另一问题（未被要求），无缺口。

**2. FineMoE — 研究问题已完成（混合）。** 动态集较 all-positive 降 evicted-unused 完成负载 60.41%[60.17,60.66]、吞吐 +24.64%（C/BPF 同效），有真实 copy 生命周期佐证；但对 demand-only 净吞吐 −12.62% 为有效负向（已披露）。缺失交付：plan.md:345 规划的“吞吐+三分类字节、含五点散点”图未产出（仅表格）——按要求归 root，仅标注不改。

**3. Hummingbird — 保守端口协议已完成且为有效负向，但“改善丢失后台吞吐”的忠实目标仍未答。** C/BPF 对 fixed 均损失约 19–20%。根因是单一在途完成栅栏：idle_executor.cpp:209,261,303（max_lp_inflight=1、completion_fence="event-query-before-next-launch"）+ idle_policy.c:51-56（下一片须待 lp_gpu_done 才发）。预测 tick（idle_policy.c:59-66）虽算出，但深度=1 使其无法跨 ~1.3µs 发射间隙重叠预测片，退化为串行；raw 显示 lp_event_waits 25.7M、yield 3.16s、tick_waits 21.5M。报告自认“不能证明原系统损失 20%”“未识别何偏差致损，需单独消融”。栅栏系评审预采纳并要求披露开销（plan-review:29-33），非隐瞒替代，但它正是论文§4.3 填气泡的吞吐机制，故负向被栅栏混淆。**最小必要后续（非求好结果重跑）**：仅给 idle 执行器加“有界在途深度 K>1、按 profiled tick 发射、保留有界 backlog 守卫”的预测流水线，重跑 idle vs fixed 与等时片对照即可；不涉及 N-BEATS、device-BPF 或全系统。

**4. POD-Attention — 已完成。** 设备 BPF（engine 2）在真实注意力核内选任务，非 host-JIT/回退/C 回退；125,280 CTA 上下文+5,550 桥接发射经独立审计；FP16 全输出对 FA 通过，开销 +0.51–1.18%。严格 verifier OFF 不属目标措辞范围，无缺口。

证据清单
- gpreempt: results-load-study-575-20260903.md:3,42-50,63-65,90；figures/load-study-575-20260903.{png,pdf}, scheduling-comparison-2x2.{png,pdf}
- finemoe: results-performance.md:66-85,99-105；plan.md:345-346（图交付规划）；raw/full-v1/block-01/finemoe-bpf/worker-result.json（copy 记录）；glob 无 *.png/pdf/svg
- hummingbird: idle_executor.cpp:209,261,303；idle_policy.c:51-56,59-66；raw/preflight-575-01/block-00/periodic/idle_bpf/result.json:65-78；results-575-20260903.md:45-46,76-77；raw-audit.md:55,89-91；plan-review.md:29-33
- pod-attention: raw-audit.md:86-97；plan.md:164-166；results-575-20260903.md:3,18-19；raw/full-575-01/block-01-pod_bpf/…/diagnostic.json

