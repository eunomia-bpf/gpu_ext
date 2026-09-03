# OpenCode subagent consultation — 2026-09-03

Two parallel, read-only OpenCode 1.18.27 sessions were requested by the user.
The first audited comparison fidelity and review coverage; the second ranked
supplementary experiments against the unfinished revision commitments.
Both returned complete final reports and terminal exit 0. They used the
configured default model, `spark-gateway/qwen3.8-flash-next-nvfp4-220k`.
`snapshot=false`; only read/glob/grep/list permissions were allowed. They
did not edit source, run a shell, launch a GPU job or publish.

- Coverage session: `ses_f98c59ce1ffe8c5Vbr4YuBbboy`.
- Experiment session: `ses_f98c59cd4ffeIhmO4VVyNl6jL4`.
- Full local event streams: `/tmp/gpubpf-opencode-review-coverage-Vdjkbv/events.jsonl`
  and `/tmp/gpubpf-opencode-supplemental-experiments-S068Uc/events.jsonl`.
  These temporary streams are not claimed as published artifacts. The complete
  final reports are preserved below; they are advisory, not experimental data.

## Main-thread reconciliation and decisions

These reports read a moving workspace and contain errors; do not copy their
claims directly into the paper.

1. UVM has **two arms**, not three. It is a no-prefetch mechanism-cost control,
   not another full research-system reproduction. The first report's opening
   seven-three-arm generalization contradicts its own UVM paragraph.
2. MoE-Infinity, FineMoE and Hummingbird include explicit component reimplementations;
   framework reuse does not make them unmodified original-system baselines.
   XSched is the original **Level-1** executor plus matched HPF decisions.
   GPreempt is a host-mapped compatibility port. POD executes actual device-BPF
   selection, but its performance runtime has verification disabled.
3. All submitted revision commitments bind the response, not only Table 1 and
   transcripts. LMCache is paused **by user direction** after all eight warm
   CPU/disk outputs disagreed with recompute. Disk I/O engagement passed, but
   there is no valid formal performance result. The old running/startup-only
   wording in the reports is superseded.
4. Review archive commit `46d2f70` now includes A/D/E follow-ups, E's updated
   score, G, and local copies of both revision comments. Paper source
   `ee1623e` was already built and visually checked; it was not merely
   waiting for a build. Page budget and remaining experiments are still open.
5. Expert Buffering's Section VI selector **is implemented at the CPU boundary**:
   nine tests, 2,131 native and 2,131 real JIT decisions. The live whole-expert
   residency adapter and GPU comparison remain missing. Do not call it absent
   altogether, or call these CPU tests reproduction.
6. Table 1 is not known to need only execution: fresh preflights can expose
   compatibility problems. Its first 575 attempt failed before GPU use because
   sudo's PATH lacked nvcc; the second built all tools but rejected empty native
   output. Removing llama-cli's output-suppressing `--log-disable` flag preserves
   the exact-output oracle and all generation settings. A regression test passes.
7. Prioritize the already agreed **Table 1, whole-expert Section VI comparison,
   and live invalid-transition/fallback tests**. No new user-approval gate is
   imposed by a subagent suggestion. Driver maintenance must still respect
   actual device ownership and the existing GPU leases.
8. Best genuinely new candidate: **separate POD process setup/attachment/first
   launch from steady-state cost**, first checking whether existing logs suffice.
   Do not combine this with an unrelated hook-count sweep, demand that new
   results reproduce a desired interval, or treat 295.63 s as isolated startup.
   Use matched CUDA-context and device-BPF selector paths, plus the existing
   inline control; FA variants are not needed to isolate mechanism setup cost.
9. Stale-state sensitivity is a secondary candidate, not an admitted run:
   native and BPF must receive the **same delayed observations**, include actual
   fault/migration engagement, and use a real existing workload. Native-fresh
   versus BPF-stale alone confounds implementation with information quality.
   Neither speculative new proposal below authorizes arbitrary mode switches,
   an unreviewed protocol, a smaller/toy substitute, or a new experiment framework.

The experiment-design skill favored completing consequential missing evidence
over adding more names. The document workflow preserved existing reviews and
published only scoped changes. No new scientific result is claimed by this
consultation.

## OpenCode final report 1 — preserved as returned

# 最终报告（只读核查，gpu_ext 仓库）

## 0. 前提确认
- 当前 review.txt 仅含 A–F 初始评审（review.txt:7–330），无 A 支持确认、D 疑虑解决、E 改 Weak accept 3 与存储讨论、G meta-review——逐字承诺以 revision-shepherd-comment.md:11–19（作者响应）与 :25–45（shepherd 追加）为准。
- 论文最新为 ee1623e 构建 16 页、结论在 p14（revision-completion-checklist.md:17–19；revision-build-review.md:9–13 记录的是前一检查点 b5e2f25），仍超"原文+2页"目标（build-review.md:29, 76–80）。

## 1. 三臂完整性（baseline / native 同策略 / 实际 BPF）
全部七项研究结构上均含三臂以上并有实际 BPF 决策执行与独立审计，但"实际 BPF"位置与原实现保真度不同：
- **GPreempt**（45/45 cell，results-load-study-575-20260903.md:3–4, 15–22）：native-CUDA / 原始 C 执行器 / 实际 BPF（host ubpf JIT + kernel BPF timeslice 回调，:63–66）。满足三臂且含**原始 C 单块实现**对比，BPF/native LC p99 −10.81%、BE 代价 −8.88%（:49），BPF/C 区间含零（:50, 53–54）。
- **XSched**（46 cell 审计，performance-full-575-20260903.md:4–6, 48–53）：native / 原始 XSched HPF+Level-1 执行器 / 同策略 BPF（ubpf JIT，:22）。含原始单块实现臂；BPF HPF 与 XSched 无显著差异（:61, 66–69）。
- **FineMoE**（20 cell，results-performance.md:3, 31–36）：demand-only / all-positive / 官方 FineMoE 路径 native C / 实际 host uBPF JIT 同选择器（:14）。BPF/C +0.21% [−0.17,+0.52] 未决（:84–85）。
- **Hummingbird**（50 cell，results-575-20260903.md:3–6, 20–31）：五臂含 fixed GPreempt，BPF 为 host uBPF JIT 1.91×10⁹ 次（:65）。**负结果**：C 与 BPF 均较 fixed 损失约 19–20%（:8–11, 45–47）。
- **MoE-Infinity**（15 cell，results-paper-v3-protected-575.md:3–6, 104–120）：baseline / paper-native / paper-BPF 齐备，BPF/native 0.9965 [0.989,1.006]（:10），但两策略臂吞吐均低于 baseline。
- **POD-Attention**（250 cell，results-575-20260903.md:3, 78–89）：**唯一实际 device-BPF 选择决策**（device engine 2，:41, 127–128；无 host 回调/C 兜底 :46）。
- **UVM 机制成本**：非 SOTA 基线，是同一 no-prefetch 决策的 native vs kernel gpubpf 比较，+3.219% [2.247, 4.202]（uvm-policy-mechanism/results/analysis.md:19–21），仅 scoped 到 CPU-resident 非首触 UVM 路径（:39–53）。

## 2. 部分端口、不能称"完整复现"
以下报告自身已明确排除完整复现，均不得包装为原始系统复现：
- MoE-Infinity：论文 v3 算法自实现+同前端部署，非原作者硬件/模型；且原始 artifact 三次 preflight 耗尽、无可运行原实现臂（activation-aware-port.md:25–27；results-paper-v3-protected-575.md:6–7；revision-experiment-status.md:238, 255）。
- XSched：仅 Level-1 抢占，非论文 Level-3（performance-full-575-20260903.md:14–15）。
- GPreempt：host-mapped flag 兼容变体，非 GDRCopy/硬件复现；基线是同定制驱动上的 native（results-load-study-575-20260903.md:68, 71–72）。
- Hummingbird：自实现 port（"not an author-released scheduler"），单在途保守执行器，仅约束该 port（results-575-20260903.md:70–77）。
- FineMoE：dynamic-set 组件端口，非 EuroSys 完整复现，MT-Bench 替换、离线 history 等限制（results-performance.md:115–120）。
- POD：可表达性+实测成本，非完整 POD serving 系统（results-575-20260903.md:12–16）。
- Expert Buffering：历史四臂是"mechanism/workload boundary…not reproduction of the original expert-atomic algorithm"（expert-buffering-policy/raw-audit.md:22–27）；**承诺的 current-batch activity、expert-atomic、inactive-first/LIFO 策略仍未实现**（revision-completion-checklist.md:33）。

## 3. 剩余显式承诺（按 meta-review 绑定强度）
硬承诺（作者响应原文，不可降级）：
1. **R6 RTX 5090 Table 1/NVBit**：仅 harness 就绪，readback 截断缺陷修于 ea9907d，受控运行时未执行（revision-completion-checklist.md:37；revision-experiment-status.md:262, 269–270）。
2. **R7 原始 agent prompts/日志**：公共 harness 与新写模板已推（1e4564c），**原始 transcripts 未恢复**（revision-completion-checklist.md:38）。
承诺项：
3. **LMCache 本地磁盘后端**（E 存储追问）：575 全负载启动失败、bundled Triton 13.1 失败/12.9 通过，preflight 02 在跑，无任何 575 存储性能结果可声明（revision-completion-checklist.md:32）。
4. **Q2 安全**：native scheduler-init 与 invalid-prefetch 转换 oracle 未完成（revision-completion-checklist.md:34；bpftime-device-smoke/results-strict-575-20260903.md:63）；性能运行时 verification-disabled，strict verifier 未建模 POD pointer/ticket ABI（revision-experiment-status.md:128–131）。
5. **Expert Buffering 策略实现**（作者响应 Q1 点名"will do for Expert Buffering's hot-expert residency"，revision-shepherd-comment.md:13）。
6. **Fig. 13 扩展**：新 engaged、独立计时的 memory/scheduling 矩阵未测；旧观测不得当调度因果证据（revision-completion-checklist.md:36）。
7. **排版修复需新构建验证**、页预算达标、最终构建/发表（revision-completion-checklist.md:17–22, 40）。

## 4. 最高影响缺口（排序）
1. R6 Table 1 与 R7 原始 transcripts——两条绑定性硬承诺均为零完成证据。
2. Q1 侧 Expert Buffering 策略实现缺失 + LMCache 磁盘测量缺失（E 的存储问题目前仍只有讨论）。
3. Q2 侧：驱动原生安全转换测试 + POD ABI 验证缺口，使"验证器保证"论述与实际 device runtime 状态不一致（verification-disabled）。
4. policy/mechanism：新 engaged Fig. 13 矩阵缺失 + 16 页超限（结论 p14）尚未收敛。
5. MoE/Hummingbird 无原始系统臂，shepherd 要求的"与原始 ad-hoc/unsafe 实现比较"目前仅 GPreempt、XSched、FineMoE（POD 有 inline selector）满足。

## OpenCode final report 2 — preserved as returned

# gpu_ext ASPLOS 修订：bpftime 状态 + 至多三项额外实验建议（只读核查，无执行）

## 一、bpftime / Table 1 现状（已核实文件与用户简报一致）
- **Table1 专用构建** `bpftime-table1-575`：108/108 完成；map 全宽修复对应 `test_offline.py:393`（threadhist 全宽含零尾回读），该文件恰有 **23 个 CPU 测试**（`def test_` 计数=23），含驱动精确 575.57.08 准入、双租约、CPU 8–15 放置+CPU16 遥测、私有段自有清理、时钟误差显式记录。**尚无 GPU preflight**——`plan.md` 明确 CPU 修复不构成本次实验结果。
- **RQ4 剩余序列（agreed，R6 硬性承诺）**：7 路径确定性精确输出门（seed 1797）→ 每路径 1 个 pp=32 preflight → pp=512×10 交错块×7 配置；`run_revision_rq4.py --phase preflight|full`，证据 `raw/preflight-575-01`、`raw/full-575-01`。NVBit v1.8 官方适配为 "matched custom adapters"。
- **验证边界**：性能运行时 `build-cuda-pr503` 验证器 OFF（CMakeCache:587）；strict 证据来自**独立** R5 构建 `ea9907d`（277 步，2 对真实 positive/reject 计数器对：32,768 回调/对、负例零计数）。二者不可互证。
- **EB Section VI（agreed 承诺）**：9 CPU 测试、2,131+2,131 native/JIT 决策、744 字节 BPF；**无 GPU 适配器、无真实驱逐**。剩余：4 处 C++ 卸载器改动+构建 → 3 正确性单元 → 5 块×3 臂（同 K FIFO 基线/Section VI native/同策略 host-BPF）=15 单元。
- **LMCache：按用户指示停止**，不新增臂、不发明 `--config bpf`。

## 二、优先级总排序
**P0（先于一切额外项，均为 agreed）**：① Table1 GPU preflight→full（仅差执行）；② EB 适配器→正确性→15 单元（先写 C++ 再计时）。以下为额外项，均排在两者之后。

## 三、三项额外实验（≤3，非冗余，含新旧区分）

### E1（额外第1位）实时非法转换回退/提交观察 —— **agreed 未完成**（checklist"transition-validation"行、safety-design §7.3–7.4、R5 边界）
- **不确定性**：非法 UVM prefetch 选择是否真的走 native 回退、调度器 init 提交路径是否真实执行（现仅 610 PMM ioctl + CPU/load fixture，未执行构造函数/setter）。
- **公平臂**：baseline=无 BPF 原生行为记录；native 臂=合法 typed 请求经生产验证器提交（正对照先行）；BPF 臂=越界动作/端点请求 → 必须回退 native prefetch（`uvm_perf_prefetch.c:117/149`）或 D575 时片路径 `NV_ERR_INVALID_ARGUMENT`（`kernel_channel_group_api.c:1482`）。
- **真实负载**：既有 40GB/1.25× 向量加 UVM 负载 + `extension/revision_sched_verifier.c`/`revision_pmm_fixture.bpf.h` 夹具（849ea75d 模块、无计时）。
- **正确性/参与**：全输出匹配；dmesg/计数器记录回退 vs 提交、冲突闩锁；钩子命中计数。
- **主指标**：回退/拒绝二值 oracle + 零输出失配；次要：延迟差无显著回退。
- **最小重复**：3 对 fresh-process 正/负（对齐 strict 的 2 对标准）。
- **负结果含义**：§verification 的回退叙述必须收窄或撤回，属实质安全缺口。
- **前提/证据**：GPU 无计时租约；**调度器 init 提交臂需显示维护窗口授权**（GDM 占用核心模块）；证据 `docs/experiment/revision-safety/prefetch-fallback-575-01/`。

### E2（额外第2位）Device-BPF 启动 vs 稳态与钩子数缩放分解 —— **genuinely new**
- **不确定性**：POD 保留的 295.63s vs 12–14s 进程壁钟差究竟由一次性初始化/attach 还是每核钩子稳态成本主导；每钩子边际成本是否随钩子数/网格近常数（支撑 discussion.tex:42–50 "Trampoline scaling" 的 TODO 与 Table 1 披露）。
- **公平臂**：baseline=FA serial/streams（无钩子）；native 臂=POD inline CUDA 选择器；BPF 臂=pod_bpf 加阶段计数（进程初始化/attach/首个被插桩 launch/稳态）×钩子数阶梯，同二进制同种子。
- **真实负载**：冻结 POD 负载（Llama-3-8B/Yi-6B 十形状），**新目录 `raw/startup-scaling-575-01/`，不动 full-575-01**。
- **正确性/参与**：沿用 FP16-vs-FA 全输出门 + CTA exactly-once 账本（子集）；每核记录数证明参与。
- **主指标**：阶段分解时间、每钩子边际斜率；稳态比值须复现既有 0.51–1.18%。
- **最小重复**：阶梯 {1,2,5,10} 钩子 × 3 成对重复（无需 5 块）。
- **负结果含义**：超线性缩放→讨论段"warp 聚合"表述必须加边界；一次性主导→部署成本须显式入表而非隐藏。
- **前提/证据**：仅需遥测插桩（无策略改动）；排在 Table1/EB 之后的 GPU 租约；反馈 `discussion.tex:49` TODO。

### E3（额外第3位）陈旧状态鲁棒性/同步间隔敏感性 —— **genuinely new**（discussion.tex:4–15 TODO，当前零证据）
- **不确定性**：延迟发布摘要是否实测导致 thrash/留冷页，过期/滞回缓解是否有效。
- **公平臂**：baseline=驱动默认 UVM FIFO/LRU；native 臂=同一实时统计策略；BPF 臂=同 BPF 策略注入固定陈旧间隔（fresh/100ms/1s）及过期变体。
- **真实负载**：40GB 向量加中途访问模式切换（首选，便宜）或 GPT-OSS-120B 解码 offload 复现。
- **正确性/参与**：输出不变；决策计数对陈旧窗口、缺页/迁移计数即参与证据。
- **主指标**：缺页/迁移率与吞吐随陈旧度的退化（thrash=缺页升+吞吐降）。
- **最小重复**：3 成对块 × 3 陈旧档 + 1 过期档。
- **负结果含义**：缓解无效→"策略应过期陈旧观测"降级为未测设想。
- **前提/证据**：需给 BPF 策略加间隔开关（少量新码）；GPU 租约排最后；证据新建 `workloads/stale-state-575/` 并回链讨论段。

## 四、结论
先完成 Table1（只差 preflight/full 执行）与 EB（先 C++ 适配器），LMCache 冻结；额外项按 **E1（agreed 安全缺口）> E2（新、复用现成负载、解释保留异常）> E3（新、需新旋钮、最推测）** 排队。所有结果（含负向）须以真实正确性/参与记录入库，不用摘要/构建替代，不承诺完成时限。
