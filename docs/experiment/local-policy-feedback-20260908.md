# Local-model read-only policy feedback — 2026-09-08

Qwen 27B OpenCode session `ses_f7d059dcdffeTZqUX0xZIV5cYc` completed naturally
(`finish=stop`). This preserves feedback requested by the separate paper owner;
this experiment session made no manuscript edits. The text below is model
feedback, not independently established facts or approved replacement prose.

## Root corrections before use

- Reject the proposed description of LMCache as an existing upstream
  read-priority/write-deferral algorithm. A native implementation of the
  same experimental adapter does not establish upstream origin. The
  [write-budget report](../../workloads/lmcache-disk/results-575-gds-write-budget-20260907.md)
  identifies local Qwen development and a shared executor-budget change;
  it explicitly does not claim a new LMCache paper algorithm.
- Do not turn the median BPF/native p99 change of -4.096% into a consistent
  advantage: the paired range is -39.718% to +33.849%, with two adverse pairs.
  These storage-request results are not vLLM TTFT or proof of transparent,
  zero-application-change integration.
- The model's claims that a combination is absent from prior systems or is
  a new policy are not established by comparing two local draft versions.
  Preserve the distinction between a workload-specific combination, an
  empirical observation, and a literature-supported novelty claim.
- The model's old/new experiment comparison is a prompt to inspect workload
  and setup differences, not proof that an earlier conclusion was overturned.
  Its line numbers describe the snapshot it read; manuscript ownership stays
  with the other task.

## Original local-model response (unendorsed)



SoA work fully closed (source folder + runner + README all done). Below is the queued read-only paper review, returned for root to aggregate.

---

# eval.tex 修订版审查结果

**审查对象**：`docs/paper/tex-revision/eval.tex`（531 行）对照 `docs/paper/tex/eval.tex`（419 行）与 `asplos-27-rebuttal/revision-comments.md`（shepherd 要求，第 42 行："Call out any new policy enabled by/implemented in your mechanism, especially any interesting policy discovered by an agentic workflow"）。

## 总体结论

修订版相对旧版新增三类内容：(a) LMCache 本地磁盘策略段（tex-revision:166–171）；(b) "Policies from Prior Systems" 七策略移植对比小节（tex-revision:395–472，表 414–433 + 图 438–444 + 结果 455–472）；(c) RQ 重构（4 RQ→2 RQ）与机制代价新数据（验证 141/12 ms，481–484；RTX 5090 panel b，505–506；hook 4.0–16.6%，526）。**四个 agent 案例（专家卸载/vLLM/GNN/Faiss）与旧版逐字相同**。因此：修订版没有把任何"agent 发现的新策略"显式 call out；七策略与 LMCache 均为既有策略复现，按标准不算新策略。最接近"新策略"的是下面例 1 的组合规则。

## 五个最具体例子（行号均指 tex-revision/eval.tex）

### 例 1（唯一接近"新策略"者；旧版已有）：vLLM 分区域 × PCIe 门控自适应预取
- **行号**：147–154（抖动发现 149；策略 151–153；结果 142–145）。
- **策略做什么**：UVM 超订下 KV-cache（时间+每请求空间局部性）与权重（步长模式）两类数据竞争显存；按区域差异化预取（权重 stride、KV 顺序），用运行时 PCIe 利用率调节预取激进度，配 LFU 驱逐（304 LOC），共 ~680 LOC。效果：TTFT 均值/p99 提升 1.7–2×、解码 1.3×（vs vLLM CPU-offload），与 LMCache 持平，消除默认 LRU 抖动。
- **agent 发现了什么**：统一顺序预取最初导致 KV 与权重页相互抖动（149）；经设备端追踪探索后收敛到"区域类型 × PCIe 利用率"双因子门控（151）。
- **gpubpf 怎样支持**：设备端 per-warp 访问追踪（SIMT 统一执行、低开销）提供局部性/区域信号；主机侧 UVM 缺页/预取回调与可插拔 LFU 程序执行决策；应用零修改、可即时分离。
- **可推广启示**：超订 LLM 服务中内存管理决策单位应是"数据类"而非"页"；PCIe 带宽利用率是廉价的一级运行时信号，可作为预取节流阀。
- **性质**：**新的策略结构（agent 发现的组合规则）**，构件本身（seq/stride 预取、LFU）成熟；"区域×信号"规则在文中对比的已发表系统（UVM/vLLM/LMCache）中不存在。
- **状态**：旧版已有（旧 eval.tex:147–154，逐字一致）。

### 例 2（旧版已有）：GNN 单块前瞻（K=1）与 PCIe 饱和诊断
- **行号**：192–197（K 对比 194–195；瓶颈诊断 196；uprobe+历史 192）。
- **策略做什么**：顺序预取（375 LOC）+ uprobe 追踪 PyTorch 分配 + 设备端追踪历史决定预取范围，恰好 K=1 块前瞻；透明取得 2.65×（183–184）。
- **agent 发现了什么**：K=1 改善 epoch 21%，K=2/K=3 降 16%，自适应 K（≤6）降 46%——**预取深度收益非单调**；最初怀疑锁竞争，后确认真正瓶颈是 PCIe 带宽饱和（196）。
- **gpubpf 怎样支持**：uprobe 钩子（分配事件）+ 设备端访问历史 + UVM 预取接口，均为可分离 eBPF。
- **可推广启示**：UVM 下"更深前瞻=更好"不成立，PCIe 饱和是新的性能 regime；未来系统设计应把 PCIe 遥测作为预取决策的一级输入。同时示范 agent"错误假设→观测纠偏"循环的价值。
- **性质**：成熟算法（顺序预取）+ agent 调参；非单调性与瓶颈诊断是**新经验知识**。
- **状态**：旧版已有（旧 185–190）。

### 例 3（旧版已有）：Faiss 相位感知预取，"驱逐主导相位检测"
- **行号**：221–232（检测器 223–224；驱逐主导 225；状态机缺陷 229；2/8 命中 231）；收敛策略 214–219。
- **策略做什么**：~890 LOC = 顺序预取 375 + K-means 迭代 stride 预测 472 + 设备端 L2 预取 45，**配默认驱逐**。
- **agent 发现了什么**：基于动量的相位检测器+周期驱逐改善 BUILD 但恶化 SEARCH（查询前向漂移迷惑检测器）；修正分类器后发现**决定性能的是驱逐策略而非相位检测**（225）；快速路径优化引入"卡死在 SEARCH 模式"的状态机缺陷，一轮内检出修复；8 个策略仅 2 个为正。
- **gpubpf 怎样支持**：设备端追踪驱动相位检测；主机侧预取/驱逐程序可替换；失败数秒内恢复、应用不重启（支撑数据 87–89）。
- **可推广启示**：混合相位负载中驱逐选择常比相位检测更重要；ANN 相位检测器必须对查询前向漂移鲁棒；2/8 命中率量化了"安全+可观测"迭代循环是 agent 策略搜索的前提。
- **性质**：成熟算法组合；驱逐主导结论是**新诊断洞见**。
- **状态**：旧版已有（旧 214–225）。

### 例 4（旧版已有）：MoE 专家步长对齐预取 + 共享层 LFU 钉住
- **行号**：109–116（步长发现 109；per-warp 追踪 111；FIFO→LFU 113；~820 LOC 115）。
- **策略做什么**：设备端访问观测（45 LOC）+ 主机侧 stride 预取（472）+ LFU 驱逐（304）；解码较 cudaMemAdvise 调优 UVM 1.76×、较 ncmoe=64 4.8×，prefill 保持在默认 UVM 4% 内。
- **agent 发现了什么**：顺序预取与 MoE 步长不匹配而劣化 decode；per-warp 追踪表明专家激活遵循**可预测步长序列**，并识别出小集合高频访问的**共享层**；先试 FIFO，收敛到 LFU（钉住共享层、驱逐冷专家页）。
- **gpubpf 怎样支持**："设备观测、主机决策"分工——设备端 per-warp 内存访问追踪，主机侧缺页/预取/驱逐回调。
- **可推广启示**：MoE 专家权重存在可利用的步长结构与热共享层核心；"设备观测/主机决策"是可复用的 MoE 内存管理架构模式。
- **性质**：成熟算法组合（stride 预取+LFU）；工作负载结构洞见是新的。
- **状态**：旧版已有（旧 108–115）。

### 例 5（本次修订新增，但**不是新策略**）：LMCache 本地磁盘读优先/写延迟受限
- **行号**：166–171。
- **策略做什么**：读优先于后台写，每个写请求累计推迟 ≤200 ms。读 p99 中位：FIFO 323.707 ms → 原生 123.141 ms → gpubpf 118.097 ms；相对 FIFO −61.9%。
- **agent 发现了什么**：正文称"agent-developed policy"，但同段即说"the same policy implemented through gpubpf"——即 **LMCache 既有原生策略的移植**，agent 贡献是实现而非发明。
- **gpubpf 怎样支持**：磁盘后端队列上的主机侧策略程序，对应用透明；gpubpf 比原生还快 4.1%。
- **可推广启示**：证明机制透明性（零应用修改即可追平/略胜原生），用于回答"实现既有策略时机制有无性能代价"。
- **性质**：**既有策略复现，按标准不算新策略**；与 395–472 的七个 SOTA 移植（MoE-Infinity/Expert Buffering/FineMoE/Hummingbird/POD-Attention/XSched/GPREEMPT，gpubpf 相对原生偏差 0.21%–1.18%，其中 MoE-Infinity 略优）同属"机制代价"证据。
- **状态**：**修订版新增**（旧版无）。

## 最小欠缺（修订版仍需补的，按严重度）

1. **shepherd 的 "call out new policy" 请求仍未被直接回答**（revision-comments.md:42）：eval.tex 没有一处文字区分 59 个探索策略中哪些是移植、哪些是新组合规则。建议在 "Policies from Prior Systems"（395–472）或 discussion 加 2–3 句：移植 = 七 SOTA + LMCache；agent 发现的新策略结构 = 例 1 的"区域×PCIe"门控（唯一接近新政策者），例 2–4 为成熟组合上的新经验发现。
2. **166 行措辞有误导风险**："an agent-developed policy" 应改为 "an agent-implemented port of LMCache's native read-priority/write-deferral policy"——否则正中 shepherd 警告的 "perceived misleading statement"（revision-comments.md:40）。
3. **旧版洞见被删除且结论反转无解释**：旧版"memory-bound 下时间片策略无效（<1%）、需按负载选对策略层"（旧 eval.tex:282–285）是最直接的"未来系统设计启示"之一；修订版仅存注释残片（tex-revision:293），而新的 memory-bound 段（312–328）改为"scheduling 收益更大、内存优先级再加 9.5–12.0%"。两处结论相反，正文无一句说明实验差异或为何改判——若新测量确已推翻旧结论，需要一句过渡解释，否则是对 shepherd "insights" 请求的实质倒退。
4. 次要：106 行与 530 行仍保留内联 `\arq{...}` 审稿人评论（两处在旧版即存在，定稿前应清理）。

