# 预测集合保护修复前的 MoE 性能归因

这份记录只分析一次完整的 `paper-bpf` cell，不构成配对性能结论。
该 cell 在重启中断的 campaign 中已经完成：8 个各含 512 输入 token、64 输出
token 的请求，另有不计入测量窗口的固定 warmup。原始
[result.json](raw/paper-v3-575/timing-849ea75d-01/block-01-attempt-01/paper-bpf/result.json)
记录 `passed=true`、512 个验证输出 token、空的 cleanup errors 和服务器退出码 0。
同轮 `paper-native` 原始记录已损坏，不能仅凭其结果摘要中的 `passed` 或吞吐数字
恢复配对，也不能计算 BPF/native 加速比。

这次测量早于 prediction-set 保护与 epoch 修复。新版本已通过真实 GPU 正确性
canary，正式配对性能测试刚启动，不能将预期改善写成结果。下文源码行号用于定位当前工作树中保留的
执行和计数路径，不表示旧 cell 已经运行了新保护逻辑。

## 1.63 token/s 的旧比较为什么不能解释 BPF 开销

旧 generic 实验的
[页级 stride/LFU BPF 结果](raw/head-to-head-575-lossless/timing/attempt-01/gpubpf_host_stride_lfu/result.json)
是 1.628600 token/s，
[llama.cpp 原生 UVM 结果](raw/head-to-head-575-lossless/timing/attempt-01/llama_uvm/result.json)
是 5.939080 token/s，
[MoE-Infinity 结果](raw/head-to-head-575-lossless/timing/attempt-01/moe_infinity_075/result.json)
是 11.637299 token/s。前两者使用 llama.cpp 的页级内存管理路径，第三者使用
MoE-Infinity 的专家级执行引擎；generic BPF 并未复刻 MoE-Infinity 的专家预测算法。
因此三个数字混合了算法、管理粒度和执行引擎差异，不是相同策略在 BPF 与用户态
实现上的开销比较。

旧 generic BPF cell 记录了 123,026,112 次 `page_fault_calls`、32,716,544 次
`lfu_accesses` 和 127,797 次 sampled LFU reorder request。这也不是下面专家级
JIT 调用数量的同口径事件。新的专家算法单 cell 达到 11.197541 token/s，说明
1.63 不是 BPF 执行方式必然带来的上限；但不能把这两个不同系统的数字算成
一次有效的策略优化 speedup。

## 完整 cell 的精确观测

以下数值来自上述 `paper-bpf/result.json` 的 measured-window delta，排除了 warmup。
[launch.json](raw/paper-v3-575/timing-849ea75d-01/block-01-attempt-01/paper-bpf/launch.json)
确认执行域为 host uBPF JIT，`MOE_REVISION_VERIFY=0`，不是内核 UVM LFU 路径，
计时中也没有运行逐次 native shadow oracle。

| 项目 | 观测 |
| --- | --- |
| 主吞吐 | 512 / 45.724326613 秒 = 11.197540520 token/s |
| 仅请求耗时之和 | 45.500575268 秒；对应 11.252604983 token/s |
| 首次可见文本 TTFT，中位数 / 最大值 | 1.577196787 / 1.978716763 秒 |
| 请求端到端耗时，中位数 / 最大值 | 5.666903215 / 5.805915583 秒 |
| 最后一次显式 drain | 0.016410827 秒 |
| BPF 调用，match / rank / eviction | 18,432 / 18,432 / 39,991；共 76,855 次 |
| 排序选出的预取候选 | 9,510,469 个，约 515.976 个 / rank 调用 |
| 整个服务器进程树 CPU 时间增量 | 103.62 CPU 秒，约相当于 2.266 个核持续工作 |
| 专家缓存 access / hit / miss | 98,148 / 84,632 / 13,516；命中率 86.22896% |
| 完成的专家预取 | 26,475 次；350,700,134,400 字节，即 350.7001344 GB |
| 预取首次使用命中 | 9,593 次，占完成预取的 36.23418% |
| 首次使用前已被驱逐的预取 | 16,701 次；221,229,195,264 字节，即 221.229195264 GB |
| 末尾仍驻留但尚未使用的预取 | 181 次 |
| `exposed_fetch_seconds_total` 增量 | 0.106463467 秒，适用范围见下文 |
| 进程 `read_bytes` 增量 | 0；warmup 前的模型加载不在此增量中 |

预取计数满足 `26,475 = 9,593 + 16,701 + 181`。
首次使用前被驱逐的预取占完成预取计数字节的 **63.082153%**。
这里的字节是已完成专家 H2D 载荷的计数，不是 PCIe 总线分析仪测得的链路字节；
候选数量也不能当作实际 DMA 数量，队列替换、已驻留或忙碌节点都可能使候选不发生搬运。

对应路径在
[expert_dispatcher.cpp:626](deps/MoE-Infinity/core/parallel/expert_dispatcher.cpp#L626)
调用 `SetDevice(CUDA)`，等待 transfer event 后在第 640–641 行累计完成次数和字节；
[expert_dispatcher.cpp:567](deps/MoE-Infinity/core/parallel/expert_dispatcher.cpp#L567)
仅在驱逐仍属于 `unused_activation_prefetch_` 的节点时累计 wasted 次数和字节。
因此最直接的低效证据是大量“搬进来、首次使用前又被驱逐”的专家载荷。
它不能直接换算为相同比例的耗时损失：计算与搬运会重叠，实际关键路径尚未分离。

## 两个容易误读的时间指标

103.62 CPU 秒包含整个服务器的 Python、NumPy、PyTorch、调度、拷贝提交以及 JIT
工作，不是 BPF 专项时间。没有逐项 JIT 计时，不能从 76,855 次调用判断 BPF 是瓶颈。
平均约 2.266 个核也不能排除某一个串行线程处在关键路径上。

`exposed_fetch` 的 0.106 秒尤其不能解释为全部专家搬运只阻塞了 0.106 秒。
[model_offload.py:484](deps/MoE-Infinity/moe_infinity/runtime/model_offload.py#L484)
只计时 `archer_engine.fetch_tensors()`，由
[模块 pre-forward hook:1691](deps/MoE-Infinity/moe_infinity/runtime/model_offload.py#L1691)
调用。这里的专家执行另走
[enqueue_expert:245](deps/MoE-Infinity/moe_infinity/distributed/expert_executor.py#L245)、
[dispatcher SetDevice:884](deps/MoE-Infinity/core/parallel/expert_dispatcher.cpp#L884)
及 [wait_expert:287](deps/MoE-Infinity/moe_infinity/distributed/expert_executor.py#L287)。
该计时器没有包住这条完整路径，名称或注释不能代替真实测量覆盖范围。
`read_bytes=0` 支持这个稳态窗口没有新增块设备读取，但不表示没有 CPU→GPU 搬运。

八个请求的首次可见文本 TTFT 合计 13.174256252 秒，约占请求总耗时的 28.95%；
它包含 prefill、首段可见输出及调度等待，不是单独测得的 prefill kernel 时间。
请求耗时覆盖主窗口的约 99.51%，最后显式 drain 仅占约 0.036%。不过
[paper_server.py:74](paper_server.py#L74) 已在每个完成请求处 drain，相关等待包含在
请求端到端耗时中；很短的最后 drain 不能证明所有搬运开销都被隐藏。

## 能支持的假设与尚缺的证据

旧 executor 缺少上游的预测集合保护，结合 63.08% 的未使用预取字节，支持优先验证
“预取互相驱逐、污染专家缓存”的假设。当前数据不能区分其中有多少来自自我驱逐、
预测不准确、容量不足或过期预测，也没有证明哪个因素主导端到端耗时。

下一轮需要让修复后的 paper-native 和 paper-bpf 使用相同 executor、模型、请求、
warmup 和缓存预算，与 native-off baseline 做完整配对重复测量。除吞吐和 TTFT 外，
应核对测量窗口内保护、过期拒绝、完成复制、首次使用命中、未使用驱逐及缓存 miss
的增量；比较 BPF 本身成本则需要有效的同算法 native/BPF 配对，必要时再隔离控制面
计时。新保护是否减少浪费、是否改善吞吐、BPF 是否追平用户态，目前都仍待实测。
