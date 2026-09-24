# gpubpf 磁盘方向外部评审建议 (2026-09-06)

来源: 运维侧重另一个 agent 对 migration-debt 第四臂的评审。基础是
extension/README.md 的 6 个 UVM hook 与 eviction_debt.bpf.c 现状。
按"离现有代码的距离"排序; ①② 小改动可直接并入第四臂, ③ 是新的跨层
机制实验, ④⑤ 是更大的架构方向。

## 1. block 层实测压力 feed, 替换 debt_pressure 代理信号 (小, 建议先做)

debt_pressure 统计的是"未消化的驱逐风险", 与磁盘实际状态没有闭环。
加一个 block tracepoint 程序 (block_rq_issue / block_rq_complete),
维护 per-device in-flight 深度与延迟直方图写入 BPF map; UVM 策略在
gpu_page_prefetch 里读实测积压决定是否抑制预取。把"猜测磁盘忙"变成
"看见磁盘忙", 顺手为 Table 1 的 kernelretsnoop 聚合提供数据源。

## 2. 落盘完成度屏障: 把 warm 标志从"信任"变成"证据" (小, 修正确性洞)

现状是 loader 在 warm 阶段"认为"写完 NVMe 后置 DEBT_CONFIG_DISK_DURABLE。
置早了策略会驱逐磁盘副本未落盘的 chunk, 崩溃后静默 KV 丢失。
拿不到 per-KV 身份没关系, 做 pool 级完成度记账: block 层按文件 extent
统计写入完成数, 覆盖整个 pool 后才允许 retroactive marking。
驱逐资格 = 已验证落盘状态; 这本身是论文卖点 (policy-grade durability
semantics), 也堵住 reviewer 最容易打的洞。

## 3. fault 驱动的 I/O 优先级提升 (中, novelty 最强)

gpubpf 独有的位置: 同时看得到 GPU fault (UVM hook) 和磁盘队列
(block 层)。当 SM 因 fault 卡在磁盘读后面、而这个读又排在 LMCache
warm 写/投机预取后面时, 单层无解。做法: UVM 策略把"正在 fault 的 VA
范围"写入 map, block 侧 eBPF (或 eBPF struct_ops I/O scheduler) 对
这些读提升优先级 = GPU fault 的 I/O 优先级继承。实验: serving 与
warm-phase 写并发, 测 p99 fault 延迟与干扰。这是只有这套栈才能做的
机制, 比"LMCache 语义注入页策略"差异化更强。

## 4. 成本分层预取: 只预取"贵"的错失 (小-中)

现有 prefetch 策略族全按访问模式 (stride/Markov/密度) 分类, 没有一个
按错失代价分类。sysmem 命中的 fault 是微秒级, 落盘路径是毫秒级。
gpu_evict_prepare 已经能看到哪些 chunk 被驱逐, 把这些 VA block 记为
"高代价", fault 时对它们激进 in-block 预取, 对 sysmem 常驻的
demand-only。关键: chunk = VA block = 2MB, 恰好落在 gpu_page_prefetch
作用域内, 完全不需要已移除的跨 block 接口。

## 5. 磁盘权威副本 / writeback 免除 (大, 架构级)

KV block 写入后不可变。磁盘副本验证落盘后, 这条数据在 sysmem/swap
路径上的任何再写出都是冗余 I/O。终极形态: 磁盘副本成为被驱逐 chunk
的权威副本, sysmem 拷贝可丢弃 (省 DRAM + 零 swap 写放大), 重载走
LMCache retrieval。注意耦合: UVM 匿名页 fault 不会自己读 LMCache
文件, 必须由 LMCache 侧拥有 reload 路径与 pinning 纪律, BPF 侧只
决定"哪些 chunk 可降级"。这是把 claim 升级为 "NVMe 作为 GPU 内存的
backing store"。风险最大, 且 1/2/3 正好都是它的前置件。

## 组合建议

| 顺序 | 事项 | 成本 | 论文价值 |
|------|------|------|----------|
| ① | block 层压力 feed | 小 | 加固第四臂 |
| ② | 落盘完成度屏障 | 小 | 正确性 + 卖点 |
| ③ | fault 驱动 I/O 优先级 | 中 | 新跨层机制 |
| ④ | 成本分层预取 | 小-中 | 与现有策略族组合 |
| ⑤ | 磁盘权威副本 | 大 | 架构级 claim |

①+②+③ 合起来是一个连贯故事: "用证据而不是提示做存储感知的 GPU
内存管理" — 压力用实测的, 落盘用验证的, 优先级按真实代价排。
相对 Bidaw/SYMPHONY 的差异化比单纯扩第四臂更强。
