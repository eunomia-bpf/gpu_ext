先把结论说清楚：

* 你现在这条：
  `--num-prompts 300 --request-rate 20`
  更像是一个「冒烟测试」配置，不是论文里常用的 evaluation setup。
* 想做得更像最近 KV offload / serving 论文，建议：

  * 把 `--num-prompts` 提到 ≥1000；
  * `--request-rate` 不要只用一个点，而是扫一组典型值，比如：`{2.5, 5, 10, 15, 20}`，其中 **10 rps** 是很多项目/论文默认的“典型负载”，**20 rps** 更偏“高负载 / 压力测试”。

下面细讲——包括我从公开代码和论文里看到的典型配置。

---

## 1. 先搞清楚 vLLM bench serve 参数语义

`vllm bench serve` 里关键的两个东西：

* `--request-rate`：平均请求到达率（req/s）

  * 默认是 `inf`，即一次性把所有请求打满，相当于“离线最大吞吐”测试。([VLLM Documentation][1])
  * 设为有限值时，用 Poisson / gamma 分布合成到达时间，模拟在线流量。([VLLM Documentation][1])

* `--num-prompts`：从数据集中抽多少条请求发给服务端。

  * 整个 benchmark 的总时长大约是：`num_prompts / request_rate`（再加上尾部长尾请求的时间）。

你现在这条：

```bash
uv run vllm bench serve \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --dataset-name sharegpt \
  --dataset-path .../ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 300 \
  --request-rate 20
```

理论上平均跑完的时间 ≈ `300 / 20 = 15s` 左右（加上预热和长尾，大概也就几十秒）。这对 smoke test 够了，但对“论文级”结果来说时间太短，不够稳定。

---

## 2. 近期项目 / 论文里 ShareGPT + online serving 的典型配置

### 2.1 KV / 内存相关系统 & 工具

1. **kvcached（Virtualized Elastic KV Cache）**

   GitHub 里的简单 benchmark 脚本就是用 vLLM + ShareGPT：([GitHub][2])

   ```bash
   vllm serve meta-llama/Llama-3.2-1B --port=12346 ...
   vllm bench serve \
     --model meta-llama/Llama-3.2-1B \
     --request-rate 10 \
     --num-prompts 1000 \
     --dataset-name sharegpt \
     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
   ```

   典型点：

   * `num-prompts = 1000`
   * `request-rate = 10` req/s

2. **LightLLM 的 ShareGPT benchmark**

   官方文档的 ShareGPT 性能测试脚本也是：([Lightllm][3])

   ```bash
   python benchmark_sharegpt.py \
     --num_prompts 1000 \
     --request_rate 10.0
   ```

   一样是 `1000` 条、`10 rps`。

3. **CacheOPT / Cache competition 论文**

   CacheOPT 这种 KV 竞争调度工作，用 trace 模拟不同到达率。对 ShareGPT 的 arrival rate 范围一般是 **0.8–1.4 req/s** 这样偏“真实负载”的区间，对 Alpaca 之类 synthetic 会拉到 20–32 req/s 做压力测试。([ResearchGate][4])
   这里不直接用 `vllm bench serve`，但给了一个“到达率量级”的参考：几十 req/s 已经算很高载了。

4. **Oneiros / MIRAGE（Parameter remapping for KV cache）**

   针对 OPT-13B + ShareGPT，他们是“多点扫 arrival rate”的做法：

   * 单模型场景：2.5, 5, 7.5, 10 req/s 一类的点；
   * 多模型/多租户（C2+ShareGPT）场景会看 10 和 20 req/s 等更高载。([UT-SysML][5])

   结论：**5–10 req/s 是常用中等负载区间，10–20 req/s 往上就是高负载 / 压力段**。

---

### 2.2 vLLM / GPUStack 一类“参考基线”文档

这类文档虽然不是做 KV offload，但给了非常具体的 ShareGPT + vLLM 配置，你可以照抄当作 baseline：

1. **Qwen3-14B on A100（GPUStack 官方实验）**

   他们 benchmark Qwen3-14B + ShareGPT 的命令：([GpuStack Docs][6])

   ```bash
   vllm bench serve \
     --model Qwen/Qwen3-14B \
     --backend openai-chat \
     --endpoint /v1/chat/completions \
     --dataset-name sharegpt \
     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
     --num-prompts 1000
   ```

   没写 `--request-rate`，所以是默认 `inf`，即“把 1000 条一次性发完”，测的是最大吞吐。最终结果里可以看到：

   * Benchmark duration ≈ 106.82 s
   * Request throughput ≈ **9.36 req/s**
   * Peak concurrent requests = 1000

   也就是说：

   * 在“打满”的情况下，**14B + ShareGPT 在 A100 上的极限吞吐量也就 9–10 req/s 的量级**。
   * 30B 模型的极限 req/s 只会更低，不会更高。

2. **GPT-OSS-20B / 120B / Qwen3-8B 等**

   GPUStack 对 GPT-OSS-20B, GPT-OSS-120B, Qwen3-8B 等一堆模型的 ShareGPT 测试全部统一为：([GpuStack Docs][7])

   ```bash
   vllm bench serve --model ... --dataset-name sharegpt \
     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
     --num-prompts 1000
   ```

   也都是：

   * `num-prompts = 1000`
   * `request-rate = inf`（默认）

这类“官方性能实验”几乎都把 `num-prompts` 设为 1000、而且跑到 1–2 分钟以上，保证统计稳定。

---

## 3. 回到你这条配置：300 prompts, 20 rps 合不合适？

拆开看：

### 3.1 `--num-prompts 300`：太少，只适合冒烟不适合论文

* 理论平均持续时间：`300 / 20 ≈ 15s`。
* vLLM 还会有一点预热，外加长尾请求，整体也就几十秒。

对“我要大致看看有没 bug、吞吐是不是任务量级”是够的；
但对“我要做 offload 论文的 baseline”：

* 时间太短，随机性比较大；
* 很容易 run-to-run 差异 > 5–10%，不太好发图/表。

**更合理的选择**（对齐别人）：

* 直接抄 GPUStack / kvcached / LightLLM：
  `--num-prompts 1000` 是现在事实上的“社区标准”。([GitHub][2])
* 如果你自己要跑多组 request-rate，还可以：

  * 正式实验：1000
  * 超长 context / 极端 case：500 / 200（不然太慢）

### 3.2 `--request-rate 20`：对 30B + 单卡来说已经是“严重过载”段

结合上面 Qwen3-14B 的结果：([GpuStack Docs][6])

* 14B + ShareGPT 在 A100 上“打满”后的 **极限吞吐** ≈ 9.36 req/s；
* 30B 模型算力需求大概是 14B 的两倍级，单卡 5090 的 FLOPs 和 A100 差不多甚至略低；
* 因此合理预期：**Qwen3-30B-A3B-FP8 + ShareGPT，在单卡 5090 上极限吞吐量大概也就 3–8 req/s 这个区间**（粗略量级判断，不是精确数字）。

在这个前提下：

* 你把 `--request-rate` 设置成 20 req/s，本质上是在“以远高于系统极限的到达率”压测；
* 结果会是：

  * 内部队列长期处于高水位；
  * P99 TTFT / TBT 飞天（几十秒甚至分钟）；
  * 如果你设置了请求超时，很容易出现失败 / drop。

这不是“错”，只是它代表的是**严重过载场景**，而不是“典型 online serving baseline”。

在 KV offload 论文里，常见的做法是：

* 选几组「从轻载到接近饱和」的到达率；
* 对每组 arrival rate 比较 baseline vs. offload 的 throughput / P99 latency；
* 过载点可以有，但不会只看一个“极端点”。

Oneiros/CacheOPT 这种的到达率设计基本都是这么干的：ShareGPT 上 0.8–10 req/s 是比较常见区间，超过 20 req/s 通常是压测/极端 case。([UT-SysML][5])

---

## 4. 结合 30B + 32GB 场景的更“像论文”的配置建议

你目前的硬件/模型组合：

* 模型：Qwen3-30B-A3B-FP8（30B FP8）
* GPU：5090 32GB（等价于“单卡大模型、显存比较紧”场景）

这是典型的“模型几乎吃满显存、KV cache 很快成为第一瓶颈”的 setup，正好适合做 KV offload 研究。

我会建议你按下面几个层次来设计实验。

### 4.1 统一 benchmark skeleton

先把“公共部分”写死：

```bash
# 服务端（示例）
vllm serve Qwen/Qwen3-30B-A3B-FP8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --no-enable-prefix-caching \
  --port 8000

# 客户端 skeleton
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --ignore-eos \
  --sharegpt-output-len 512  # 固定输出长度，减少噪声
```

你改 KV offload 的实现时，保证下面东西保持一致：

* 模型 / engine 参数；
* dataset（ShareGPT V3 unfiltered cleaned split）；
* `num-prompts = 1000`；
* 输出长度策略（比如固定 512 token）。

### 4.2 arrival-rate / request-rate 的分层设计

**1）基准点（中等负载）**

参考 kvcached / LightLLM 的做法，**10 rps** 是一个很自然的“中等负载基准点”：([GitHub][2])

```bash
--request-rate 10
```

* 这个点你一定要跑，因为可以和大量现有工作在“RPS 量级 + ShareGPT + 1000 prompts”上对齐。

**2）轻载 / SLA 友好点**

用来展示“在轻载下，offload 的开销不会把 latency 打爆”：

```bash
--request-rate 2.5
--request-rate 5
```

* 2.5/5 req/s 大概率 < 系统极限吞吐，队列不会积太多请求；
* 可以看到 TTFT/TBT 相对 baseline 的 overhead。

**3）高载 / 接近饱和点**

用来观察“靠近 GPU 饱和时，offload 能不能多撑一点”：

```bash
--request-rate 15
--request-rate 20
```

* 15 rps、20 rps 就是像 Oneiros / CacheOPT 里“高载 / 压力段”的量级。([UT-SysML][5])
* 对 30B 单卡来说，20 rps 基本必然是严重过载，你可以用这个点专门画一张“overload 下 tail latency / fail ratio”对比图。

**4）离线最大吞吐点（可选）**

如果你想跟 GPUStack 类型的文档对齐：

```bash
--request-rate inf
```

* 这时 `bench serve` 会一次性把 1000 条打满，更类似 `vllm bench throughput`；
* 用来描述“最大 token/s”，和 offload 对峰值吞吐的影响。

---

## 5. 简单回答你那条命令“多少算合适”

结合上面所有信息，直接给出判断：

1. `--num-prompts 300`

   * 只能算“方便调试的 quick test”，**不太适合当 paper baseline**；
   * 建议换成至少 `--num-prompts 1000`，和 Qwen3 / GPT-OSS / kvcached / LightLLM 等现有实验保持一致。([GitHub][2])

2. `--request-rate 20`

   * 对 **Qwen3-30B-FP8 + 单 32GB 卡** 来说，这个 RPS 几乎肯定是“严重过载”区间；
   * 适合用来看“在很大的 queueing 压力下，offload 能不能救 tail latency / 减少 OOM / 减少 drop”；
   * 但如果你只跑这一点，会很难跟 Oneiros / CacheOPT / GPUStack 这些工作对齐，因为它们一般会扫多个 arrival rate，从轻载一路扫到接近饱和。

3. 如果你要一句话版本的建议：

> * 做「像论文」的 baseline：**`--num-prompts 1000` + `--request-rate 10`** 是最合理的起点；
> * 做完整实验：在上面配置基础上再扫 `request-rate ∈ {2.5, 5, 10, 15, 20}`；
> * 你现在的 `300 / 20` 当调试配置用可以，正式结果不建议只用这一档。

如果你愿意，把你想对比的 offload 方案（vLLM 原生 CPU offload / 你自己的实现 / LMCache 类组件）列一下，我可以帮你直接设计一套“论文级” evaluation matrix（模型×到达率×max_model_len×batch_size），避免后面重复返工。

[1]: https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/ "serve - vLLM"
[2]: https://github.com/ovg-project/kvcached?utm_source=chatgpt.com "ovg-project/kvcached: Virtualized Elastic KV Cache for ..."
[3]: https://lightllm-en.readthedocs.io/en/latest/getting_started/benchmark.html?utm_source=chatgpt.com "Benchmark Testing Guide — Lightllm"
[4]: https://www.researchgate.net/publication/389946193_Mitigating_KV_Cache_Competition_to_Enhance_User_Experience_in_LLM_Inference "(PDF) Mitigating KV Cache Competition to Enhance User Experience in LLM Inference"
[5]: https://ut-sysml.ece.utexas.edu/publications/prints/socc2025_li.pdf "Oneiros: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving"
[6]: https://docs.gpustack.ai/2.0/performance-lab/qwen3-14b/a100/ "A100 - GPUStack"
[7]: https://docs.gpustack.ai/2.0/performance-lab/gpt-oss-120b/a100/?utm_source=chatgpt.com "Optimizing GPT-OSS-120B Throughput on NVIDIA A100 ..."
