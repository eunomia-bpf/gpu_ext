# Corrected OpenCode review

> NVBit 在 1s 下保守界约 4700–5100 ppb（5 μs 区间宽度 ÷ 1s），加上实际相对漂移率后仍可能突破 10,000 ppb 门槛，故修复不保证充分。gpubpf 的 89.5% uncertain 由分类器定义决定，端点负趋势已通过仿射插值且漂移界 1590 ppb 合格，在无逐样本区间数据时不可归因于末端括号失误。Driver RM 相关性（NV2080 gpuTime + PLATFORM_API，含最接近 CPU pair 中值）是原则上的下一步，需集成 RM handle/control 并暴露界定 closest-pair width。CUPTI 因合约不保证 raw PTIMER/%globaltimer 域而被拒；compile-only diagnostic 仅测局部排序，未经运行，不能据以确认任何不可靠性。

The first two model responses contained unit and causal-attribution errors.
They are not adopted. The paragraph above is the final response after both
corrections; repository conclusions still rest on source, documentation, raw
logs, and executable tests rather than the reviewer.

