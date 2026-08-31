你看看最新的 gpubpf 论文的 revision 要求, 去把实验按照 revisioj要求推进, 不能停下来, 我们'/home/yunwei37/workspace/gpu/gpu_ext/docs/paper/asplos-27-rebuttal/revision-plan.md'

现在是什么模块

DKMS 是啥> 为啥不是 open kernel module

能不能开一个 branch port 过去

继续

确保本地的做多少就要 commit push 多少

我们还要求做什么实验? 不止这几个吧? 你看看

项目          MoE-Infinity
   已经做了什么  完成同模型 GPT-OSS-120B 的四配置实验设计；模型、GGUF、NVMe、驱动、BTF、
                 hook admission 全部通过；28 个离线测试和 23 个 ABI 断言通过；执行了第 1
                 次真实 GPU preflight
   结论          加载模型并建立约 61 GiB expert store 后，512-token warm-up 有 353 行路由
                 到同一 expert，超过上游固定 256 行 buffer。请求未完成，因此没有正确性或性
                 能样本。详见 gpu_ext/workloads/moe-infinity/runtime-preflight.md 要修好

The revision plan overall looks good. Please ensure other clarifications presented in the original author response are integrated into the paper where appropriate.

Policy versus mechanism

In addition to expressibility/feasibility of the mechanism, it would be great to improve the overall exposition, which has not come up quite as clear in the paper. Specifically, please make the below points clear:

Whether your mechanism can implement many of the existing policy (which you say you will do)

When implementing an existing policy using your mechanism, how does the performance compare with the original ad-hoc/unsafe/monolithic implementation.

Just to be clear: there's definitely value in having a safe/flexible mechanism even if all the case studies one presents only match the SOTA, rather than surpassing it.

But to help the readers put the results in perspective, it would be good to explicitly discuss whether the headline improvement numbers throughout (especially in abstract and introduction) come from the policy or the mechanism, and if this more general mechanism has any performance drawbacks when implementing an existing policy. This is important to avoid any perceived misleading statement.

Call out any new policy enabled by/implemented in your mechanism (especially any interesting policy discovered by an agentic workflow). This sort of insights would be very useful to the broader community when we think about future systems design.
Another suggestion you can consider: in your original response, there are answers to common themes of questions from reviewers (e.g., safety, or comparison/reimplementation of SOTA) that scatter across multiple subsections. It might help readers navigate the discussion to group relevant points under common headings, and label them accordingly.

Typographic nits:

In section 5.3, paragraph headings have two periods at the end.
The curly braces in the bib don't seem to be escaped properly.
Please let me know if you any question about the reviews. 这个记录下来原话到仓库里

Author [Yusheng Zheng]·Aug 3
We thank the reviewers and the committee. We will implement these changes:

Q1, state-of-the-art baselines (Reviewers E, F). We will add at least three runnable state-of-the-art research baselines, such as MoE-Infinity, XSched, and an extension of the existing LMCache comparison to its local-disk backend. When an artifact cannot run on our hardware but its policy fits our hooks, we will implement the policy instead, as we already do for GPREEMPT's priority timeslicing (Fig. 12) and will do for Expert Buffering's hot-expert residency. Where neither route is open we will name the system and say why.

Q2, safety and design depth (Reviewers B, F). We will add transition-validation pseudocode, the SIMT verifier algorithm, examples of rejected policies, a failure-mode taxonomy, and an explicit account of the trusted computing base to the design and implementation sections.

Policy versus mechanism (Reviewer F). We will add a policy expressibility table separating what is feasible in user space, what requires driver modification, and what gpubpf supports, and we will expand Fig. 13. For each policy class we will state which of the two the result depends on.

Measurements, artifacts, and discussion. We will add RTX 5090 numbers to Table 1 (Reviewer A) and release the agent prompts and benchmark harnesses (Reviewer E). The text will also address thrashing under stale state, CXL tiers, per-tenant policies, trampoline scaling, and portability (Reviewers A, D), and will distinguish our software co-location setting from the static partitioning the SemiAnalysis critique targets(Reviewer E).

@A2Shepherd·Aug 14
The revision plan overall looks good. Please ensure other clarifications presented in the original author response are integrated into the paper where appropriate.

Policy versus mechanism

In addition to expressibility/feasibility of the mechanism, it would be great to improve the overall exposition, which has not come up quite as clear in the paper. Specifically, please make the below points clear:

Whether your mechanism can implement many of the existing policy (which you say you will do)

When implementing an existing policy using your mechanism, how does the performance compare with the original ad-hoc/unsafe/monolithic implementation.

Just to be clear: there's definitely value in having a safe/flexible mechanism even if all the case studies one presents only match the SOTA, rather than surpassing it.

But to help the readers put the results in perspective, it would be good to explicitly discuss whether the headline improvement numbers throughout (especially in abstract and introduction) come from the policy or the mechanism, and if this more general mechanism has any performance drawbacks when implementing an existing policy. This is important to avoid any perceived misleading statement.

Call out any new policy enabled by/implemented in your mechanism (especially any interesting policy discovered by an agentic workflow). This sort of insights would be very useful to the broader community when we think about future systems design.
Another suggestion you can consider: in your original response, there are answers to common themes of questions from reviewers (e.g., safety, or comparison/reimplementation of SOTA) that scatter across multiple subsections. It might help readers navigate the discussion to group relevant points under common headings, and label them accordingly.

Typographic nits:

In section 5.3, paragraph headings have two periods at the end.
The curly braces in the bib don't seem to be escaped properly.
Please let me know if you any question about the reviews.

Add comment 我们要优先把这些都做完

按照这个做, 别的不做, 先不要改论文, 原话保存下来

然后实验先去补充完整, 按照 auto research 的标准去迭代

任何 skills 里面都要提及禁折腾 hash, 本地也要禁止任何折腾 hash 的东西
