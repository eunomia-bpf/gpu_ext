# Closest-policy overlap audit

This experiment combines established scheduling ideas on a different,
already-measured actuator.  It must not be described as inventing deadline
urgency, bubble harvesting, or starvation protection.

## GPreempt

GPreempt supplies the mechanism used here: two CUDA contexts, asymmetric
timeslices, a blocking kernel, an early hint and a host-visible release flag.
Its evaluated userspace policy preempts for every LC request.  The completed
local knee sweep shows that this fixed choice preserves LC response at a large
and growing BE-goodput cost.  The new experiment changes only admission to
that actuator and retains fixed GPreempt as the strongest matched baseline.

Local source: `docs/paper-material/ref-paper/gpreempt_atc25.pdf`.

## Hummingbird

Hummingbird defines SLO as exclusive-execution P99 and admits low-priority
split kernels only when no high-priority kernels are pending and a bubble is
detected.  Its one-outstanding-kernel bound limits preemption latency; a tick
pipeline and consolidation reduce synchronization overhead.  The completed
local port already tests its idle/bubble component and shows that a conservative
executor can lose BE throughput.

The proposed policy does not split kernels, detect device bubbles, predict idle
intervals, or reproduce Hummingbird.  It borrows the isolated-P99 SLO convention
but instead decides whether to invoke GPreempt from request slack and observed
BE completions.

Local source: `docs/reference/2026-hu-hummingbird-v2.pdf`.

## UrgenGo

UrgenGo is the closest conceptual overlap.  It defines urgency as inverse task
laxity using arrival time, deadline and estimated remaining GPU work, then uses
dynamic stream binding and delayed kernel launch.  It also uses high-urgency
thresholds and periodically reevaluates urgency in asynchronously launched task
chains.

Therefore “deadline/slack-aware scheduling” is prior art.  This experiment's
independent question is narrower: whether a two-threshold urgency gate composed
with a verified BE-completion floor improves the measured fixed-GPreempt
LC/BE tradeoff, and whether native C and host-JIT BPF carry that identical
stateful policy.  It does not reproduce UrgenGo's task graph, remaining-kernel
estimator, stream pool, delay loop, ROS integration, or autonomous-driving
evaluation.

Local source: `docs/reference/2025-zhu-urgengo-v1.pdf`.

## UniBoost

UniBoost applies soft, continuous arrival-time/service-based priority shaping
to LLM requests.  Its MemGuard geometrically limits priority revisions and its
boost rule prevents long-request starvation while adapting across workload
distributions.  That work motivates tail-aware hysteresis and confirms that a
hard always-preempt rule need not dominate.

The proposed experiment schedules one LC model against one continuous BE model;
it has no tokens, KV eviction, request-size distribution, boost function, or
gamma adaptation.  Its one-completion floor is not UniBoost's policy and should
not be compared numerically with UniBoost.

Local source: `docs/reference/2026-li-tail-aware-scheduling.pdf` (official ICML
2026 paper downloaded from the authors' project page).

## Claim boundary

The strongest honest label is **agent-designed cross-policy composition on the
gpubpf/GPreempt hook**, not a fundamentally novel scheduler.  Positive data can
show useful expressibility and a policy/mechanism separation.  Negative data can
show that the current signals and blocking actuator are insufficient.  Neither
outcome establishes general deadline guarantees, fairness, or faithful
reproduction of the three closest systems.
