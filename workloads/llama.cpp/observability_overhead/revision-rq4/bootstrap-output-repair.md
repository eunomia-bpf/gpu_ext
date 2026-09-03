# Table 1 bootstrap-output repair and diagnostic review — 2026-09-03

Status: actual private runtime rebuilt; CPU output checks pass. The
[second three-cell untimed diagnostic](targeted-diagnostic-results.md) now
passes exact stdout and aggregate counts after the additional PTX-pass fix;
the first failed diagnostic is retained. No performance cells ran. The failed
seven-arm [preflight 03](preflight-575-results.md) remains unchanged.

## Implemented small step

The runtime now configures its existing logger once before CUDA fatbin
registration and before shared-memory initialization. This preserves the
existing `BPFTIME_LOG_OUTPUT` destinations: console goes to stderr, a path goes
to that file, and an unset target stays quiet. Its later loader-derived logging
configuration still applies. The Linux `cuobjdump` extraction child alone now
redirects diagnostic stdout to stderr; application stdout is not redirected,
filtered, truncated, or otherwise normalized beyond the existing oracle.
The controlled [runtime patch](runtime-575/runtime-575.patch) includes both
changes and its new bootstrap header; reverse-apply checking passes against
the actual private source. The main bpftime and strict R5 trees were not edited.

The [CPU check](runtime-575/test_bootstrap_output.py) compiles the **actual new
runtime header and actual VM registry header**, then causes duplicate VM
registration. Console/file/unset logger modes each preserve exact
`application-output` stdout; configured diagnostics remain observable. This
checks logger routing, not CUDA attachment or whole-application correctness.
The original test output is [bootstrap-cpu.log](../../../../docs/experiment/revision-safety/table1-runtime-build-575-03/bootstrap-cpu.log).

## Targeted untimed entry point

[`run_targeted_diagnostic.py`](run_targeted_diagnostic.py) reuses the existing
`run_correctness_cell`, private loader/segment, owned cleanup, safety telemetry
and shared leases. It runs the unchanged eight-token correctness command,
not llama-bench, and never builds tools or resumes/overwrites an output directory.
The main entry also installs/restores `run_cmd_owned`, as the formal runner does.

- Default histogram sequence: native baseline, current NVBit histogram, then
  gpubpf histogram. Each output must exactly match the new baseline using the
  original normalization; every original per-tool engagement check remains.
  Fresh aggregate sample and nonzero-thread counts are compared after all
  three pass. A difference is unresolved counting semantics/coverage, **not
  automatically BPF loss**. Matching totals alone do not prove per-launch or
  per-slot coverage.
- Separate optional launch-latency sequence: native baseline, then gpubpf
  launchlate, with `SPDLOG_LEVEL=info` recorded for both loader and target.
  Zero host correlation, underflow/overflow or clock errors still fail the
  original engagement check. No clock offset or latency performance is inferred.
- Any failed correctness/engagement check stops the sequence; an exception or
  interrupt retains its error and end time before releasing the existing lease.
  Unconfirmed CUDA cleanup still preserves the private loader/segment through
  the existing fatal-cleanup path. There is no retry or seven-arm fallback.

The closed preflight-03 manifest supplies unchanged workload parameters and
already-built BPF probe paths. The default NVBit path is the separately rebuilt
[`observability.so`](nvbit_adapters/observability/observability.so), not the old
preflight library: see the [EXIT-predicate repair](nvbit-exit-predicate-repair.md)
and commit `52ab4d9`. Its recorded build size is 3,183,016 bytes. Historical
901,120 versus 720,896 totals are diagnostic leads, not a trusted reference
count. No new NVBit build is performed by this entry point.

`diagnostic.json` records current file sizes/timestamps, not copied runtime
metadata from preflight 03: current agent/server libraries and agent object,
runtime patch/report (`8f7d2d5`), selected BPF binary and its prepared C/BPF
sources, current NVBit library/sources/repair note, and the client/model paths.
Each cell retains the original command, stdout/stderr and safety/probe records.
It is explicitly **not full preflight or performance evidence** and does not
create a resumable formal `result.json`. Exit 0 means only these selected
diagnostic checks passed; exit 2 preserves a failed check/count disagreement.
The original full seven-arm correctness and performance gates are unchanged.

Root-only launch examples from the repository root, after exclusive GPU
admission. The first has now run as recorded above; launchlate has not.
Keep the coordinator's normal affinity for the CPU-16 telemetry worker:

```sh
sudo -n env PATH=/usr/local/cuda-12.9/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  CUDA_HOME=/usr/local/cuda-12.9 \
  python3 -B \
  workloads/llama.cpp/observability_overhead/revision-rq4/run_targeted_diagnostic.py \
  --task threadhist \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 \
  --nvbit-tool workloads/llama.cpp/observability_overhead/revision-rq4/nvbit_adapters/observability/observability.so \
  --output-dir workloads/llama.cpp/observability_overhead/revision-rq4/raw/diagnostic-histogram-575-01

sudo -n env PATH=/usr/local/cuda-12.9/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  CUDA_HOME=/usr/local/cuda-12.9 \
  python3 -B \
  workloads/llama.cpp/observability_overhead/revision-rq4/run_targeted_diagnostic.py \
  --task launchlate \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 \
  --output-dir workloads/llama.cpp/observability_overhead/revision-rq4/raw/diagnostic-launchlate-575-01
```

CPU verification: `taskset -c 17 python3 -B -m unittest -q
test_targeted_diagnostic test_offline` in this directory passed **34 tests**
(10 targeted plus the unchanged 24 offline checks). The new tests use CPU
fixtures/mocks, including exact-output rejection, count disagreement,
launchlate error preservation, info logging at both endpoints, fresh inventory,
no build/timing calls, output refusal and SIGTERM/helper/lease restoration.
Root independently reran the same 34 tests successfully. This is preparation
evidence only; no GPU, service, original raw or dependency was changed here.

The first incremental [build attempt](../../../../docs/experiment/revision-safety/table1-runtime-build-575-02/build.log)
printed duplicate logger-definition errors because the old logger header lacked
an include guard and the agent included it twice. Although the build command
returned zero and linked, its agent object still predated the edit: this attempt
is **invalid**, not a successful build. Why that compiler path masked failure
has not been established. Removing the redundant direct include fixes the
source. The prior object was retained outside the build at
`/tmp/gpubpf-table1-prior-object-JzWC8Y/agent.cpp.o`; no source or result was deleted.
The [next build](../../../../docs/experiment/revision-safety/table1-runtime-build-575-03/build.log)
used `CCACHE_DISABLE=1`, rebuilt 12 steps, exited zero with no compile errors,
and produced a new 4,311,064-byte agent object containing the bootstrap-logger
symbol and a new 242,981,200-byte agent library. This proves the source reached
the built artifact, not that a new GPU correctness check passed.

## OpenCode findings and root corrections

Session `ses_f98bad72dffeaaFk502Y2khNdN` used the configured default model.
After a 31-minute read phase was stopped (exit 130, all raw events retained),
the **same session** resumed with every tool denied and returned the complete
report below, exit zero. Local streams remain at
`/tmp/gpubpf-opencode-table1-diagnostic-OjOLJa/{events,final-only}.jsonl`.

Root rechecked the implementation rather than accepting its causal labels:

- Raw stdout visibly contains VM-registration and extraction diagnostics.
  Their rerouting is implemented above; complete llama-cli output remains to
  be verified on the new binary. Lack of `agent.log` alone does not prove all
  logger configuration failed: loader and agent destinations differ.
- Zero host-launch counters with 220 device underflows establish missing host
  correlation, **not uniquely that the uprobe callback never ran**. A failed
  lookup or wrong shared state could also produce missing increments. Loading
  order is a hypothesis to test using repaired diagnostic logging.
- Host `CLOCK_REALTIME` and device `%globaltimer` have not been correlated in
  this run. All 220 NVBit entries fail the ordering test. Do not guess an epoch,
  clamp deltas, or subtract arbitrary offsets; both paths need actual clock
  correlation and its uncertainty before latency can be interpreted.
- The exit path's configured 8,192 logical-thread slots cannot cover a launch
  whose histogram reaches 22,528 distinct slots. `default_trampoline.cu` drops
  out-of-bound IDs. Its per-thread buffer also returns an error when full;
  it does **not** deliberately overwrite old entries as one reading of the
  advisory wording might suggest. One-second polling drains at most one entry
  per slot per call. Size, produced/dropped counts and complete drain must be
  repaired before comparing full event-collection cost.
- `901120 = 220 * 4096` is arithmetic, **not evidence each launch had 4096
  threads**. Launch shapes can differ. The separate BPF histogram's 20% deficit
  still needs a per-launch/control-flow audit; ring-buffer limits do not explain
  an array-histogram deficit by themselves.

## Complete OpenCode report, retained as returned



**诊断报告（仅读证据，未改动任何文件）**

**一、stdout污染（已证）** gpubpf单元格客户端stdout混入约24KB诊断：默认spdlog logger（stdout sink，带毫秒模式）在agent.cpp:831 bpftime_set_logger生效前输出VM factory告警（bpftime_vm_compat.hpp:251，llama_cli.log第5行起），且INFO级(ptxpass core.cpp:321, 05:34:12)穿透SPDLOG_LEVEL=warn，agent.log未生成→配置logger路径未生效；"Extracting PTX file"×120为cuobjdump/ptx提取子进程继承fd1直写stdout。应用输出本身仍在（第250-251行），但gate按精确比较判valid=false——正确拦截，非丢行。支持途径：BPFTIME_LOG_OUTPUT=console→stderr（bpftime_logger.hpp:64-68、bpftime_config.cpp:154）；候选最小修复：logger在任何SPDLOG调用前初始化+子进程stdout重定向到日志fd；须重跑correctness验证stdout==47B。

**二、launchlate（已证+推断）** gpubpf：probe.log:65-73显示uprobe符号解析0x221730并attach成功，但Host launches=0/underflows=220→host侧处理器从未触发。设备侧220次kprobe与launch数一致。未证主因候选：agent在dlopen libggml-cuda前attach（frida_uprobe_attach_impl.cpp:222-226模块不在maps即-EINVAL，INFO级无agent.log可查）→需开info日志重跑区分。NVBit：clock_errors=220==selected_launches（inject_funcs.cu:70-74）：host用CLOCK_REALTIME（observability.cu:56-62,260）对比%globaltimer，全部gpu<host，且probe.log:68-70 MONOTONIC≈1.26e13≪REALTIME≈1.79e18→跨时钟域（globaltimer疑boot纪元，未直接测值）。gpubpf修复host hook后同域问题会同样爆发。候选修复：启动期标定globaltimer↔host偏移写入clock_offset，保留errors/完整计数要求；gate（run_revision_rq4.py:551-558,497-501）保持不弱化。

**三、事件量与公平性（已证部分）** NVBit 901120=220×4096全nonzero；gpubpf仅16384=2轮poll×8192（kernelretsnoop.c:90每秒1次；nv_gpu_ringbuf_map.cpp:61-77每次每slot只取1条，slot环仅max_entries=16条、无丢失计数）→容量/poll节奏下覆盖必然不等，paired比较对该单元格无效；且run_revision_rq4.py:1039默认8192 thread slots，NVBit实测22528 nonzero slot，default_trampoline.cu:367越界丢弃→按构造丢覆盖（root已证）。是覆盖差、环覆盖写丢还是poll停滞（probe.log仅2条计数、后续sticky打印缺失）未定论，不得定因；需producer侧tail计数与加密poll+无缓冲输出区分。结论：当前gpubpf事件量≪NVBit，overhead公平性不成立；须修复slot容量/丢弃计数后新建preflight重跑。
