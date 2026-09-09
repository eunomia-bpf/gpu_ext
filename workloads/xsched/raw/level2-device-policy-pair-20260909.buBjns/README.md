# XSched Level-2 host/device policy comparison

Continue the repaired NVBit-actuator path with five blocks and three arms:
baseline (no XSched), native_port (original host HPF plus device C guardian
policy port), bpf_port (bpftime host HPF plus device eBPF guardian decisions).
Both policy arms use the same trusted NVBit actuator and runtime HAL.
This is a policy-port comparison, not the original cuXtra artifact running
on sm_120. That native SASS route remains separately under repair.

Reuse the successful native_port sample from
../level2-perlaunch-enable-20260909.bOiYh5/cells/block-01-native_port as
global block 0. run.sh runs only BPF and baseline for that block, then four
additional rotated three-arm blocks. Its first block order is
native/BPF/baseline; the reused native observation precedes the remaining
arms by a publication/coordination gap, which must be retained as a timing
limitation. Do not silently exclude it or run it again.

Global blocks 1..4 map directly to local block indices in remaining;
block0-missing's local block 1 supplies the two missing global block-0 arms.
All use 6 processes, 24 streams, 50 tasks/stream, 340 CTAs, 256 threads,
9511106 recurrence repetitions and the same 5 ms LC start delay.

Primary observations are LC GPU-service p99 and BE completed kernels/s,
with host elapsed times retained separately. This does not measure full
arrival/queue latency. The XG5 device probe and existing host diagnostic
logging remain enabled in both policy arms; resulting native/BPF costs
include those diagnostics and are not isolated BPF instruction overhead.
The baseline has no instrumented guardian. Completed Level-1 measurements
remain unchanged.

The corrected host tool and XG5 carrier are the exact built artifacts from
bOiYh5, with no rebuild between measurements. Both experiment leases cover
the run; the prior tool is restored on exit. No module change, calibration,
clock check, extra gate or new paper experiment is introduced. No manuscript
is edited. Runtime results are pending.

## First complete block (remaining four blocks running)

All three global block-0 arms now finish 1200 kernels and six worker exits
zero, with no sample_diagnostics. LC service p99 is baseline 2183.887968 ms,
native 760.289344 ms, BPF 767.745984 ms. BE throughput is respectively
10.195219, 9.894779 and 9.897796 kernels/s. Thus the first block shows a
large LC tail improvement with lower BE throughput; BPF/native LC p99 is
about 0.98% higher. The reused native timing gap and enabled diagnostics
remain limitations. These are partial-campaign observations, not the
completed five-block result. The other four blocks continue unchanged.

Global block 1 also completes all three arms and all worker exits zero:
baseline/native/BPF LC service p99 is 2168.047584/769.950592/793.632992 ms;
BE throughput is 10.199266/9.891486/9.874405 kernels/s. This second block
retains the same tail/throughput tradeoff; BPF's LC p99 is higher than native.
Six of fifteen cells are complete. The remaining three blocks continue.

Global block 2 completes all three arms: baseline/native/BPF LC service p99
is 2089.770304/786.260608/892.468448 ms; BE throughput is
10.181734/9.889721/9.909920 kernels/s. All six workers in each arm exit
zero, with no sample_diagnostics. The larger BPF/native tail difference in
this block is retained, not excluded. Nine of fifteen cells are complete;
the final two blocks continue.
