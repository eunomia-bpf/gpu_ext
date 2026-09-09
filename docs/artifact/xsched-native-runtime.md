# Native XSched Level-2 reproduction

The [native runner](../../workloads/xsched/level2/native/run_native_blob.py)
accepts explicit workload, HAL and server locations. From the repository
root, replace the quoted placeholders with your built components and a new
output directory:

```sh
python3 -B workloads/xsched/level2/native/run_native_blob.py run \
  --workload "WORKLOAD_BINARY" \
  --hal-install-dir "NATIVE_HAL_INSTALL" \
  --xserver-native "NATIVE_XSERVER_BINARY" \
  --reps 9511106 --tasks 50 --blocks 340 --threads 256 \
  --repetitions 1 --output "NEW_OUTPUT_DIRECTORY"
```

This launches GPU work; it is not a build command. On the shared host, hold
both leases listed in [ARTIFACT.md](../../ARTIFACT.md). The runner builds
nothing and keeps the historical component locations as its defaults.
Overrides reach the component lookup, worker library paths, server command
and recorded protocol without changing module globals or the native
cuXtra actuator. There is no NVBit/BPF fallback on this route.

The [completed native bring-up](../../workloads/xsched/raw/level2-native-retabs-20260909.oJDCbU/README.md)
is one cell: six workers exit zero, 400 LC / 800 BE service records and a
resume launch in each BE process. It is not a matched multi-arm campaign.
The earlier [shared-actuator policy-port comparison](../../workloads/xsched/raw/level2-device-policy-pair-20260909.buBjns/README.md)
is a separate result and must not be relabeled as this original actuator.

Use the [native source instructions](../../workloads/xsched/level2/native/README.md)
and [component build guide](../../workloads/xsched/level2-build/README.md)
for preparation. The [eight-patch preparation record](../../workloads/xsched/raw/native-source-prep-20260909.0g1Nxc/README.md)
identifies the additional recorded queue diagnostics and submodule sources
still being packaged; source preparation alone is not a completed fresh build.
The portable path change was checked on CPU without repeating a GPU cell.
