# Existing XSched workload: service-only output adapter

The opt-in `XG_SERVICE_ONLY=1` output branch is built successfully. The root
added only 23 lines of bounded output glue; the CUDA computation and existing
default output path are unchanged. The local Qwen task remains responsible
for the three-arm Level-2 runner, not this output adapter.

The original ten positional arguments remain. Use zero for the unused final
offset argument in service-only mode. Result JSON contains:

- `metric_scope: gpu_service_and_host_elapsed`, `service_only: true`;
- `start_host_ns`, `completion_host_ns`, `host_elapsed_ns`;
- `samples` with `entry_ns`, `exit_ns`, `service_ns`, `blocks_done`;
- existing `role`, `process_id` and `outputs_validated`.

GPU service span is exit minus entry. It can include pauses and the spread
between blocks; it is not active-device execution time or arrival queue time.
Host elapsed spans GO-to-stream-completion, before host output checking.
No arrival-to-GPU subtraction is performed in this output mode.

Built binary (1054184 bytes; not committed):
`.output/service-output-20260908.66ppYl/priority_workload`.
The old `workloads/xsched/build/priority_workload` remains untouched.

Build from `workloads/xsched`, under both shared leases:

```sh
/usr/local/cuda-12.9/bin/nvcc -ccbin /usr/bin/g++-13 -O3 -std=c++17 -lineinfo -gencode arch=compute_120,code=sm_120 -I/usr/local/cuda-12.9/targets/x86_64-linux/include priority_workload.cu -L/usr/local/cuda-12.9/targets/x86_64-linux/lib -Xlinker=-rpath -Xlinker=/usr/local/cuda-12.9/targets/x86_64-linux/lib -lcupti -o level2-build/.output/service-output-20260908.66ppYl/priority_workload
```

`build.log` is empty because nvcc exited zero without diagnostics. No GPU
measurement, driver change or completed-cell replay accompanied this build.
Next use is the existing native/BPF Level-2 tool-route comparison. This
component build is not evidence that that full comparison has completed.
