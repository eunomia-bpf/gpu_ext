#!/usr/bin/env bash
set -euo pipefail
native_raw=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-native-resolver-snapshot-20260909.KKFqHz
workloads/xsched/deps/xsched/output/bin/xserver HPF 50000 > "$native_raw/xserver-layout.log" 2>&1 &
native_server_pid=$!
trap 'kill -INT "$native_server_pid" 2>/dev/null || true; wait "$native_server_pid" || true' EXIT
date -Is
gdb --batch -x "$native_raw/load-layout.gdb" \
    workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload \
    > "$native_raw/layout.log" 2>&1
date -Is

