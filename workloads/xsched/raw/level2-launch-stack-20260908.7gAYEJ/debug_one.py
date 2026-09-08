#!/usr/bin/env python3
"""Collect a stack for the actual first-launch crash; not a timing cell."""
import os
import subprocess
import sys
import time
from pathlib import Path

output = Path(__file__).resolve().parent
xsched = output.parents[1]
sys.path.insert(0, str(xsched / "level2"))
import run_tool_pair as run

target = "_ZN49_GLOBAL__N__c4a841cd_20_priority_workload_cu_main12compute_taskEPNS_9TaskStampEPfiiy"
workload = xsched / "level2-build/.output/service-output-20260908.66ppYl/priority_workload"
cpus = run.allowed_cpus(10)
command, environment = run.server_spec("native_port")
server = run.ManagedProcess("xserver", command, environment, cpus[0])
try:
    time.sleep(0.5)
    environment = run.worker_env("native_port", "be", target)
    preload = environment.pop("LD_PRELOAD")
    command = ["taskset", "-c", ",".join(map(str, cpus[2:])), "gdb", "-q", "--batch",
               "-iex", "set debuginfod enabled off",
               "-ex", f"set environment LD_PRELOAD {preload}",
               "-ex", "run", "-ex", "thread apply all bt 12", "--args", str(workload),
               "be", "1", "4", "50", "9511106", "340", "256", "1", "0"]
    result = subprocess.run(command, input="GO\n", text=True, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="", flush=True)
    print(f"GDB_EXIT={result.returncode}", flush=True)
finally:
    server.stop()
    run.write_process_log(output / "xserver.json", server)
