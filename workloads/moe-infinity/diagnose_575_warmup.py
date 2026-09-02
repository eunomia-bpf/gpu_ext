#!/usr/bin/env python3
"""Capture one owned 575 MoE warm-up native backtrace; never a timing sample."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import run_575_head_to_head as current
import run_moe_head_to_head as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--cuda129-triton", action="store_true")
    args = parser.parse_args()
    lease = base.LeaseSet.acquire()
    server = None
    log = None
    before = None
    result = {"timing_sample": False, "diagnostic": "gdb-native-warmup"}
    try:
        admitted = current.require_admission(18080)
        before = admitted["safety"]
        args.output.mkdir(parents=True, exist_ok=False)
        base.atomic_write_json(args.output / "admission.json", admitted)
        command, cwd = base.server_command("moe_infinity_075", 18080, args.output, args.store)
        command = command[:3] + ["/usr/bin/gdb", "-q", "-batch",
                                  "-ex", "set pagination off",
                                  "-ex", "set print thread-events off",
                                  "-ex", "run", "-ex", "thread apply all bt 20",
                                  "-ex", "quit", "--args"] + command[3:]
        base.atomic_write_json(args.output / "launch.json", {"command": command, "cwd": str(cwd)})
        log = (args.output / "gdb.log").open("x", buffering=1)
        environment = base.controlled_environment("moe_infinity_075")
        if args.cuda129_triton:
            environment.update(TRITON_PTXAS_BLACKWELL_PATH="/usr/local/cuda-12.9/bin/ptxas",
                               TRITON_PTXAS_PATH="/usr/local/cuda-12.9/bin/ptxas",
                               TRITON_CACHE_DIR=str((args.output / "triton-cache").absolute()))
        result["environment"] = environment
        server = subprocess.Popen(command, cwd=cwd, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        base.wait_ready(server, 18080, args.output / "gdb.log", 1800)
        tokens = json.loads(base.PROMPTS.read_text())["records"][0]["prompt_token_ids"]
        try:
            result["response"] = base.nonstream_completion("moe_infinity_075", 18080, tokens,
                                                             args.output / "warmup.json", timeout=600)
            result["status"] = "completed"
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            # gdb writes the stopped threads before exiting; let that live handle finish.
            try:
                result["gdb_exit"] = server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                result["gdb_exit"] = "backtrace timeout"
    finally:
        if server is not None:
            base.stop_owned_process_group(server)
        if log is not None:
            log.close()
        if before is not None:
            result["post_safety"] = base.wait_for_post_server_safety(before)
        if args.output.exists():
            base.atomic_write_json(args.output / "result.json", result)
        lease.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
