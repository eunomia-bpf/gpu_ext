#!/usr/bin/env python3
"""Finite owned-context/GDRCopy smoke, not GPreempt scheduling performance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("gpreempt_smoke_safety", ROOT / "workloads/moe-infinity/run_moe_head_to_head.py")
assert SPEC and SPEC.loader
safety = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = safety
SPEC.loader.exec_module(safety)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    lease = None
    result = {"status": "running", "kind": "GPreempt-575 owned-context/GDRCopy smoke",
              "performance_claim": False, "timeout_seconds": 30}
    process = None
    before = None
    try:
        lease = safety.LeaseSet.acquire()
        before = safety.safety_snapshot()
        safety.validate_pre_server_safety(before)
        if before["gpu"]["driver"] != "575.57.08":
            raise RuntimeError("this smoke requires the separately prepared 575 compatibility driver")
        if not Path("/dev/gdrdrv").exists():
            raise RuntimeError("GDRCopy device is not available; this runner never loads modules")
        executable = HERE / "build/ninja/test-basic"
        with (output / "smoke.log").open("x") as stream:
            process = subprocess.Popen(["taskset", "-c", "8-15", str(executable)],
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
            result["returncode"] = process.wait(timeout=30)
        log = (output / "smoke.log").read_text(errors="replace")
        facts = {
            "timeslice_request_accepted": "PASS set priority" in log,
            "gdr_flag_roundtrip": "PASS GDRcopy flag roundtrip" in log,
            "cleanup_passed": "PASS all (finite smoke only; not scheduling performance)" in log,
        }
        result["checks"] = facts
        if result["returncode"] != 0 or not all(facts.values()):
            raise RuntimeError("original-policy compatibility smoke failed; retain log and result")
        result["status"] = "passed"
        result["effective_timeslice_verified"] = False
    except BaseException as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        try:
            if process is not None:
                safety.stop_owned_process_group(process)
            if before is not None:
                result["safety_after"] = safety.wait_for_post_server_safety(before)
            result["safety_before"] = before
        except BaseException as exc:
            result.update(status="failed", cleanup_error=str(exc))
            raise
        finally:
            safety.atomic_write_json(output / "result.json", result)
            if lease is not None:
                lease.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}; cleaning up owned smoke process")
    signal.signal(signal.SIGTERM, interrupted)
    print(json.dumps(run(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
