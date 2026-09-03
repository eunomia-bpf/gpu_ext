"""Temporarily pin one identified workspace OpenCode; never signal that process."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import build_adapter as owned

WORKSPACE = Path(__file__).resolve().parents[4]
SIGNALS = (signal.SIGINT, signal.SIGTERM)


def start_ticks(path):
    return int(path.read_text().rsplit(")", 1)[1].split()[19])


def identity(pid, expected=None):
    root = Path(f"/proc/{pid}")
    ticks = start_ticks(root / "stat")
    if (pid == os.getpid() or root.joinpath("comm").read_text().strip() != "opencode" or
            root.joinpath("cwd").resolve(strict=True) != WORKSPACE or
            (expected is not None and ticks != expected) or start_ticks(root / "stat") != ticks):
        raise RuntimeError("OpenCode PID/comm/cwd/start-ticks identity mismatch")
    return ticks


def threads(pid):
    result = []
    for path in sorted(Path(f"/proc/{pid}/task").glob("[0-9]*/stat")):
        try:
            ticks = start_ticks(path)
            mask = sorted(os.sched_getaffinity(int(path.parent.name)))
            if start_ticks(path) == ticks:
                result.append(dict(tid=int(path.parent.name), start_ticks=ticks, cpus=mask))
        except (FileNotFoundError, ProcessLookupError):
            pass  # A thread may exit while its immutable identity is sampled.
    if not result:
        raise RuntimeError("no live target threads")
    return result


def restore(pid, ticks, original, pinned):
    results, deadline = [], time.monotonic() + 3
    while True:
        identity(pid, ticks)
        for thread in threads(pid):
            try:
                identity(pid, ticks)
                path = Path(f"/proc/{pid}/task/{thread['tid']}/stat")
                current = sorted(os.sched_getaffinity(thread["tid"]))
                if start_ticks(path) != thread["start_ticks"]:
                    continue
                status = "already_original" if current == original else "preserved_external_mask"
                if current == pinned:
                    os.sched_setaffinity(thread["tid"], original)
                    if start_ticks(path) != thread["start_ticks"]:
                        raise RuntimeError("thread identity changed during affinity restoration")
                    if sorted(os.sched_getaffinity(thread["tid"])) != original:
                        raise RuntimeError("affinity restoration readback mismatch")
                    status = "restored"
                results.append(dict(thread, current_cpus=current, status=status))
            except (FileNotFoundError, ProcessLookupError):
                results.append(dict(thread, status="thread_exited"))
            except Exception as error:
                results.append(dict(thread, status="restore_error", error=str(error)))
        identity(pid, ticks)
        final = threads(pid)  # Also cover threads created during the first restore pass.
        if not any(t["cpus"] == pinned for t in final) or time.monotonic() >= deadline:
            return dict(actions=results, final_threads=final)
        time.sleep(0.05)


def run(args):
    record = dict(pid=args.pid, cpu=args.cpu, command=args.command, complete=False,
                  checks=0, started_utc=datetime.now(timezone.utc).isoformat(), errors=[])
    child, original, ticks, interrupted = None, None, None, None
    pin_started = False

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        for sig in SIGNALS:
            signal.signal(sig, signal.SIG_IGN)

    # Exclusive creation rejects existing files, including dangling symlinks.
    with args.record.open("x+") as output:
        def save():
            output.seek(0)
            json.dump(record, output, indent=2)
            output.write("\n")
            output.truncate()
            output.flush()
            os.fsync(output.fileno())

        previous = {sig: signal.signal(sig, stop) for sig in SIGNALS}
        try:
            ticks = identity(args.pid, args.start_ticks)
            initial = threads(args.pid)
            identity(args.pid, ticks)
            masks = {tuple(t["cpus"]) for t in initial}
            if len(masks) != 1 or args.cpu not in initial[0]["cpus"]:
                raise RuntimeError("initial threads need one uniform mask containing the requested CPU")
            original = initial[0]["cpus"]
            if original == [args.cpu]:
                raise RuntimeError("target is already pinned; ownership of this mask is ambiguous")
            record.update(start_ticks=ticks, cwd=str(WORKSPACE), original_cpus=original, initial_threads=initial)
            save()  # Durable original identities and masks before the first change.
            pin_started = True
            for thread in initial:
                if interrupted:
                    raise InterruptedError(f"signal {interrupted}")
                identity(args.pid, ticks)
                try:
                    current = sorted(os.sched_getaffinity(thread["tid"]))
                    if (start_ticks(Path(f"/proc/{args.pid}/task/{thread['tid']}/stat")) != thread["start_ticks"] or
                            current != original):
                        raise RuntimeError("thread identity or original affinity changed before pinning")
                    os.sched_setaffinity(thread["tid"], [args.cpu])
                    if start_ticks(Path(f"/proc/{args.pid}/task/{thread['tid']}/stat")) != thread["start_ticks"]:
                        raise RuntimeError("thread identity changed during pinning")
                except (FileNotFoundError, ProcessLookupError):
                    pass
            while True:
                if interrupted:
                    raise InterruptedError(f"signal {interrupted}")
                identity(args.pid, ticks)
                current = threads(args.pid)
                if any(t["cpus"] != [args.cpu] for t in current):
                    raise RuntimeError("target thread affinity escaped the requested CPU")
                record["checks"] += 1
                record["last_check_utc"] = datetime.now(timezone.utc).isoformat()
                record["last_threads"] = current
                if child is None:
                    child = subprocess.Popen(args.command, start_new_session=True)
                    record["child_pid"] = child.pid
                    save()
                if child.poll() is not None:
                    record["child_exit_code"] = child.returncode
                    break
                time.sleep(1)
        except Exception as error:
            record["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            for sig in SIGNALS:
                signal.signal(sig, signal.SIG_IGN)
            try:
                if child is not None:
                    # The controller needs time to stop its separately owned GPU
                    # worker, telemetry and post-gates before the final 3+3s helper.
                    if child.poll() is None:
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            pass
                    owned.stop_owned(child)
                    record["child_exit_code"] = child.returncode
                    record["owned_child_group_empty"] = True
            except Exception as error:
                record["errors"].append(f"child cleanup: {error}")
            try:
                if pin_started:
                    record["restoration"] = restore(args.pid, ticks, original, [args.cpu])
                    if (any(t["status"] in ("preserved_external_mask", "restore_error")
                            for t in record["restoration"]["actions"]) or
                            any(t["cpus"] != original for t in record["restoration"]["final_threads"])):
                        record["errors"].append("affinity restoration incomplete or external changes preserved")
            except Exception as error:
                record["errors"].append(f"restoration: {error}")
            record["signal"] = interrupted
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            record["complete"] = not record["errors"] and not interrupted and record.get("child_exit_code") == 0
            try:
                save()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--cpu", type=int, choices=[17], required=True)
    parser.add_argument("--start-ticks", type=int, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if args.pid <= 1 or not args.command:
        parser.error("positive target PID and nonempty --command are required")
    raise SystemExit(run(args))
