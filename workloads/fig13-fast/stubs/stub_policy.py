#!/usr/bin/env python3
import os
import signal
import sys
import time

role = sys.argv[1] if len(sys.argv) > 1 else "policy"
state = {"stopping": False}


def on_signal(_sig, _frame):
    state["stopping"] = True


signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)

print(f"stub policy started role={role} pid={os.getpid()}")
while not state["stopping"]:
    time.sleep(0.2)

if role == "sched":
    print()
    print("=== Statistics ===")
    print("task_init:      2")
    print("bind:           2")
    print("task_destroy:   2")
    print("timeslice_mod:  16")
    print("policy_hit:     12")
    print("policy_miss:    4")
else:
    print()
    print("=== Per-PID Statistics ===")
    print("  High priority PID stub:")
    print("    Total activated: 8")
    print()
    print("=== Summary ===")
    print("  Total current chunks: 4")
    print("  Total activated: 12")
    print("  Policy allow (moved): 7")
    print("  Policy deny (not moved): 5")

print(f"stub policy stopped role={role}")
sys.exit(0)
