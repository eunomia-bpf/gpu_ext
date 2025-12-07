#!/usr/bin/env python3
"""
Test chunk_trace with multiple uvmbench processes to verify PID tracking.
"""

import subprocess
import time
import os
import signal
import sys

CHUNK_TRACE = "/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src/chunk_trace"
UVMBENCH = "/home/yunwei37/workspace/gpu/co-processor-demo/memory/micro/uvmbench"
OUTPUT_FILE = "/tmp/chunk_trace_test.csv"

def main():
    # Kill any existing processes
    subprocess.run(["sudo", "pkill", "-9", "chunk_trace"], capture_output=True)
    subprocess.run(["pkill", "-9", "uvmbench"], capture_output=True)
    time.sleep(1)

    # Start chunk_trace
    print("Starting chunk_trace...")
    with open(OUTPUT_FILE, 'w') as f:
        trace_proc = subprocess.Popen(
            ["sudo", CHUNK_TRACE],
            stdout=f,
            stderr=subprocess.DEVNULL
        )
    time.sleep(2)

    # Start two uvmbench processes
    print("Starting two uvmbench processes...")
    uvm1 = subprocess.Popen(
        [UVMBENCH, "--size_factor=0.6", "--mode=uvm", "--iterations=1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    uvm2 = subprocess.Popen(
        [UVMBENCH, "--size_factor=0.6", "--mode=uvm", "--iterations=1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print(f"uvmbench user-space PIDs: {uvm1.pid} and {uvm2.pid}")

    # Wait for uvmbench to complete
    print("Waiting for uvmbench processes to complete...")
    uvm1.wait()
    uvm2.wait()
    print("uvmbench processes completed.")

    time.sleep(2)

    # Stop chunk_trace
    print("Stopping chunk_trace...")
    subprocess.run(["sudo", "kill", str(trace_proc.pid)], capture_output=True)
    trace_proc.wait()

    # Analyze results
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    with open(OUTPUT_FILE, 'r') as f:
        lines = f.readlines()

    if len(lines) <= 1:
        print("ERROR: No trace events captured!")
        return 1

    # Skip header
    header = lines[0].strip()
    print(f"Header: {header}")
    print(f"Total events: {len(lines) - 1}")

    # Parse and count PIDs (kernel pid), owner_pid, and va_space
    pid_counts = {}
    owner_pid_counts = {}
    va_space_counts = {}
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 5:
            pid = parts[2]
            owner_pid = parts[3] if parts[3] else "N/A"
            va_space = parts[4] if parts[4] else "N/A"
            pid_counts[pid] = pid_counts.get(pid, 0) + 1
            if owner_pid != "N/A" and owner_pid and owner_pid != "0":
                owner_pid_counts[owner_pid] = owner_pid_counts.get(owner_pid, 0) + 1
            if va_space != "N/A" and va_space and va_space != "0x0":
                va_space_counts[va_space] = va_space_counts.get(va_space, 0) + 1

    print(f"\n=== Kernel Thread PID Statistics ===")
    print(f"Unique kernel PIDs: {len(pid_counts)}")
    print(f"\nKernel PID breakdown:")
    for pid, count in sorted(pid_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  PID {pid}: {count} events")

    print(f"\n=== Owner PID Statistics (from va_block) ===")
    print(f"Unique owner PIDs: {len(owner_pid_counts)}")
    print(f"\nOwner PID breakdown:")
    for pid, count in sorted(owner_pid_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  Owner PID {pid}: {count} events")

    print(f"\n=== VA Space Statistics (unique per process) ===")
    print(f"Unique va_space pointers: {len(va_space_counts)}")
    print(f"\nVA Space breakdown:")
    for vs, count in sorted(va_space_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {vs}: {count} events")

    print(f"\n=== Expected vs Actual ===")
    print(f"Expected uvmbench PIDs: {uvm1.pid}, {uvm2.pid}")
    print(f"Captured owner PIDs: {list(owner_pid_counts.keys())}")

    # Check if we captured the right owner PIDs
    expected_pids = {str(uvm1.pid), str(uvm2.pid)}
    captured_owner_pids = set(owner_pid_counts.keys())

    if expected_pids == captured_owner_pids:
        print("\n✓ SUCCESS: Owner PIDs exactly match expected uvmbench PIDs!")
    elif expected_pids.issubset(captured_owner_pids):
        print("\n✓ SUCCESS: Owner PIDs include expected uvmbench PIDs (plus some others)")
    elif captured_owner_pids.intersection(expected_pids):
        print(f"\n⚠ PARTIAL: Only captured some expected owner PIDs")
        print(f"  Missing: {expected_pids - captured_owner_pids}")
    else:
        print(f"\n✗ MISMATCH: Owner PIDs don't match expected!")
        print(f"  Expected: {expected_pids}")
        print(f"  Got: {captured_owner_pids}")

    # Show sample events
    print(f"\n=== Sample Events (first 5) ===")
    for line in lines[1:6]:
        print(f"  {line.strip()}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
