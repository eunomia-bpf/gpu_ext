#!/usr/bin/env python3
import os
import sys
import time

sleep_s = float(os.environ.get("FIG13_FAST_STUB_SLEEP", "1.5"))
label = os.path.basename(sys.argv[0]) if len(sys.argv) > 1 else "stub"
t0 = time.perf_counter()
time.sleep(sleep_s)
elapsed_ms = (time.perf_counter() - t0) * 1000.0

print("UVM Microbenchmark (fig13-fast CPU stub)")
print("========================================")
print(f"Stub tenant: {label}")
print(f"Sleep: {sleep_s}")
print()
print("Results:")
print(f"  Median time: {elapsed_ms:.3f} ms")
print(f"  Min time: {elapsed_ms:.3f} ms")
print(f"  Max time: {elapsed_ms:.3f} ms")
print(f"  Bandwidth: 1.000 GB/s")
sys.exit(0)
