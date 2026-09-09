# Off-only hook scalability

User requested larger block counts and explicitly selected Off, not On.
Native is the uninstrumented timing baseline. Off retains per-thread BPF
callback execution with automatic warp disabled. Same shared_update object,
same runtime, no driver or paper changes. Only the benchmark input cap
changes from 64 to 8192 CTAs in a separate source/build copy.

Plan: 128/512/2048/8192 CTAs, 128 threads/CTA, arithmetic iterations
0/128/1024, ten alternating Native/Off pairs each (240 measurements).
Each cell uses eight warmups and 128 timed launches. Full output correctness
and map checks remain in the reused runner. No On cells. CUDA event times
cover all timed launches, not host verification or setup.
