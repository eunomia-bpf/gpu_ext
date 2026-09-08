# Larger-grid automatic warp and hook experiment

User explicitly requested root to run automatic warp/hook now. Reuses the
existing workload, original shared_update BPF object, strict runner and
current built runtime. No driver changes or paper changes.

Timing: 128 threads/CTA; 16/32/64 CTAs at work=0, followed by 64 CTAs
at work=8/32/128. Each setting has ten rotating native/off/on groups,
eight warmups and 128 timed launches (180 measurements). Calls are counted
separately using the same settings and two launches without warmup; those
diagnostic timings are not performance results. Earlier 2/4/8-CTA and
one-CTA work sweeps remain unchanged. CUDA event times cover all 128 launches.

Run: python3 -u run.py > runner.log 2>&1
The imported runner acquires both shared experiment locks for each sweep.
