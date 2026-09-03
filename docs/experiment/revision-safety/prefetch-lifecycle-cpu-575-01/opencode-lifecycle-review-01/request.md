You are an independent read-only systems reviewer. Do not call tools, edit
files, run commands, launch GPU work, or use the network. The complete relevant
files are attached directly.

Audit whether the attached lifecycle coordinator is ready for one live,
UVM-only RTX 5090 Q2 campaign. Focus on safety and evidence integrity:

1. Does it acquire the two existing shared leases exactly once without
   creating, replacing, or modifying their root-owned files, and close them
   before any authoritative successful record?
2. Are the candidate, exact restore module, fresh stage/output, BTF ABI, full
   parameter inventory, NVIDIA core identity, services, sessions, GPU idle
   state, holders, and kernel history all fail-closed at the correct points?
3. On every body or recording failure, can recovery still remove only the
   candidate, insert the exact old UVM with captured parameters, validate it
   before restarting the originally active services, and withhold services if
   the old runtime is not proven safe?
4. Can SIGINT/SIGTERM interrupt physical recovery or leave an authoritative
   `complete: true` lifecycle/summary before the lease is closed and the final
   completion point commits? Check the two pending-signal snapshots and all
   record-write failure paths.
5. Is any forbidden force unload, modprobe fallback, depmod, module install,
   reboot, broad process kill, or unrelated driver/service mutation reachable?
6. Do the CPU failure-injection tests substantively exercise these claims, and
   does the plan avoid claiming that the live campaign has already run?

Return `READY` or `REQUIRED FIXES`. List blockers first with exact function or
line references, then limitations. Do not infer any live measurements.
