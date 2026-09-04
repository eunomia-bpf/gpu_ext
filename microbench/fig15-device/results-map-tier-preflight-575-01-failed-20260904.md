# Device-map preflight attempt 01: retained RPC payload race

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/map-tier-preflight-575-01`

This attempt is **invalid and contributes no performance result**. The first
`host_update` cell passed. The following `rpc_lookup` cell executed the target
and detached cleanly, but all 32 output entries contained the initialized
key-0 value instead of their distinct per-lane values, so the complete map
oracle rejected the run and no later arm executed.

Source inspection found a multi-lane correctness race in the legacy GPU-to-host
helper bridge. Each lane copied its key into one shared request buffer before
acquiring the bridge lock; the host therefore serviced the serialized calls
using the last shared payload. The same ordering affected RPC update and
delete. The repair moves each active lane's key/value/flags publication inside
the lock, before the request signal, without changing the device-resident or
direct host-mapped fast paths. The retained attempt is not resumed or relabelled.

