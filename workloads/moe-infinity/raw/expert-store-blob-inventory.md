# Expert-store partition blobs left outside Git for these campaigns

The `archer_param_*` files below are MoE-Infinity CPU expert offload stores. The runner
rebuilds them from the weights under `deps/hf-cache` (`run_moe_head_to_head.py` sets
`offload_dir = attempt_dir / 'moe-offload'`), so they are inputs rather than measurements.
Every measurement record of these campaigns is committed next to this inventory: all
`*.json`, `*.jsonl`, `*.log`, `*.tsv` and `strace/open.trace.*` files, plus the store
`archer_index` and `name_id_map.json` metadata.

| File | Size (bytes) |
| --- | --- |
| `correctness-preflight-610-20260831-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_0` | 10733903872 |
| `correctness-preflight-610-20260831-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_1` | 10732941312 |
| `correctness-preflight-610-20260831-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_2` | 10731290624 |
| `correctness-preflight-610-20260831-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_3` | 10730954752 |
| `correctness-preflight-610-20260831-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_4` | 10711715840 |
| `correctness-preflight-610-20260831-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_5` | 10732302336 |
| `correctness-preflight-610-20260831-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_6` | 922480640 |
| `head-to-head-575/preflight/expert-store/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_0` | 10733903872 |
| `head-to-head-575/preflight/expert-store/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_1` | 10732941312 |
| `head-to-head-575/preflight/expert-store/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_2` | 10731290624 |
| `head-to-head-575/preflight/expert-store/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_3` | 10730954752 |
| `head-to-head-575/preflight/expert-store/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_4` | 10711715840 |
| `head-to-head-575/preflight/expert-store/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_5` | 10732302336 |
| `head-to-head-575/preflight/expert-store/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_6` | 922480640 |
| `repaired-preflight/attempt-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_0` | 10733903872 |
| `repaired-preflight/attempt-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_1` | 10732941312 |
| `repaired-preflight/attempt-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_2` | 10731290624 |
| `repaired-preflight/attempt-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_3` | 10730954752 |
| `repaired-preflight/attempt-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_4` | 10711715840 |
| `repaired-preflight/attempt-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_5` | 10732302336 |
| `repaired-preflight/attempt-01/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_6` | 922480640 |
| `repaired-preflight/attempt-02/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_0` | 10733903872 |
| `repaired-preflight/attempt-02/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_1` | 10732941312 |
| `repaired-preflight/attempt-02/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_2` | 10731290624 |
| `repaired-preflight/attempt-02/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_3` | 10730954752 |
| `repaired-preflight/attempt-02/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_4` | 10711715840 |
| `repaired-preflight/attempt-02/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_5` | 10732302336 |
| `repaired-preflight/attempt-02/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_6` | 922480640 |
| `repaired-preflight/attempt-03/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_0` | 10733903872 |
| `repaired-preflight/attempt-03/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_1` | 10732941312 |
| `repaired-preflight/attempt-03/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_2` | 10731290624 |
| `repaired-preflight/attempt-03/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_3` | 10730954752 |
| `repaired-preflight/attempt-03/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_4` | 10711715840 |
| `repaired-preflight/attempt-03/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_5` | 10732302336 |
| `repaired-preflight/attempt-03/moe_infinity_075/moe-offload/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/archer_param_6` | 922480640 |

Total: 326477946880 bytes across 35 files.
