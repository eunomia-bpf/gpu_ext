# Stage 4 Reduced-Capacity Prefetch Matrix

Status: `IMPLEMENTED_NOT_EXECUTED`

The runner covers `custom_no_policy`, `prefetch_none`, `prefetch_always_max`, and `prefetch_adaptive_sequential` at 0.95x, 1.05x, and 1.10x measured effective capacity. Each combination has three timing runs and one enhanced trace run; 1.10x adds one Nsight representative.

No Stage 4 policy timing, migration, fault, decision, eviction, or refault result exists yet. Stage 3 natural-capacity values remain historical input and are not mixed into this data set.

The matrix keeps the 300 second per-run limit. An 8 GiB `prefetch_none` timeout triggers at most one complete matrix restart at 6 GiB. A second timeout records `PREFETCH_NONE_UNBOUNDED_EVEN_AT_REDUCED_CAPACITY` and stops.
