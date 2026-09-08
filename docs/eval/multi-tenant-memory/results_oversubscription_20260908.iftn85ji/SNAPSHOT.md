# Running experiment snapshot

Snapshot inventory taken at 2026-09-08T14:35:48.688406+00:00.

The authorized matrix is 300 cells, ending at ratio 1.5. The experiment
continues while this checkpoint is committed. Partially written/current-cell
logs are retained as progress, not completed measurements. The analysis
preview is an earlier snapshot and includes only complete workload/ratio points.
Re-run analyze.py to refresh it after the experiment finishes.

| Workload | Ratio | Completed cells |
|---|---|---:|
| gemm | ratio-0.8 | 25/25 |
| gemm | ratio-1.0 | 25/25 |
| gemm | ratio-1.2 | 25/25 |
| gemm | ratio-1.5 | 25/25 |
| hotspot | ratio-0.8 | 25/25 |
| hotspot | ratio-1.0 | 25/25 |
| hotspot | ratio-1.2 | 25/25 |
| hotspot | ratio-1.5 | 25/25 |
| kmeans_sparse | ratio-0.8 | 25/25 |
| kmeans_sparse | ratio-1.0 | 25/25 |
| kmeans_sparse | ratio-1.2 | 25/25 |
| kmeans_sparse | ratio-1.5 | 3/25 |

Total observed at inventory: 278/300.

The local-model driver build at 05:29:40–05:34:17 PDT overlapped the
live sweep. Retain those observations and disclose the overlap when choosing
final measurements. This snapshot does not establish an uncontended run
for every point. The coordinated earlier runtime build window is documented
in the analysis README.

The live runner will encounter the user-cancelled ratio 1.8 directory guard
after finishing ratio 1.5. That expected scope-stop exit is documented in
`hotspot/ratio-1.8/USER_CANCELLED.md`; driver restoration must still succeed.

The scheduler comparison script changes are a source checkpoint with Python
syntax checked. No additional scheduler benchmark was run for this commit.
