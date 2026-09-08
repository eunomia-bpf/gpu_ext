User cancelled ratio 1.8 on 2026-09-08 while HotSpot ratio 1.5 was running.
This directory activates the live runner output-exists guard before any ratio 1.8 measurement.
After all three ratio 1.5 workloads complete, the live runner reports Refusing to repeat existing measurements here and exits 1: an authorized scope stop, not an experimental failure.
Driver restoration must still report RESTORATION_OK=1.
Final matrix: 3 workloads x 4 ratios (0.8, 1.0, 1.2, 1.5) x 5 policies x 5 repetitions = 300 cells.
Do not retry ratio 1.8 or remove this guard while the live runner is active.
