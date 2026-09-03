# In-progress raw-data checkpoint

This checkpoint retains the six completed isolated controls and the first
three complete four-arm blocks of the fixed ten-block campaign. It is **not a
completed experiment or a final performance result**. The original running
campaign continues with the unchanged protocol and no restart.

Only closed block-00, block-01 and block-02 cells are included in this commit;
the next block was still running. All forty mixed cells and an independent
post-run audit remain required before reporting the full comparison.

Each retained mixed cell contains 400 LC samples, 800 BE samples, complete
per-worker output validation, actual policy engagement and pre/post safety.
The retained original protocol specifies 50 kernels per stream, six processes,
four streams each, ten randomized four-arm blocks, six isolated controls,
repetitions 9511106, 340 blocks and 256 threads per kernel.
