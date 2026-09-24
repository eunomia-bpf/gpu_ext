# Historical experiment records — 2026-09-24

This archive preserves 4,149 previously ignored text records, totaling
15,592,507,688 bytes before packaging. It contains experiment logs, structured
outputs, analysis files and retained source inputs from earlier runs, including
failed and incomplete attempts. Packaging these records adds no new measurement
or claim that an experiment succeeded.

The compressed archive is 498,713,856 bytes, split into 12 ordered parts.
Parts 000–010 are 41,943,040 bytes each; part 011 is 37,340,416 bytes.
[inventory.json](inventory.json) lists every original repository-relative file
path and byte count. [parts.json](parts.json) gives the ordered part list and
sizes. XZ content checks are disabled; verification used direct comparison of
every decompressed file with its original and every part with the corresponding
archive bytes. All 4,149 file comparisons and the part comparisons passed.

## Restore into an empty directory

Run from this directory. Choose a new destination so existing experiment
records are preserved:

```sh
mkdir /tmp/gpu-records-20260924
cat records.tar.xz.part-??? | tar -xJf - --keep-old-files -C /tmp/gpu-records-20260924
```

Restored paths are relative to the gpu_ext repository root, for example
`/tmp/gpu-records-20260924/workloads/stale-state-575/raw/...`. Use the inventory
to locate a particular cohort. The original files remain untouched in the
working experiment environment and retain their existing ignore rules.

## Scope

The inventory covers previously ignored text records under workload,
microbenchmark and scheduling-result directories. Already tracked files remain
in their existing locations. It excludes copied tool-build trees, generated BPF
headers, caches, downloaded dependencies, models and executable build products.
The already-published FineMoE `worker-result.json` is recoverable from its
[existing seven parts](../../../workloads/finemoe/raw/README.md), so it is not
duplicated here. Manuscript files are excluded.
