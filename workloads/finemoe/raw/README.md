# Retained FineMoE raw records

The complete preflight-v2 all-positive `worker-result.json` is 283,958,690
bytes. GitHub rejects that single file, so its unchanged bytes are stored in
seven ordered parts beside it: `worker-result.json.part-00` through
`worker-result.json.part-06`. Parts 00–05 are each 41,943,040 bytes; part 06 is
32,300,450 bytes. This is file packaging only: no events, requests, numerical
arrays, or failed attempts are removed. The original whole file remains in the
experiment workspace and is ignored by Git.

After a fresh checkout, reconstruct it before running the existing offline
audit. From `raw/preflight-v2/block-00/all-positive`, first ensure that no
`worker-result.json` already exists, then concatenate the explicit ordered parts:

```sh
(set -C; cat worker-result.json.part-00 worker-result.json.part-01 \
  worker-result.json.part-02 worker-result.json.part-03 \
  worker-result.json.part-04 worker-result.json.part-05 \
  worker-result.json.part-06 > worker-result.json)
```

The experiment and analyzers still consume the original JSON schema. Archiving
uses a normal byte split, file sizes, and direct byte comparison, not content
digests or a new experiment gate.
