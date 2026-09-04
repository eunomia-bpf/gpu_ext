# OpenCode/Qwen device-side subagent diagnosis

Date: 2026-09-04

## Outcome

OpenCode itself is working, but the configured
`spark-gateway/qwen3.8-27b-nvfp4-200k` inference path was not usable during
this check. Even a four-token request sent directly to the compatible chat
endpoint received no response in 120 seconds. The same gateway's Qwen Flash
model completed an OpenCode smoke test in 2.06 seconds and a read-only
device-source review in 11.85 seconds. Therefore the current failure boundary
is the 27B model backend or its queue, not repository discovery, attachments,
OpenCode startup, authentication, or the gateway front end.

This diagnosis did not change or restart the model service and did not expose
credentials.

## Observations

- OpenCode version `1.18.27` starts in about 0.28 seconds. The host runs Linux
  `6.15.11-061511-generic`.
- Both the OpenCode catalog and an authenticated `/v1/models` request contain
  `qwen3.8-27b-nvfp4-200k`; the latter returned HTTP 200 in 87 ms. Catalog
  membership establishes registration, not inference health.
- Gateway liveness and readiness returned HTTP 200 in 61--69 ms.
- In a fresh empty directory, `opencode run --pure --format json` with every
  tool denied and the prompt `Reply with exactly: OK` emitted no JSON event
  before the 120-second outer timeout. The local log shows that OpenCode had
  created a session and entered the 27B model stream, so this was not a CLI
  parsing or startup failure.
- A direct, authenticated chat-completions request to the 27B model used only
  57 prompt tokens and requested at most four completion tokens. It also
  produced no HTTP response before a 120.002-second client timeout. This rules
  out OpenCode's roughly 2K-token system prompt as the cause of the stall.
- Earlier OpenCode log records show the same model stream returning an empty
  `AI_APICallError` after about 125 seconds, followed by an automatic retry
  about 120 seconds later. This explains the earlier 300-second silent run:
  the outer timeout expired during the second model attempt, before OpenCode
  could emit a text result. The server-side reason for the missing first token
  cannot be distinguished as unavailable versus saturated without backend
  logs.
- As a control, `qwen3.8-flash-next-nvfp4-220k` returned a direct 64-token
  request in 0.88 seconds. Through OpenCode, it returned `OK` in 2.06 seconds
  with exit status zero and complete `step_start`, `text`, and `step_finish`
  events.

## Device-source audit control

The same deny-all OpenCode invocation attached only
`microbench/fig15-device/map_probe.bpf.c` and asked for a two-sentence SIMT
audit. Qwen Flash correctly identified that the map keys and update values
depend on the lane ID. Its additional explanation focused on proving the
range guard and is not used as verifier evidence: the real admission probe in
[`map-verifier-admission/results-20260904.md`](../workloads/llama.cpp/observability_overhead/revision-rq4/map-verifier-admission/results-20260904.md)
shows the stronger boundary. Only `noop` passes strict admission; all six map
programs are rejected for lane-varying branches and shared-map effects,
including varying keys, values, or lookup sinks.

The successful control session was
`ses_f926566c7ffecHn7LoVgtxBywf`. It was advisory and read-only; it did not run
the verifier or a GPU workload.

## Reusable non-interactive invocation

Use a primary agent, put the message immediately after `opencode run`, attach
files only after the message, keep the explicit model, and require structured
completion events. Do not pass `--auto`.

```bash
env CUDA_VISIBLE_DEVICES='' \
  OPENCODE_CONFIG_CONTENT='{"snapshot":false,"share":"disabled","autoupdate":false,"permission":{"*":"deny"},"agent":{"build":{"permission":{"*":"deny"}}},"tools":{"write":false,"edit":false,"bash":false,"webfetch":false,"task":false}}' \
  timeout --kill-after=5s 60s \
  opencode run "$prompt" \
    --pure \
    --format json \
    --model spark-gateway/qwen3.8-flash-next-nvfp4-220k \
    --agent build \
    --dir "$audit_dir" \
    --title "$unique_title" \
    --file "$attachment" > "$events"
```

A successful caller must require all three conditions:

1. `opencode` exits with status zero before the timeout;
2. the JSONL contains a nonempty `text` event; and
3. it ends with `step_finish` whose reason is `stop`.

For the requested 27B model, first run the same command with a 120-second
one-line smoke prompt and no attachment. Until that preflight succeeds, do not
launch parallel reviews or interpret timeout as a review verdict. Qwen Flash
is the currently validated fallback for quick advisory review; neither model
output replaces builds, tests, strict-verifier runs, or GPU measurements.
