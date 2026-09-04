#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
raw_dir="$script_dir/raw"
events="$raw_dir/opencode-events.jsonl"
review="$script_dir/opencode-review.md"
prompt="$script_dir/opencode-review-prompt.md"

if [[ $(opencode --version) != '1.18.27' ]]; then
    echo "OpenCode 1.18.27 is required" >&2
    exit 65
fi
if ! command -v jq >/dev/null; then
    echo "jq is required to retain the structured review" >&2
    exit 69
fi

attachments=(
    "$script_dir/RESULTS.md"
    "$script_dir/future-gpu-protocol.md"
    "$script_dir/audit_sources.sh"
    "$script_dir/run_cpu_lifecycle.sh"
    "$script_dir/run_ptrace_diagnostic.sh"
    "$raw_dir/run-environment.tsv"
    "$raw_dir/lifecycle.tsv"
    "$raw_dir/lifecycle-summary.tsv"
    "$raw_dir/ptrace-diagnostic.tsv"
    "$raw_dir/attach-ptrace-syscalls.txt"
    "$raw_dir/semantic-checks.tsv"
    "$raw_dir/open-module-delta.tsv"
    "$raw_dir/open-module-delta-summary.tsv"
    "$raw_dir/module-artifacts.tsv"
    "$raw_dir/module-symbols.txt"
    "$raw_dir/revisions.tsv"
)
for attachment in "${attachments[@]}"; do
    if [[ ! -f "$attachment" ]]; then
        echo "review attachment missing: $attachment" >&2
        exit 66
    fi
done

file_args=()
for attachment in "${attachments[@]}"; do
    file_args+=(--file "$attachment")
done

OPENCODE_CONFIG_CONTENT='{"snapshot":false,"share":"disabled","permission":{"*":"deny"},"tools":{"write":false,"edit":false,"bash":false,"webfetch":false,"task":false}}' \
timeout 300 opencode run \
    "$(<"$prompt")" \
    --model opencode/ling-3.0-flash-fin-free \
    --format json \
    --dir "$script_dir" \
    --title 'revision-deployment-575-read-only-review' \
    "${file_args[@]}" > "$events"

session_id=$(jq -r 'select(.sessionID != null) | .sessionID' "$events" | sed -n '1p')
if [[ -z "$session_id" ]]; then
    echo "OpenCode output did not contain a session ID" >&2
    exit 1
fi
response=$(jq -r 'select(.type == "text" and .part.text != null) | .part.text' \
    "$events")
if [[ -z "$response" ]]; then
    echo "OpenCode output did not contain a text response" >&2
    exit 1
fi
verdict=$(printf '%s\n' "$response" | \
    awk '/^VERDICT: (PASS|FAIL)$/{value=$0} END {print value}')
if [[ -z "$verdict" ]]; then
    echo "OpenCode response did not end with the required verdict" >&2
    exit 1
fi

{
    printf '# OpenCode read-only evidence review\n\n'
    printf -- '- OpenCode version: 1.18.27\n'
    printf -- '- Model: opencode/ling-3.0-flash-fin-free\n'
    printf -- '- Session: %s\n' "$session_id"
    printf -- '- Tool policy: all permissions denied; write, edit, bash, webfetch, and task disabled\n\n'
    printf '%s\n' "$response"
} > "$review"

printf 'session_id\tverdict\n%s\t%s\n' "$session_id" "${verdict#VERDICT: }" \
    > "$raw_dir/opencode-verdict.tsv"
echo "$verdict"
