#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


BASE: Path
PAUSE_GAP = timedelta(minutes=30)
FOCUS_PREFIXES = [
    "6b21980a",
    "1ffa360b",
    "e19dd100",
    "b1e7bc20",
    "9d654c47",
    "6c0aa1fb",
    "de6eabd4",
    "648091fb",
    "d662e303",
    "16156e61",
]
COUNTED_TOOLS = [
    "Read",
    "Edit",
    "Write",
    "Bash",
    "Grep",
    "Glob",
    "Agent",
    "TaskCreate",
    "TaskUpdate",
]
BUILD_RE = re.compile(r"\b(make|clang)\b", re.I)
MODULE_RE = re.compile(r"\b(sudo|insmod|rmmod)\b", re.I)
BENCH_RE = re.compile(r"run_exp|llama-bench|python.*bench|bench", re.I)
ERROR_RE = re.compile(r"error", re.I)
XID_RE = re.compile(r"xid", re.I)
VERIFIER_RE = re.compile(r"verifier", re.I)
FAIL_RE = re.compile(r"fail", re.I)

CASE_SPECS = {
    "A": {
        "name": "FAISS phase-adaptive",
        "session_prefix": "1ffa360b",
        "marker_strings": [
            "Read/Edit/Write on extension/prefetch_faiss_phase*.c",
            "Bash command make prefetch_faiss_phase",
            "Agent prompt matching Test FAISS phase*",
        ],
        "use_regex_fallback": False,
        "structured": {
            "file_tools": ["Edit", "Write"],
            "file_suffixes": [
                "extension/prefetch_faiss_phase.bpf.c",
                "extension/prefetch_faiss_phase.c",
            ],
            "bash_contains": [
                "make prefetch_faiss_phase",
            ],
            "agent_patterns": [
                re.compile(r"test faiss phase", re.I),
            ],
        },
        "patterns": [
            re.compile(r"extension/prefetch_faiss_phase\.bpf\.c", re.I),
            re.compile(r"make prefetch_faiss_phase", re.I),
            re.compile(r"config_[a-z0-9_]*faiss_phase", re.I),
            re.compile(r"test faiss phase", re.I),
        ],
    },
    "B": {
        "name": "GPU preemption kfunc",
        "session_prefix": "b1e7bc20",
        "marker_strings": [
            "preempt_kfunc",
            "test_preempt_kfunc",
            "bench_preempt_kfunc",
            "gpu_preempt_kfunc_plan",
            "bpf_nv_gpu_preempt_tsg",
        ],
        "patterns": [
            re.compile(r"preempt_kfunc", re.I),
            re.compile(r"test_preempt_kfunc", re.I),
            re.compile(r"bench_preempt_kfunc", re.I),
            re.compile(r"gpu_preempt_kfunc_plan", re.I),
            re.compile(r"bpf_nv_gpu_preempt_tsg", re.I),
        ],
    },
    "C": {
        "name": "Cross-block multi-stride",
        "session_prefix": "6b21980a",
        "marker_strings": [
            "cross_block_prefetch_plan",
            "prefetch_cross_block_v2",
            "adjacent-stride",
            "multi-stride",
            "prefetch_adjacent_stride",
        ],
        "patterns": [
            re.compile(r"cross_block_prefetch_plan", re.I),
            re.compile(r"prefetch_cross_block_v2", re.I),
            re.compile(r"adjacent[-_ ]stride", re.I),
            re.compile(r"multi[-_ ]stride", re.I),
            re.compile(r"prefetch_adjacent_stride", re.I),
        ],
    },
}


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iter_record_timestamps(obj: dict) -> list[datetime]:
    out: list[datetime] = []
    for value in (obj.get("timestamp"), obj.get("snapshot", {}).get("timestamp")):
        dt = parse_ts(value)
        if dt is not None:
            out.append(dt)
    return out


def event_timestamp(obj: dict) -> datetime | None:
    for value in (obj.get("timestamp"), obj.get("snapshot", {}).get("timestamp")):
        dt = parse_ts(value)
        if dt is not None:
            return dt
    return None


def flatten_strings(value, out: list[str], skip_large_keys: bool = False) -> None:
    if isinstance(value, str):
        out.append(value)
        return
    if isinstance(value, list):
        for item in value:
            flatten_strings(item, out, skip_large_keys=skip_large_keys)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if skip_large_keys and key in {"originalFile", "oldString", "newString", "diff"}:
                continue
            flatten_strings(item, out, skip_large_keys=skip_large_keys)


def collect_record_text(obj: dict) -> str:
    chunks: list[str] = []
    msg = obj.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "thinking" and isinstance(block.get("thinking"), str):
                    chunks.append(block["thinking"])
                elif block_type == "text" and isinstance(block.get("text"), str):
                    chunks.append(block["text"])
                elif block_type == "tool_use":
                    if isinstance(block.get("name"), str):
                        chunks.append(block["name"])
                    flatten_strings(block.get("input", {}), chunks, skip_large_keys=True)
                elif block_type == "tool_result":
                    flatten_strings(block.get("content"), chunks, skip_large_keys=True)
    if isinstance(obj.get("toolUseResult"), dict):
        result = obj["toolUseResult"]
        for key in ("stdout", "stderr", "filePath", "query", "matches"):
            flatten_strings(result.get(key), chunks, skip_large_keys=True)
    return "\n".join(chunks)


def active_windows(timestamps: Iterable[datetime]) -> list[tuple[datetime, datetime]]:
    ordered = sorted(timestamps)
    if not ordered:
        return []
    windows: list[tuple[datetime, datetime]] = []
    start = ordered[0]
    prev = ordered[0]
    for current in ordered[1:]:
        if current - prev > PAUSE_GAP:
            windows.append((start, prev))
            start = current
        prev = current
    windows.append((start, prev))
    return windows


def duration_str(delta: timedelta) -> str:
    minutes = int(round(delta.total_seconds() / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def money_str(amount: float) -> str:
    return f"${amount:,.2f}"


def token_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000.0) * 15.0 + (output_tokens / 1_000_000.0) * 75.0


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_window(window: tuple[datetime, datetime]) -> str:
    start, end = window
    return f"{fmt_dt(start)} -> {fmt_dt(end)} ({duration_str(end - start)})"


def safe_ratio(a: float, b: float) -> float:
    return a / b if b else math.inf


@dataclass
class TranscriptMetrics:
    path: Path
    kind: str
    session_id: str
    line_count: int = 0
    malformed_lines: int = 0
    assistant_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_output_proxy: int = 0
    action_output_proxy: int = 0
    response_output_proxy: int = 0
    total_edits: int = 0
    bpf_edit_count: int = 0
    tool_counts: Counter = field(default_factory=Counter)
    total_tool_invocations: int = 0
    bash_invocations: int = 0
    build_count: int = 0
    module_ops_count: int = 0
    benchmark_count: int = 0
    agent_spawns: int = 0
    bash_output_lines: int = 0
    error_lines: int = 0
    xid_lines: int = 0
    verifier_lines: int = 0
    fail_lines: int = 0
    timestamps: list[datetime] = field(default_factory=list)
    tool_events: list[tuple[datetime, str]] = field(default_factory=list)

    @property
    def short_id(self) -> str:
        return self.session_id.split("-")[0]

    @property
    def first_ts(self) -> datetime | None:
        return min(self.timestamps) if self.timestamps else None

    @property
    def last_ts(self) -> datetime | None:
        return max(self.timestamps) if self.timestamps else None

    @property
    def duration(self) -> timedelta:
        if not self.timestamps:
            return timedelta(0)
        return self.last_ts - self.first_ts

    @property
    def active_windows(self) -> list[tuple[datetime, datetime]]:
        return active_windows(self.timestamps)

    @property
    def active_duration(self) -> timedelta:
        total = timedelta(0)
        for start, end in self.active_windows:
            total += end - start
        return total


def extract_file_path(tool_input: dict) -> str | None:
    for key in ("file_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return None


def bash_lines_from_result(obj: dict, block: dict) -> list[str]:
    result = obj.get("toolUseResult")
    chunks: list[str] = []
    if isinstance(result, dict):
        for key in ("stdout", "stderr"):
            value = result.get(key)
            if isinstance(value, str) and value:
                chunks.append(value)
    if not chunks:
        content = block.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            strings: list[str] = []
            flatten_strings(content, strings, skip_large_keys=True)
            if strings:
                chunks.append("\n".join(strings))
    lines: list[str] = []
    for chunk in chunks:
        lines.extend(chunk.splitlines())
    return lines


def parse_transcript(path: Path, kind: str) -> TranscriptMetrics:
    metrics = TranscriptMetrics(path=path, kind=kind, session_id=path.stem if kind == "primary" else "")
    tool_by_id: dict[str, str] = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            metrics.line_count += 1
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                metrics.malformed_lines += 1
                continue
            if not metrics.session_id:
                metrics.session_id = obj.get("sessionId", "")
            metrics.timestamps.extend(iter_record_timestamps(obj))
            ts = event_timestamp(obj)
            if obj.get("type") == "assistant":
                metrics.assistant_turns += 1
                msg = obj.get("message", {})
                usage = msg.get("usage", {})
                metrics.input_tokens += int(usage.get("input_tokens") or 0)
                output_tokens = int(usage.get("output_tokens") or 0)
                metrics.output_tokens += output_tokens
                content = msg.get("content")
                has_thinking = False
                has_tool_use = False
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "thinking":
                            has_thinking = True
                        if block.get("type") == "tool_use":
                            has_tool_use = True
                            name = block.get("name", "")
                            metrics.tool_counts[name] += 1
                            metrics.total_tool_invocations += 1
                            if ts is not None:
                                metrics.tool_events.append((ts, name))
                            tool_id = block.get("id")
                            if isinstance(tool_id, str):
                                tool_by_id[tool_id] = name
                            tool_input = block.get("input", {})
                            if name in {"Edit", "Write"}:
                                metrics.total_edits += 1
                                file_path = extract_file_path(tool_input if isinstance(tool_input, dict) else {})
                                if isinstance(file_path, str) and file_path.endswith(".bpf.c"):
                                    metrics.bpf_edit_count += 1
                            if name == "Bash":
                                metrics.bash_invocations += 1
                                command = ""
                                if isinstance(tool_input, dict):
                                    command = tool_input.get("command", "") or ""
                                if BUILD_RE.search(command):
                                    metrics.build_count += 1
                                if MODULE_RE.search(command):
                                    metrics.module_ops_count += 1
                                if BENCH_RE.search(command):
                                    metrics.benchmark_count += 1
                            if name == "Agent":
                                metrics.agent_spawns += 1
                if has_tool_use:
                    metrics.action_output_proxy += output_tokens
                elif has_thinking:
                    metrics.thinking_output_proxy += output_tokens
                else:
                    metrics.response_output_proxy += output_tokens
            elif obj.get("type") == "user":
                msg = obj.get("message", {})
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_result":
                            continue
                        tool_use_id = block.get("tool_use_id")
                        if tool_by_id.get(tool_use_id) != "Bash":
                            continue
                        lines = bash_lines_from_result(obj, block)
                        metrics.bash_output_lines += len(lines)
                        for result_line in lines:
                            if ERROR_RE.search(result_line):
                                metrics.error_lines += 1
                            if XID_RE.search(result_line):
                                metrics.xid_lines += 1
                            if VERIFIER_RE.search(result_line):
                                metrics.verifier_lines += 1
                            if FAIL_RE.search(result_line):
                                metrics.fail_lines += 1
    if not metrics.session_id and kind == "primary":
        metrics.session_id = path.stem
    return metrics


def aggregate_metrics(items: Iterable[TranscriptMetrics], label: str) -> dict:
    records = list(items)
    return {
        "label": label,
        "transcripts": len(records),
        "input_tokens": sum(item.input_tokens for item in records),
        "output_tokens": sum(item.output_tokens for item in records),
        "assistant_turns": sum(item.assistant_turns for item in records),
        "tool_invocations": sum(item.total_tool_invocations for item in records),
        "builds": sum(item.build_count for item in records),
        "benchmarks": sum(item.benchmark_count for item in records),
        "bpf_edits": sum(item.bpf_edit_count for item in records),
        "agent_spawns": sum(item.agent_spawns for item in records),
        "wall_clock": sum((item.duration for item in records), timedelta(0)),
        "active_time": sum((item.active_duration for item in records), timedelta(0)),
        "thinking_output_proxy": sum(item.thinking_output_proxy for item in records),
        "action_output_proxy": sum(item.action_output_proxy for item in records),
        "response_output_proxy": sum(item.response_output_proxy for item in records),
    }


def structured_case_match(obj: dict, spec: dict) -> bool:
    structured = spec.get("structured")
    if not structured:
        return False
    msg = obj.get("message", {})
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        tool_input = block.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        file_tools = set(structured.get("file_tools", ["Read", "Edit", "Write"]))
        if name in file_tools:
            file_path = extract_file_path(tool_input)
            if isinstance(file_path, str):
                if any(file_path.endswith(suffix) for suffix in structured.get("file_suffixes", [])):
                    return True
        if name == "Bash":
            command = tool_input.get("command", "") or ""
            if any(token.lower() in command.lower() for token in structured.get("bash_contains", [])):
                return True
        if name == "Agent":
            strings: list[str] = []
            flatten_strings(tool_input, strings, skip_large_keys=True)
            text = "\n".join(strings)
            if any(pattern.search(text) for pattern in structured.get("agent_patterns", [])):
                return True
    return False


def find_active_windows_for_case(session_path: Path, windows: list[tuple[datetime, datetime]], spec: dict) -> list[int]:
    matched: set[int] = set()
    with session_path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            ts = event_timestamp(obj)
            if ts is None:
                continue
            matched_here = structured_case_match(obj, spec)
            if not matched_here and spec.get("use_regex_fallback", True):
                text = collect_record_text(obj)
                matched_here = any(pattern.search(text) for pattern in spec["patterns"])
            if not matched_here:
                continue
            for idx, (start, end) in enumerate(windows):
                if start <= ts <= end:
                    matched.add(idx)
                    break
    return sorted(matched)


def metrics_for_windows(paths: list[Path], windows: list[tuple[datetime, datetime]]) -> dict:
    tool_by_id: dict[str, str] = {}
    out = {
        "input_tokens": 0,
        "output_tokens": 0,
        "assistant_turns": 0,
        "tool_invocations": 0,
        "total_edits": 0,
        "bpf_edits": 0,
        "builds": 0,
        "benchmarks": 0,
        "module_ops": 0,
        "agent_spawns": 0,
        "tool_counts": Counter(),
        "timestamps": [],
        "transcript_paths": set(),
    }
    if not windows:
        return out

    def inside(ts: datetime) -> bool:
        return any(start <= ts <= end for start, end in windows)

    for path in paths:
        with path.open(errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                ts = event_timestamp(obj)
                if ts is None or not inside(ts):
                    continue
                out["timestamps"].extend(iter_record_timestamps(obj))
                out["transcript_paths"].add(path)
                if obj.get("type") == "assistant":
                    out["assistant_turns"] += 1
                    msg = obj.get("message", {})
                    usage = msg.get("usage", {})
                    out["input_tokens"] += int(usage.get("input_tokens") or 0)
                    out["output_tokens"] += int(usage.get("output_tokens") or 0)
                    content = msg.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict) or block.get("type") != "tool_use":
                                continue
                            name = block.get("name", "")
                            out["tool_counts"][name] += 1
                            out["tool_invocations"] += 1
                            tool_id = block.get("id")
                            if isinstance(tool_id, str):
                                tool_by_id[tool_id] = name
                            tool_input = block.get("input", {})
                            if name in {"Edit", "Write"}:
                                out["total_edits"] += 1
                                file_path = extract_file_path(tool_input if isinstance(tool_input, dict) else {})
                                if isinstance(file_path, str) and file_path.endswith(".bpf.c"):
                                    out["bpf_edits"] += 1
                            if name == "Bash":
                                command = ""
                                if isinstance(tool_input, dict):
                                    command = tool_input.get("command", "") or ""
                                if BUILD_RE.search(command):
                                    out["builds"] += 1
                                if MODULE_RE.search(command):
                                    out["module_ops"] += 1
                                if BENCH_RE.search(command):
                                    out["benchmarks"] += 1
                            if name == "Agent":
                                out["agent_spawns"] += 1
    out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


def count_nested_jsonl_metrics(session_id: str) -> tuple[int, int, list[Path]]:
    session_dir = BASE / session_id
    if not session_dir.is_dir():
        return 0, 0, []
    paths = sorted(session_dir.rglob("*.jsonl"))
    total_lines = 0
    for path in paths:
        with path.open(errors="replace") as handle:
            total_lines += sum(1 for _ in handle)
    return len(paths), total_lines, paths


def prefix_lookup(primary_by_prefix: dict[str, TranscriptMetrics], prefix: str) -> TranscriptMetrics:
    if prefix not in primary_by_prefix:
        raise KeyError(f"missing session prefix {prefix}")
    return primary_by_prefix[prefix]


def peak_tool_density(session: TranscriptMetrics) -> tuple[int, datetime | None, datetime | None, Counter]:
    events = sorted(session.tool_events, key=lambda item: item[0])
    if not events:
        return 0, None, None, Counter()
    best_count = 0
    best_start = None
    best_end = None
    best_counter = Counter()
    left = 0
    for right, (right_ts, _) in enumerate(events):
        while right_ts - events[left][0] > timedelta(hours=1):
            left += 1
        count = right - left + 1
        if count > best_count:
            best_count = count
            best_start = events[left][0]
            best_end = right_ts
            best_counter = Counter(name for _, name in events[left : right + 1])
    return best_count, best_start, best_end, best_counter


def render_markdown(
    primary_metrics: list[TranscriptMetrics],
    nested_metrics: list[TranscriptMetrics],
    focus_metrics: list[TranscriptMetrics],
    nested_counts: dict[str, tuple[int, int]],
    case_data: dict[str, dict],
) -> str:
    primary_by_prefix = {item.short_id: item for item in primary_metrics}
    primary_agg = aggregate_metrics(primary_metrics, "Primary sessions only")
    all_agg = aggregate_metrics(primary_metrics + nested_metrics, "All transcripts (primary + nested)")

    longest_window = None
    longest_session = None
    for session in primary_metrics:
        for window in session.active_windows:
            if longest_window is None or (window[1] - window[0]) > (longest_window[1] - longest_window[0]):
                longest_window = window
                longest_session = session

    most_edits_session = max(primary_metrics, key=lambda item: item.total_edits)
    raw_error_session = max(primary_metrics, key=lambda item: item.error_lines)
    rate_candidates = [item for item in primary_metrics if item.bash_output_lines >= 100]
    error_rate_session = max(
        rate_candidates,
        key=lambda item: safe_ratio(item.error_lines, item.bash_output_lines),
    ) if rate_candidates else None
    peak_session = None
    peak_density = (0, None, None, Counter())
    for session in primary_metrics:
        density = peak_tool_density(session)
        if density[0] > peak_density[0]:
            peak_density = density
            peak_session = session

    total_direct_tools_primary = primary_agg["tool_invocations"] - primary_agg["agent_spawns"]
    tokens_per_iteration = safe_ratio(
        all_agg["input_tokens"] + all_agg["output_tokens"], all_agg["benchmarks"]
    )
    thinking_action_ratio = safe_ratio(
        all_agg["thinking_output_proxy"], all_agg["action_output_proxy"]
    )

    lines: list[str] = []
    lines.append("# Q6 Precise Claude Transcript Metrics for `gpu_ext`")
    lines.append("")
    lines.append(f"Generated: {fmt_dt(datetime.now(timezone.utc))}")
    lines.append("")
    lines.append("## Scope and Method")
    lines.append("")
    lines.append(f"- Corpus root: `{BASE}`")
    lines.append(f"- Primary sessions: {len(primary_metrics)} top-level `*.jsonl` transcripts")
    lines.append(
        f"- Nested transcripts: {len(nested_metrics)} `subagents/*.jsonl` files under per-session directories"
    )
    lines.append(
        "- Per-session tables below use the primary transcript only, matching the main session file. Nested subagent work is counted separately in the `Subagent files` and `Subagent lines` columns."
    )
    lines.append(
        "- Aggregate totals are reported in two scopes: `Primary sessions only` and `All transcripts (primary + nested)`."
    )
    lines.append(
        "- Time metrics use every timestamp found in each JSONL record (`timestamp` and `snapshot.timestamp` when present). Active windows split on gaps greater than 30 minutes."
    )
    lines.append(
        "- Token metrics sum `message.usage.input_tokens` and `message.usage.output_tokens` from assistant records only. Cached-token fields are intentionally excluded."
    )
    lines.append(
        "- Bash failure metrics count output lines from Bash `tool_result` records only; assistant reasoning text is excluded."
    )
    lines.append(
        "- Case-study windows are defined as primary-session active windows that contain case-specific markers, then measured across all transcripts under that parent session during those windows."
    )
    lines.append("")
    lines.append("### Malformed Lines")
    lines.append("")
    malformed = [
        (item.short_id, item.malformed_lines)
        for item in primary_metrics
        if item.malformed_lines
    ]
    malformed.extend(
        (f"{item.short_id}:{item.path.name}", item.malformed_lines)
        for item in nested_metrics
        if item.malformed_lines
    )
    if malformed:
        for key, count in malformed:
            lines.append(f"- `{key}`: {count} malformed line(s), all NUL-only and skipped")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Scope | Transcripts | Input tokens | Output tokens | Total tokens | Cost | Summed wall-clock | Summed active span | Tool calls | Builds | Benchmarks | `.bpf.c` edits | Agent spawns |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for agg in (primary_agg, all_agg):
        total_tokens = agg["input_tokens"] + agg["output_tokens"]
        lines.append(
            "| {label} | {transcripts:,} | {input_tokens:,} | {output_tokens:,} | {total_tokens:,} | {cost} | {wall} | {active} | {tools:,} | {builds:,} | {benchmarks:,} | {bpf_edits:,} | {agent_spawns:,} |".format(
                label=agg["label"],
                transcripts=agg["transcripts"],
                input_tokens=agg["input_tokens"],
                output_tokens=agg["output_tokens"],
                total_tokens=total_tokens,
                cost=money_str(token_cost(agg["input_tokens"], agg["output_tokens"])),
                wall=duration_str(agg["wall_clock"]),
                active=duration_str(agg["active_time"]),
                tools=agg["tool_invocations"],
                builds=agg["builds"],
                benchmarks=agg["benchmarks"],
                bpf_edits=agg["bpf_edits"],
                agent_spawns=agg["agent_spawns"],
            )
        )
    lines.append("")
    lines.append("## All Primary Sessions Summary")
    lines.append("")
    lines.append("| Session | Primary lines | Bad lines | First timestamp | Last timestamp | Wall-clock | Active span | Active windows | Assistant turns | Total tokens | Cost | Tool calls | Builds | Benchmarks | `.bpf.c` edits | Agent spawns | Subagent files | Subagent lines |")
    lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in sorted(primary_metrics, key=lambda metric: metric.first_ts or datetime.min.replace(tzinfo=timezone.utc)):
        nested_files, nested_lines = nested_counts[item.session_id]
        total_tokens = item.input_tokens + item.output_tokens
        lines.append(
            "| {session} | {line_count:,} | {bad:,} | {first} | {last} | {wall} | {active} | {windows} | {assistant:,} | {tokens:,} | {cost} | {tools:,} | {builds:,} | {benchmarks:,} | {bpf_edits:,} | {agent_spawns:,} | {nested_files:,} | {nested_lines:,} |".format(
                session=item.short_id,
                line_count=item.line_count,
                bad=item.malformed_lines,
                first=fmt_dt(item.first_ts),
                last=fmt_dt(item.last_ts),
                wall=duration_str(item.duration),
                active=duration_str(item.active_duration),
                windows=len(item.active_windows),
                assistant=item.assistant_turns,
                tokens=total_tokens,
                cost=money_str(token_cost(item.input_tokens, item.output_tokens)),
                tools=item.total_tool_invocations,
                builds=item.build_count,
                benchmarks=item.benchmark_count,
                bpf_edits=item.bpf_edit_count,
                agent_spawns=item.agent_spawns,
                nested_files=nested_files,
                nested_lines=nested_lines,
            )
        )
    lines.append("")
    lines.append("## Requested Session Details")
    lines.append("")
    lines.append("### Timing and Tokens")
    lines.append("")
    lines.append("| Session | Primary lines | First timestamp | Last timestamp | Wall-clock | Active span | Active windows | Assistant turns | Input tokens | Output tokens | Total tokens | Cost | Subagent files | Subagent lines |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in focus_metrics:
        nested_files, nested_lines = nested_counts[item.session_id]
        lines.append(
            "| {session} | {line_count:,} | {first} | {last} | {wall} | {active} | {windows} | {assistant:,} | {input_tokens:,} | {output_tokens:,} | {total_tokens:,} | {cost} | {nested_files:,} | {nested_lines:,} |".format(
                session=item.short_id,
                line_count=item.line_count,
                first=fmt_dt(item.first_ts),
                last=fmt_dt(item.last_ts),
                wall=duration_str(item.duration),
                active=duration_str(item.active_duration),
                windows=len(item.active_windows),
                assistant=item.assistant_turns,
                input_tokens=item.input_tokens,
                output_tokens=item.output_tokens,
                total_tokens=item.input_tokens + item.output_tokens,
                cost=money_str(token_cost(item.input_tokens, item.output_tokens)),
                nested_files=nested_files,
                nested_lines=nested_lines,
            )
        )
    lines.append("")
    lines.append("### Tool Counts")
    lines.append("")
    tool_headers = ["Session"] + COUNTED_TOOLS + ["Total tool calls"]
    lines.append("| " + " | ".join(tool_headers) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:" for _ in tool_headers[1:]]) + " |")
    for item in focus_metrics:
        values = [item.short_id] + [fmt_int(item.tool_counts.get(name, 0)) for name in COUNTED_TOOLS] + [fmt_int(item.total_tool_invocations)]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("### Action and Failure Metrics")
    lines.append("")
    lines.append("| Session | Builds | Module ops | Benchmarks | Total edits | `.bpf.c` edits | Agent spawns | Bash output lines | Error lines | Xid lines | Verifier lines | Fail lines |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in focus_metrics:
        lines.append(
            "| {session} | {builds:,} | {module_ops:,} | {benchmarks:,} | {total_edits:,} | {bpf_edits:,} | {agent_spawns:,} | {bash_lines:,} | {error_lines:,} | {xid_lines:,} | {verifier_lines:,} | {fail_lines:,} |".format(
                session=item.short_id,
                builds=item.build_count,
                module_ops=item.module_ops_count,
                benchmarks=item.benchmark_count,
                total_edits=item.total_edits,
                bpf_edits=item.bpf_edit_count,
                agent_spawns=item.agent_spawns,
                bash_lines=item.bash_output_lines,
                error_lines=item.error_lines,
                xid_lines=item.xid_lines,
                verifier_lines=item.verifier_lines,
                fail_lines=item.fail_lines,
            )
        )
    lines.append("")
    lines.append("### Active Windows")
    lines.append("")
    for item in focus_metrics:
        lines.append(f"- `{item.short_id}`:")
        for index, window in enumerate(item.active_windows, start=1):
            lines.append(f"  - window {index}: {fmt_window(window)}")
    lines.append("")
    lines.append("## Case-Study Windows")
    lines.append("")
    lines.append("| Case | Primary session | Selected active windows | Case first timestamp | Case last timestamp | Wall-clock span | Active span | Transcript count in window | Assistant turns | Input tokens | Output tokens | Total tokens | Cost | Total edits | `.bpf.c` edits | Builds | Benchmarks | Agent spawns |")
    lines.append("| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case_key in ("A", "B", "C"):
        payload = case_data[case_key]
        lines.append(
            "| {case} ({name}) | {session} | {window_count} | {first} | {last} | {wall} | {active} | {transcripts} | {assistant:,} | {input_tokens:,} | {output_tokens:,} | {total_tokens:,} | {cost} | {edits:,} | {bpf_edits:,} | {builds:,} | {benchmarks:,} | {agent_spawns:,} |".format(
                case=case_key,
                name=payload["name"],
                session=payload["session_prefix"],
                window_count=len(payload["windows"]),
                first=fmt_dt(payload["first_ts"]),
                last=fmt_dt(payload["last_ts"]),
                wall=duration_str(payload["duration"]),
                active=duration_str(payload["active_duration"]),
                transcripts=len(payload["metrics"]["transcript_paths"]),
                assistant=payload["metrics"]["assistant_turns"],
                input_tokens=payload["metrics"]["input_tokens"],
                output_tokens=payload["metrics"]["output_tokens"],
                total_tokens=payload["metrics"]["total_tokens"],
                cost=money_str(token_cost(payload["metrics"]["input_tokens"], payload["metrics"]["output_tokens"])),
                edits=payload["metrics"]["total_edits"],
                bpf_edits=payload["metrics"]["bpf_edits"],
                builds=payload["metrics"]["builds"],
                benchmarks=payload["metrics"]["benchmarks"],
                agent_spawns=payload["metrics"]["agent_spawns"],
            )
        )
    lines.append("")
    lines.append("### Case Window Details")
    lines.append("")
    for case_key in ("A", "B", "C"):
        payload = case_data[case_key]
        markers = ", ".join(f"`{marker}`" for marker in payload["marker_strings"])
        lines.append(f"- Case {case_key} ({payload['name']}), markers: {markers}")
        for index, window in enumerate(payload["windows"], start=1):
            lines.append(f"  - window {index}: {fmt_window(window)}")
    lines.append("")
    lines.append("## Notable Extremes and Derived Statistics")
    lines.append("")
    lines.append(
        f"- Longest single no-break active window: `{longest_session.short_id}` at {fmt_window(longest_window)}"
    )
    lines.append(
        f"- Most edits in one primary session: `{most_edits_session.short_id}` with {most_edits_session.total_edits:,} Edit/Write invocations"
    )
    lines.append(
        f"- Highest raw Bash error-line count: `{raw_error_session.short_id}` with {raw_error_session.error_lines:,} matching lines"
    )
    if error_rate_session is not None:
        lines.append(
            f"- Highest Bash error-line rate among sessions with at least 100 Bash output lines: `{error_rate_session.short_id}` with {error_rate_session.error_lines:,}/{error_rate_session.bash_output_lines:,} = {safe_ratio(error_rate_session.error_lines, error_rate_session.bash_output_lines):.2%}"
        )
    if peak_session is not None:
        breakdown = ", ".join(
            f"{name}={count}" for name, count in peak_density[3].most_common()
        )
        lines.append(
            f"- Peak tool-use density: `{peak_session.short_id}` with {peak_density[0]:,} tool calls in one hour, window {fmt_dt(peak_density[1])} -> {fmt_dt(peak_density[2])}; breakdown: {breakdown}"
        )
    lines.append(
        f"- Inclusive token cost estimate at Claude Opus pricing ($15/M input, $75/M output): {money_str(token_cost(all_agg['input_tokens'], all_agg['output_tokens']))}"
    )
    if math.isfinite(tokens_per_iteration):
        lines.append(
            f"- Average tokens per benchmark-triggered policy iteration (proxy = total inclusive tokens / total inclusive benchmark commands): {tokens_per_iteration:,.1f}"
        )
    if math.isfinite(thinking_action_ratio):
        lines.append(
            f"- Thinking/action output-token proxy ratio (inclusive, record-level proxy): {all_agg['thinking_output_proxy']:,}/{all_agg['action_output_proxy']:,} = {thinking_action_ratio:.4f} ({thinking_action_ratio:.2%})"
        )
    lines.append(
        f"- Subagent spawn vs direct tool work in primary sessions: {primary_agg['agent_spawns']:,} Agent calls vs {total_direct_tools_primary:,} non-Agent tool calls ({safe_ratio(primary_agg['agent_spawns'], primary_agg['tool_invocations']):.2%} of primary tool invocations were subagent spawns)"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `Assistant turns` means raw assistant records in the transcript, not deduplicated human-visible conversation turns. A single interactive turn can emit multiple assistant records (for example thinking, then tool use, then final text)."
    )
    lines.append(
        "- The thinking/action split is a proxy based on assistant record type: records that contain `tool_use` blocks are counted as `action`, records with `thinking` but no `tool_use` are counted as `thinking`."
    )
    lines.append(
        "- Case-study metrics are inclusive across nested subagent transcripts during the selected windows. The per-session tables above remain primary-transcript-only to preserve one row per top-level session."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    global BASE
    parser = argparse.ArgumentParser(description="Recompute the historical Q6 transcript metrics.")
    parser.add_argument("--corpus", type=Path, required=True, help="archived project JSONL directory")
    parser.add_argument("--output", type=Path, required=True, help="destination Markdown report")
    args = parser.parse_args()
    BASE = args.corpus.resolve()
    if not BASE.is_dir():
        parser.error(f"corpus directory does not exist: {BASE}")
    primary_paths = sorted(path for path in BASE.glob("*.jsonl"))
    primary_metrics = [parse_transcript(path, "primary") for path in primary_paths]
    primary_by_prefix = {item.short_id: item for item in primary_metrics}
    missing = sorted(set(FOCUS_PREFIXES) - primary_by_prefix.keys())
    if missing:
        parser.error(f"historical corpus is incomplete; missing session prefixes: {', '.join(missing)}")

    nested_paths: list[Path] = []
    nested_counts: dict[str, tuple[int, int]] = {}
    session_nested_paths: dict[str, list[Path]] = {}
    for item in primary_metrics:
        file_count, total_lines, paths = count_nested_jsonl_metrics(item.session_id)
        nested_counts[item.session_id] = (file_count, total_lines)
        session_nested_paths[item.session_id] = paths
        nested_paths.extend(paths)
    nested_metrics = [parse_transcript(path, "nested") for path in nested_paths]

    focus_metrics = [primary_by_prefix[prefix] for prefix in FOCUS_PREFIXES]

    case_data: dict[str, dict] = {}
    for case_key, spec in CASE_SPECS.items():
        session = prefix_lookup(primary_by_prefix, spec["session_prefix"])
        windows = session.active_windows
        matched_indices = find_active_windows_for_case(session.path, windows, spec)
        selected_windows = [windows[index] for index in matched_indices]
        relevant_paths = [session.path] + session_nested_paths[session.session_id]
        metrics = metrics_for_windows(relevant_paths, selected_windows)
        timestamps = sorted(metrics["timestamps"])
        first_ts = min(timestamps) if timestamps else None
        last_ts = max(timestamps) if timestamps else None
        active_span = sum((end - start for start, end in selected_windows), timedelta(0))
        case_data[case_key] = {
            "name": spec["name"],
            "session_prefix": spec["session_prefix"],
            "windows": selected_windows,
            "metrics": metrics,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "duration": (last_ts - first_ts) if first_ts and last_ts else timedelta(0),
            "active_duration": active_span,
            "marker_strings": spec.get("marker_strings", [pattern.pattern for pattern in spec["patterns"]]),
        }

    markdown = render_markdown(
        primary_metrics=primary_metrics,
        nested_metrics=nested_metrics,
        focus_metrics=focus_metrics,
        nested_counts=nested_counts,
        case_data=case_data,
    )
    args.output.write_text(markdown)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
