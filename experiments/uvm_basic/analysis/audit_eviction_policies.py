#!/usr/bin/env python3
"""Conservative static audit for gpu_ext struct_ops memory policies."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


INITIAL_CANDIDATES = {
    "eviction_fifo",
    "prefetch_always_max_cycle_moe",
    "prefetch_cooperative",
}


def map_declarations(text: str) -> list[dict[str, object]]:
    maps: list[dict[str, object]] = []
    pattern = re.compile(
        r"struct\s*\{(?P<body>.*?)\}\s*(?P<name>[A-Za-z_]\w*)\s+SEC\(\"\.maps\"\);",
        re.S,
    )
    for match in pattern.finditer(text):
        body = match.group("body")
        map_type = re.search(r"__uint\(type,\s*([^\)]+)\)", body)
        entries = re.search(r"__uint\(max_entries,\s*([^\)]+)\)", body)
        maps.append(
            {
                "name": match.group("name"),
                "type": map_type.group(1).strip() if map_type else None,
                "max_entries": entries.group(1).strip() if entries else None,
            }
        )
    return maps


def function_body(text: str, function: str) -> str:
    match = re.search(rf"\b{re.escape(function)}\s*\([^)]*\)\s*\{{", text, re.S)
    if not match:
        # BPF_PROG wraps the function name and arguments.
        match = re.search(rf"BPF_PROG\(\s*{re.escape(function)}\s*,.*?\)\s*\{{", text, re.S)
    if not match:
        return ""
    start = match.end()
    depth = 1
    index = start
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return text[start : index - 1]


def audit_file(path: Path) -> dict[str, object] | None:
    text = path.read_text(errors="replace")
    if 'SEC(".struct_ops")' not in text:
        return None
    policy = path.name.removesuffix(".bpf.c")
    hooks = sorted(set(re.findall(r'SEC\("struct_ops/([^"/]+)"\)', text)))
    kfuncs = sorted(set(re.findall(r"\b(bpf_gpu_[A-Za-z0-9_]+)\s*\(", text)))
    activate = function_body(text, "gpu_block_activate")
    access = function_body(text, "gpu_block_access")
    has_move_head = "bpf_gpu_block_move_head" in text
    has_move_tail = "bpf_gpu_block_move_tail" in text
    has_migrate = "bpf_gpu_migrate_range" in text
    has_wq = bool(re.search(r"\bbpf_wq\b|\bbpf_wq_", text))
    direct_list = bool(re.search(r"\blist_(?:move|move_tail|add|add_tail|del)\s*\(", text))
    loops = bool(re.search(r"\b(?:for|while)\s*\(", text))
    printks = len(re.findall(r"\bbpf_printk\s*\(", text))
    combined = "gpu_page_prefetch" in hooks and any(
        hook in hooks for hook in ("gpu_block_activate", "gpu_evict_prepare")
    )
    non_fixed_key = bool(
        re.search(r"bpf_map_lookup_elem\([^,]+,\s*&(?:idx|slot|rk|wq_key)\s*\)", text)
    )
    reasons: list[str] = []
    if policy not in INITIAL_CANDIDATES:
        reasons.append("not in the explicitly scoped Stage 4 initial candidate set")
    if has_move_head and "bpf_gpu_block_move_head" in activate:
        reasons.append("gpu_block_activate calls bpf_gpu_block_move_head")
    if direct_list:
        reasons.append("direct linked-list mutation detected")
    if has_wq:
        reasons.append("uses bpf_wq; bounded completion and context safety are not proven")
    if has_migrate:
        reasons.append("performs cross-VA-block migration via bpf_gpu_migrate_range")
    if printks:
        reasons.append("contains bpf_printk in a memory-management hot path")
    if "gpu_block_access" in hooks and access and "return 1" in access:
        reasons.append("behavior depends on gpu_block_access, which is not a reliable current hook")
    if policy == "eviction_fifo" and not has_move_head and not has_move_tail:
        reasons.append("FIFO comments claim list ordering but the implementation does not reorder chunks")

    suitable = not reasons
    return {
        "policy": policy,
        "file": str(path),
        "struct_ops_hooks": hooks,
        "kfunc_calls": kfuncs,
        "maps": map_declarations(text),
        "calls_bpf_gpu_block_move_head": has_move_head,
        "calls_bpf_gpu_block_move_tail": has_move_tail,
        "calls_bpf_gpu_migrate_range": has_migrate,
        "uses_bpf_wq": has_wq,
        "direct_list_head_access": direct_list,
        "contains_loop": loops,
        "non_fixed_map_key": non_fixed_key,
        "combines_prefetch_and_eviction": combined,
        "bpf_printk_count": printks,
        "suitable_for_initial_pressure_test": suitable,
        "decision": "APPROVED_FOR_SMOKE" if suitable else "REJECTED_INITIAL_STAGE4",
        "rejection_reasons": reasons,
    }


def audit(extension_dir: Path) -> dict[str, object]:
    policies = []
    for path in sorted(extension_dir.glob("*.bpf.c")):
        item = audit_file(path)
        if item:
            policies.append(item)
    return {
        "schema_version": "1.0",
        "evidence_class": "SOURCE_STATIC_AUDIT",
        "extension_dir": str(extension_dir.resolve()),
        "policies": policies,
        "approved_for_smoke": [
            item["policy"] for item in policies if item["suitable_for_initial_pressure_test"]
        ],
        "limitations": [
            "Static approval is necessary but not sufficient; attach/detach and runtime smoke remain required.",
            "The audit does not prove semantic correctness of NVIDIA UVM list ordering.",
        ],
    }


def write_markdown(data: dict[str, object], path: Path) -> None:
    rows = [
        "# Stage 4 Eviction Policy Safety Audit",
        "",
        "Evidence class: `SOURCE_STATIC_AUDIT`.",
        "",
        "| Policy | Hooks | Kfuncs | Combined | Decision | Reasons |",
        "|---|---|---|---|---|---|",
    ]
    for item in data["policies"]:
        reasons = "; ".join(item["rejection_reasons"]) or "none"
        rows.append(
            f"| `{item['policy']}` | {', '.join(item['struct_ops_hooks']) or 'none'} | "
            f"{', '.join(item['kfunc_calls']) or 'none'} | "
            f"{'yes' if item['combines_prefetch_and_eviction'] else 'no'} | "
            f"`{item['decision']}` | {reasons} |"
        )
    rows += [
        "",
        "## Enforcement",
        "",
        "Only policies marked `APPROVED_FOR_SMOKE` may enter the 64 MiB and 0.95x smoke gates. "
        "Static approval does not grant permission to skip the runtime safety checks.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows))


def write_legacy_stage3(data: dict[str, object], experiment_dir: Path) -> None:
    names = {
        "eviction_fifo", "eviction_cycle_moe", "prefetch_always_max_cycle_moe",
        "prefetch_cooperative", "eviction_lfu", "eviction_mru",
    }
    rows = []
    for item in data["policies"]:
        if item["policy"] not in names:
            continue
        rows.append({
            "policy": item["policy"],
            "source": item["file"],
            "hooks": item["struct_ops_hooks"],
            "kfuncs": item["kfunc_calls"],
            "move_head": item["calls_bpf_gpu_block_move_head"],
            "move_tail": item["calls_bpf_gpu_block_move_tail"],
            "direct_list_operation": item["direct_list_head_access"],
            "sleepable_kfunc_or_wq": item["uses_bpf_wq"] or item["calls_bpf_gpu_migrate_range"],
            "high_frequency_printk": bool(item["bpf_printk_count"]),
            "bounded_state": any(entry["type"] in {"BPF_MAP_TYPE_ARRAY", "BPF_MAP_TYPE_PERCPU_ARRAY"}
                                 for entry in item["maps"]),
            "initial_oversub_suitable": not (
                item["calls_bpf_gpu_block_move_head"] or item["direct_list_head_access"] or
                item["uses_bpf_wq"] or item["calls_bpf_gpu_migrate_range"] or
                bool(item["bpf_printk_count"])
            ),
        })
    target = experiment_dir / "results" / "stage3" / "eviction_policy_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Eviction Policy Safety Audit", "",
        "Evidence class: `STATIC_SOURCE_AUDIT`. No policy was attached or executed.", "",
        "| Policy | Hooks | move_head | move_tail | Direct list | Sleepable/WQ | printk | Bounded state | Initial suitable |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | {', '.join(row['hooks']) or 'none'} | {row['move_head']} | "
            f"{row['move_tail']} | {row['direct_list_operation']} | {row['sleepable_kfunc_or_wq']} | "
            f"{row['high_frequency_printk']} | {row['bounded_state']} | "
            f"{row['initial_oversub_suitable']} |"
        )
    (experiment_dir / "docs" / "EVICTION_POLICY_SAFETY_AUDIT.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--gpu-ext-root", type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()
    if args.gpu_ext_root or args.experiment_dir:
        if not (args.gpu_ext_root and args.experiment_dir):
            parser.error("--gpu-ext-root and --experiment-dir must be used together")
        data = audit(args.gpu_ext_root / "extension")
        write_legacy_stage3(data, args.experiment_dir.resolve())
        print(f"wrote {args.experiment_dir.resolve() / 'results/stage3/eviction_policy_audit.json'}")
        return 0
    if not (args.extension_dir and args.json and args.markdown):
        parser.error("Stage 4 mode requires --extension-dir, --json, and --markdown")
    data = audit(args.extension_dir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    write_markdown(data, args.markdown)
    print(json.dumps({"approved_for_smoke": data["approved_for_smoke"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
