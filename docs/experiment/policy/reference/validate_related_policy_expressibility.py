#!/usr/bin/env python3
"""Offline structural checks for the related-policy expressibility inventory."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_INVENTORY = HERE / "related-policy-expressibility.json"

CLASSIFICATIONS = {"FULL", "ANALOGUE", "PARTIAL", "NO"}
EVIDENCE_LEVELS = {"source", "build", "engagement", "performance"}
CATEGORIES = {
    "uvm_oversubscription",
    "prefetch_placement",
    "moe_expert_caching",
    "kv_weight_tiering",
    "gpu_scheduling_qos",
    "multi_gpu_storage",
    "userspace_device_ebpf",
}
DISALLOWED_LOCAL_ASSETS = {
    "forest_isca25.pdf",
    "helm_sc25.pdf",
    "dream_ics25.pdf",
    "suv_micro24.pdf",
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def require_nonempty_strings(
    errors: list[str], value: object, field: str, record_id: str
) -> None:
    if not isinstance(value, list) or not value:
        fail(errors, f"{record_id}: {field} must be a non-empty list")
        return
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            fail(errors, f"{record_id}: {field}[{index}] must be a non-empty string")


def validate(inventory_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {inventory_path}: {exc}"]

    if data.get("schema_version") != 1:
        fail(errors, "schema_version must be 1")

    categories = data.get("categories")
    if not isinstance(categories, list) or set(categories) != CATEGORIES:
        fail(errors, f"categories must contain exactly: {sorted(CATEGORIES)}")

    records = data.get("records")
    if not isinstance(records, list):
        return errors + ["records must be a list"]
    if len(records) < 30:
        fail(errors, f"records must contain at least 30 papers; found {len(records)}")

    ids: set[str] = set()
    urls: set[str] = set()
    category_counts: Counter[str] = Counter()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            fail(errors, f"records[{index}] must be an object")
            continue

        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            record_id = f"records[{index}]"
            fail(errors, f"{record_id}: id must be a non-empty string")
        elif record_id in ids:
            fail(errors, f"{record_id}: duplicate id")
        else:
            ids.add(record_id)

        for field in ("title", "venue", "mapping_reason"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                fail(errors, f"{record_id}: {field} must be a non-empty string")

        year = record.get("year")
        if not isinstance(year, int) or isinstance(year, bool) or not 2000 <= year <= 2100:
            fail(errors, f"{record_id}: year must be an integer in [2000, 2100]")

        category = record.get("category")
        if category not in CATEGORIES:
            fail(errors, f"{record_id}: invalid category {category!r}")
        else:
            category_counts[category] += 1

        classification = record.get("classification")
        if classification not in CLASSIFICATIONS:
            fail(errors, f"{record_id}: invalid classification {classification!r}")

        url = record.get("primary_url")
        if not isinstance(url, str):
            fail(errors, f"{record_id}: primary_url must be a string")
        else:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                fail(errors, f"{record_id}: primary_url must be an absolute HTTPS URL")
            if url in urls:
                fail(errors, f"{record_id}: duplicate primary_url {url}")
            urls.add(url)

        require_nonempty_strings(errors, record.get("observes"), "observes", record_id)
        require_nonempty_strings(errors, record.get("acts"), "acts", record_id)

        missing = record.get("missing_primitives")
        if not isinstance(missing, list) or any(
            not isinstance(item, str) or not item.strip() for item in missing
        ):
            fail(errors, f"{record_id}: missing_primitives must be a list of strings")
        if classification == "FULL" and missing:
            fail(errors, f"{record_id}: FULL records cannot list missing primitives")
        if classification in {"PARTIAL", "NO"} and not missing:
            fail(errors, f"{record_id}: {classification} records must state missing primitives")

        programs = record.get("in_tree_programs")
        if not isinstance(programs, list):
            fail(errors, f"{record_id}: in_tree_programs must be a list")
        else:
            for program in programs:
                if not isinstance(program, str) or not program.strip():
                    fail(errors, f"{record_id}: invalid in_tree_programs entry")
                    continue
                relative = Path(program)
                if relative.is_absolute() or ".." in relative.parts:
                    fail(errors, f"{record_id}: program path must be repository-relative: {program}")
                    continue
                candidate = REPO_ROOT / relative
                if not candidate.is_file():
                    fail(errors, f"{record_id}: in-tree program does not exist: {program}")

        evidence = record.get("evidence")
        if not isinstance(evidence, dict):
            fail(errors, f"{record_id}: evidence must be an object")
        else:
            level = evidence.get("level")
            if level not in EVIDENCE_LEVELS:
                fail(errors, f"{record_id}: invalid evidence level {level!r}")
            note = evidence.get("note")
            if not isinstance(note, str) or not note.strip():
                fail(errors, f"{record_id}: evidence.note must be non-empty")
            paths = evidence.get("paths")
            if not isinstance(paths, list):
                fail(errors, f"{record_id}: evidence.paths must be a list")
            else:
                if level != "source" and not paths:
                    fail(errors, f"{record_id}: {level} evidence requires at least one path")
                for evidence_path in paths:
                    if not isinstance(evidence_path, str) or not evidence_path.strip():
                        fail(errors, f"{record_id}: invalid evidence.paths entry")
                        continue
                    relative = Path(evidence_path)
                    if relative.is_absolute() or ".." in relative.parts:
                        fail(
                            errors,
                            f"{record_id}: evidence path must be repository-relative: {evidence_path}",
                        )
                        continue
                    if not (REPO_ROOT / relative).exists():
                        fail(errors, f"{record_id}: evidence path does not exist: {evidence_path}")

        serialized = json.dumps(record, ensure_ascii=False)
        for bad_name in DISALLOWED_LOCAL_ASSETS:
            if bad_name in serialized:
                fail(errors, f"{record_id}: references known-mismatched local asset {bad_name}")

    missing_categories = CATEGORIES - set(category_counts)
    if missing_categories:
        fail(errors, f"categories with no records: {sorted(missing_categories)}")
    for category, count in sorted(category_counts.items()):
        minimum = 2 if category == "userspace_device_ebpf" else 4
        if count < minimum:
            fail(errors, f"{category}: expected at least {minimum} papers, found {count}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", nargs="?", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()

    errors = validate(args.inventory.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    data = json.loads(args.inventory.read_text(encoding="utf-8"))
    counts = Counter(record["category"] for record in data["records"])
    print(f"validated {len(data['records'])} related-policy records")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
