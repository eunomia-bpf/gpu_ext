#!/usr/bin/env python3
"""Validate the local policy-paper inventory using paths, sizes, and PDF tools."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.json"


def valid_https(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def main() -> int:
    errors: list[str] = []
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    papers = data.get("papers")
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(papers, list) or len(papers) < 40:
        errors.append("papers must contain at least 40 entries")
        papers = []

    ids: set[int] = set()
    declared_files: set[str] = set()
    retained_count = 0
    retained_bytes = 0
    for index, paper in enumerate(papers):
        label = f"papers[{index}]"
        if not isinstance(paper, dict):
            errors.append(f"{label} must be an object")
            continue
        paper_id = paper.get("id")
        if not isinstance(paper_id, int) or paper_id in ids:
            errors.append(f"{label} has an invalid or duplicate id")
        else:
            ids.add(paper_id)
            label = f"paper {paper_id}"
        for field in ("title", "venue", "category", "title_check_status", "notes"):
            value = paper.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label}: {field} must be a non-empty string")
        if not valid_https(paper.get("source_landing_url")):
            errors.append(f"{label}: source_landing_url must be HTTPS")

        filename = paper.get("local_filename")
        size = paper.get("size_bytes")
        if filename is None:
            if size is not None:
                errors.append(f"{label}: missing PDF cannot declare size_bytes")
            if paper.get("title_check_status", "").startswith("verified-"):
                errors.append(f"{label}: missing PDF cannot be marked verified")
            continue

        if not isinstance(filename, str) or Path(filename).name != filename:
            errors.append(f"{label}: local_filename must be a plain filename")
            continue
        if filename in declared_files:
            errors.append(f"{label}: duplicate local_filename {filename}")
            continue
        declared_files.add(filename)
        if not valid_https(paper.get("direct_pdf_url")):
            errors.append(f"{label}: retained PDF must have an HTTPS direct URL")
        if paper.get("title_check_status") != "verified-title-and-authors-first-page":
            errors.append(f"{label}: retained PDF lacks the required title check")

        path = HERE / filename
        if not path.is_file():
            errors.append(f"{label}: missing local file {filename}")
            continue
        actual_size = path.stat().st_size
        if not isinstance(size, int) or size != actual_size:
            errors.append(
                f"{label}: recorded size {size!r} does not match {actual_size} bytes"
            )
        retained_count += 1
        retained_bytes += actual_size
        for command in (
            ["pdfinfo", str(path)],
            ["pdftotext", "-f", "1", "-l", "1", str(path), "-"],
        ):
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode:
                errors.append(
                    f"{label}: {' '.join(command[:1])} failed: {result.stderr.strip()}"
                )

    actual_files = {path.name for path in HERE.glob("*.pdf") if path.is_file()}
    for filename in sorted(actual_files - declared_files):
        errors.append(f"undeclared PDF: {filename}")
    for filename in sorted(declared_files - actual_files):
        errors.append(f"declared PDF is absent: {filename}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"validated {len(papers)} entries; "
        f"{retained_count} PDFs; {retained_bytes} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
