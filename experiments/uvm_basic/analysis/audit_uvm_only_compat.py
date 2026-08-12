#!/usr/bin/env python3
"""Statically audit whether custom nvidia-uvm can use the loaded distribution core."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def required_versions(module: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in command("modprobe", "--dump-modversions", str(module)).splitlines():
        crc, symbol = line.split(None, 1)
        versions[symbol] = crc.lower()
    return versions


def symvers(path: Path) -> dict[tuple[str, str], str]:
    values: dict[tuple[str, str], str] = {}
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 4:
            values[(fields[2], fields[1])] = fields[0].lower()
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--custom-uvm", type=Path, required=True)
    parser.add_argument("--custom-symvers", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    custom_uvm = args.custom_uvm.resolve()
    system_uvm = Path(command("modinfo", "-n", "nvidia_uvm")).resolve()
    release = os.uname().release
    version = command("modinfo", "-F", "version", str(system_uvm))
    distribution_symvers = Path(
        f"/var/lib/dkms/nvidia/{version}/{release}/x86_64/module/Module.symvers"
    )
    kernel_symvers = Path(f"/lib/modules/{release}/build/Module.symvers")
    paths = (custom_uvm, system_uvm, args.custom_symvers, distribution_symvers, kernel_symvers)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing compatibility input: " + ", ".join(missing))

    custom_required = required_versions(custom_uvm)
    system_required = required_versions(system_uvm)
    shared = custom_required.keys() & system_required.keys()
    shared_mismatches = {
        symbol: {"custom": custom_required[symbol], "distribution": system_required[symbol]}
        for symbol in shared if custom_required[symbol] != system_required[symbol]
    }
    custom_only = custom_required.keys() - system_required.keys()
    kernel_exports = symvers(kernel_symvers)
    custom_only_kernel = {}
    for symbol in sorted(custom_only):
        expected = custom_required[symbol]
        actual = kernel_exports.get(("vmlinux", symbol))
        custom_only_kernel[symbol] = {
            "required_crc": expected,
            "kernel_crc": actual,
            "match": expected == actual,
        }

    distribution_exports = symvers(distribution_symvers)
    custom_exports = symvers(args.custom_symvers)
    dist_core = {symbol: crc for (module, symbol), crc in distribution_exports.items() if module == "nvidia"}
    custom_core = {symbol: crc for (module, symbol), crc in custom_exports.items() if module == "nvidia"}
    shared_core = dist_core.keys() & custom_core.keys()
    core_mismatches = {
        symbol: {"custom": custom_core[symbol], "distribution": dist_core[symbol]}
        for symbol in shared_core if custom_core[symbol] != dist_core[symbol]
    }
    custom_core_only = custom_core.keys() - dist_core.keys()
    custom_core_only_required = sorted(custom_core_only & custom_required.keys())
    nv_interfaces = sorted(symbol for symbol in custom_required if symbol.startswith("nvUvmInterface"))
    nv_interface_mismatches = [
        symbol for symbol in nv_interfaces
        if system_required.get(symbol) != custom_required[symbol]
    ]
    version_match = command("modinfo", "-F", "version", str(custom_uvm)) == version
    vermagic_match = command("modinfo", "-F", "vermagic", str(custom_uvm)).startswith(release + " ")
    static_compatible = (
        version_match and vermagic_match and not shared_mismatches and not core_mismatches
        and not nv_interface_mismatches
        and all(item["match"] for item in custom_only_kernel.values())
        and not custom_core_only_required
    )
    status = ("SUPPORTED_BY_STATIC_MODVERSION_EVIDENCE" if static_compatible
              else "FULL_STACK_OR_REBUILD_REQUIRED")
    data: dict[str, Any] = {
        "evidence_class": "STATIC_KERNEL_MODULE_ABI",
        "status": status,
        "recommendation": "TRY_UVM_ONLY_SWITCH_FIRST" if static_compatible else "DO_NOT_TRY_UVM_ONLY_SWITCH",
        "limitations": [
            "Static modversion evidence cannot prove that insmod will succeed.",
            "A manual maintenance-window load must still check unresolved symbols and kernel logs.",
        ],
        "kernel_release": release,
        "driver_version_match": version_match,
        "vermagic_match": vermagic_match,
        "custom_uvm": {"path": str(custom_uvm), "sha256": digest(custom_uvm)},
        "distribution_uvm": {"path": str(system_uvm), "sha256": digest(system_uvm)},
        "required_symbols": {
            "custom_count": len(custom_required), "distribution_count": len(system_required),
            "shared_count": len(shared), "shared_crc_mismatches": shared_mismatches,
            "custom_only": custom_only_kernel,
        },
        "nvidia_interfaces": {"count": len(nv_interfaces), "crc_mismatches": nv_interface_mismatches},
        "nvidia_core_exports": {
            "distribution_count": len(dist_core), "custom_count": len(custom_core),
            "shared_count": len(shared_core), "shared_crc_mismatches": core_mismatches,
            "custom_only_exports": sorted(custom_core_only),
            "custom_only_exports_required_by_custom_uvm": custom_core_only_required,
        },
        "inputs": {
            "distribution_symvers": str(distribution_symvers),
            "custom_symvers": str(args.custom_symvers.resolve()),
            "kernel_symvers": str(kernel_symvers),
        },
        "root_operation_executed": False,
    }
    output = experiment / "results" / "uvm_only_compatibility.json"
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    lines = [
        "# UVM-only Module Compatibility", "", f"Status: `{status}`.", "",
        f"Recommendation: `{data['recommendation']}`.", "",
        f"- Shared UVM required symbols: {len(shared)}; CRC mismatches: {len(shared_mismatches)}",
        f"- Shared `nvUvmInterface*` symbols: {len(nv_interfaces)}; CRC mismatches: {len(nv_interface_mismatches)}",
        f"- Shared distribution/custom `nvidia.ko` exports: {len(shared_core)}; CRC mismatches: {len(core_mismatches)}",
        f"- Custom-only required kernel symbols: {len(custom_only_kernel)}; all kernel CRCs match: {all(item['match'] for item in custom_only_kernel.values())}",
        f"- Custom-only `nvidia.ko` exports required by custom UVM: {len(custom_core_only_required)}", "",
        "This supports trying the UVM-only switch before a full-stack switch. It is not runtime load proof; unresolved-symbol and kernel-log checks remain mandatory during the manual maintenance window.", "",
        "The manual UVM-only action checks only active CUDA/UVM users and the `nvidia_uvm` use count. It intentionally keeps the loaded `nvidia`, `nvidia_modeset`, and `nvidia_drm` stack in place; full-stack switching retains stricter display and all-device-user checks.", "",
    ]
    (experiment / "docs" / "UVM_ONLY_COMPATIBILITY.md").write_text("\n".join(lines))
    print(json.dumps({"status": status, "recommendation": data["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
