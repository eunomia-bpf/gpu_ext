"""Stage the private EB source copy or check its patch; never import torch/CUDA."""
import argparse
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[1] / "finemoe/deps/FineMoE-EuroSys26"
PATCH = HERE / "adapter-source.patch"


def check_patch(source):
    subprocess.run(["patch", "--dry-run", "--batch", "--forward", "--fuzz=0",
                    "-p1", "-i", str(PATCH)], cwd=source, check=True)


def stage(destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = destination.resolve()
    build = (HERE / "build").resolve()
    if not destination.is_relative_to(build) or destination == build:
        raise ValueError("private source destination must be a fresh child of section-vi/build")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    check_patch(FROZEN)
    destination.mkdir(parents=True)
    ignored_artifacts = shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.o", "*.a", ".git", "build")
    def ignore(directory, names):
        ignored = set(ignored_artifacts(directory, names))
        if Path(directory) == FROZEN / "core" and "core" in names:
            loop = Path(directory) / "core"
            if not loop.is_symlink() or loop.readlink() != Path("core"):
                raise ValueError("unexpected core/core entry; do not hide a source dependency")
            ignored.add("core")  # Existing non-source, self-referential build artifact.
        return ignored
    for directory in ("core", "finemoe", "op_builder"):
        # Keep finemoe/ops/{core,op_builder} as relative aliases within this copy.
        shutil.copytree(FROZEN / directory, destination / directory, ignore=ignore,
                        symlinks=True)
    for filename in ("setup.py", "requirements.txt", "README.md"):
        shutil.copy2(FROZEN / filename, destination / filename)
    subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(PATCH)],
                   cwd=destination, check=True)
    implementation = destination / "core/eb_section_vi"
    implementation.mkdir()
    for filename in ("policy.h", "adapter_state.h", "adapter_state.cpp", "adapter_live.inc"):
        shutil.copy2(HERE / filename, implementation / filename)
    print(f"private_source={destination}")
    print("source_staged_only: no torch import, offloader build, or GPU execution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path)
    args = parser.parse_args()
    if args.stage is None:
        check_patch(FROZEN)
    else:
        stage(args.stage)
