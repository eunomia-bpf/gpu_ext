"""Explicit heavy-build entry, for the root launcher's later untimed window only.

This script does not stage sources or install packages. It uses the existing
FineMoE build environment and builds only into the already staged private copy.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def group_members(pgid):
    # Same owned-PGID survivor check as gpreempt/run_three_way.py. The worker
    # starts its own session; a finished leader does not imply finished compilers.
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid and int(fields[3]) == pgid:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(process):
    for sig, seconds in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        if process.poll() is not None and not group_members(process.pid):
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass  # An empty group does not yet establish that the leader reaped.
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if process.poll() is not None and not group_members(process.pid):
                return
            time.sleep(0.05)
    if process.poll() is not None and not group_members(process.pid):
        return
    raise RuntimeError(f"owned build group {process.pid} survived cleanup")


def interrupted(signum, _frame):
    raise InterruptedError(f"build wrapper received signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to((HERE / "build").resolve()) or not (source / "core/eb_section_vi/adapter_live.inc").is_file():
        raise ValueError("expected the privately staged EB source")
    if not args.worker:
        if args.log is None:
            parser.error("--log NEW_LOG is required")
        with args.log.open("x") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--worker", "--source", str(source)]
            previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
            process = None
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, start_new_session=True)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                return process.wait()
            finally:
                # Repeated TERM must not interrupt bounded cleanup halfway through.
                for sig in previous:
                    signal.signal(sig, signal.SIG_IGN)
                try:
                    if process is not None:
                        stop_owned(process)
                        process.stdout.close()
                    log.write("owned_build_group_empty=true\n")
                    log.flush()
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)
    os.environ.update(CC="/usr/bin/gcc-13", CXX="/usr/bin/g++-13",
                      CUDA_HOME="/usr/local/cuda-12.9", CUDA_VISIBLE_DEVICES="",
                      TORCH_CUDA_ARCH_LIST="12.0",
                      BUILD_OPS="1", MAX_JOBS="1", TORCH_NO_COMPILER_WRAPPER="1",
                      LDFLAGS="-Wl,--build-id=none", CPLUS_INCLUDE_PATH=str(HERE.parents[1] / "finemoe"))
    output_dir = source / "finemoe/ops/prefetch"
    if list(output_dir.glob("prefetch_op*.so")):
        raise ValueError("use a fresh private source without an old offloader binary")
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.argv = [str(source / "setup.py"), "build_ext", "--inplace"]
    runpy.run_path(str(source / "setup.py"), run_name="__main__")
    binaries = list(output_dir.glob("prefetch_op*.so"))
    if len(binaries) != 1 or binaries[0].is_symlink() or binaries[0].stat().st_size <= 0:
        raise RuntimeError("build did not create exactly one nonempty private offloader")
    binary = binaries[0].resolve()
    # Import exactly the newly built extension, not the frozen package or a
    # site-packages module. Do not construct the runtime or initialize CUDA here.
    spec = importlib.util.spec_from_file_location("prefetch_op", binary)
    torch = sys.modules["torch"]  # Already imported by the actual build.
    if torch.cuda.is_initialized():
        raise RuntimeError("build unexpectedly initialized CUDA")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (Path(module.__file__).resolve() != binary or
            getattr(module, "expert_buffering_runtime_revision", None) != "section-vi-private-adapter-v1" or
            not callable(getattr(module, "expert_buffering_snapshot", None))):
        raise RuntimeError("new offloader import lacks the Section VI runtime interface")
    if torch.cuda.is_initialized():
        raise RuntimeError("offloader import unexpectedly initialized CUDA")
    print(f"private_offloader={binary} bytes={binary.stat().st_size}")
    print("private_offloader_import=passed runtime_revision=section-vi-private-adapter-v1 cuda_initialized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
