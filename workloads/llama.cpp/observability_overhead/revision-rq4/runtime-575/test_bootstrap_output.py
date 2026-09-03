"""Compile/run the actual bootstrap logger without CUDA or a loader segment."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    source = Path(__file__).resolve().with_name("test_bootstrap_logger.cpp")
    with tempfile.TemporaryDirectory(prefix="bpftime-bootstrap-output-cpu-") as directory:
        temp = Path(directory)
        binary = temp / "test-logger"
        flags = subprocess.check_output(["pkg-config", "--cflags", "--libs", "spdlog"], text=True).split()
        command = ["c++", "-std=c++20", "-O0", "-Wl,--build-id=none", str(source),
                   "-I" + str(runtime / "runtime/agent"),
                   "-I" + str(runtime / "runtime/include"),
                   "-I" + str(runtime / "vm/compat/include"), "-o", str(binary), *flags]
        subprocess.run(command, check=True, timeout=30)
        for destination in ("console", str(temp / "agent.log"), None):
            environment = os.environ.copy()
            environment.pop("LD_PRELOAD", None)
            environment["SPDLOG_LEVEL"] = "warn"
            if destination is None:
                environment.pop("BPFTIME_LOG_OUTPUT", None)
            else:
                environment["BPFTIME_LOG_OUTPUT"] = destination
            result = subprocess.run([str(binary)], env=environment, capture_output=True,
                                    text=True, check=True, timeout=10)
            if result.stdout != "application-output\n":
                raise RuntimeError(f"application stdout polluted: {result.stdout!r}")
            if destination == "console":
                diagnostics = result.stderr
            elif destination:
                diagnostics = Path(destination).read_text()
                if result.stderr:
                    raise RuntimeError("file-target diagnostics leaked to stderr")
            else:
                diagnostics = result.stderr
                if diagnostics:
                    raise RuntimeError("unset logger target is not quiet")
            if destination and "already registered, overwriting" not in diagnostics:
                raise RuntimeError("expected duplicate-registration diagnostic was lost")
            print(f"PASS destination={destination or 'unset'} exact_stdout=yes")
    print("CPU bootstrap logger only; full injected llama-cli correctness still requires a GPU rerun.")


if __name__ == "__main__":
    main()
