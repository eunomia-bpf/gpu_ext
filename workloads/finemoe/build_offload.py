"""Build the official CPU/CUDA-runtime extension without loading a model/GPU.

Use the existing CUDA-capable PyTorch installation, not the author's global setup
script. Runtime imports use a separate dependency overlay prepared independently.
"""
import os
from pathlib import Path
import runpy
import subprocess
import sys

here = Path(__file__).resolve().parent
source = here / "deps" / "FineMoE-EuroSys26"
if "--worker" not in sys.argv:
    if len(sys.argv) != 3 or sys.argv[1] != "--log":
        raise SystemExit("usage: build_offload.py --log NEW_BUILD_LOG")
    destination = Path(sys.argv[2]).resolve()
    with destination.open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        raise SystemExit(process.wait())
os.environ.update(CC="/usr/bin/gcc-13", CXX="/usr/bin/g++-13",
                  CUDA_HOME="/usr/local/cuda-12.9", CUDA_VISIBLE_DEVICES="",
                  BUILD_OPS="1", MAX_JOBS="2",
                  TORCH_NO_COMPILER_WRAPPER="1", LDFLAGS="-Wl,--build-id=none",
                  CPLUS_INCLUDE_PATH=str(here))
os.chdir(source)
sys.path.insert(0, str(source))
sys.argv = [str(source / "setup.py"), "build_ext", "--inplace"]
runpy.run_path(str(source / "setup.py"), run_name="__main__")
