"""Build only the changed MoE store extension with an explicit real compiler."""
import os
from pathlib import Path
import runpy
import sys

import setuptools

here = Path(__file__).resolve().parent
os.environ.update(CC="/usr/bin/gcc-13", CXX="/usr/bin/g++-13",
                  CUDA_HOME="/usr/local/cuda-12.9", CUDA_VISIBLE_DEVICES="",
                  CPLUS_INCLUDE_PATH=str(here.parents[1] / "extension"),
                  MOE_ENABLE_SM120="1", MOE_ENABLE_SM90="0", NVTX_DISABLE="1",
                  MAX_JOBS="2", CUTLASS_DIR=str(here / "deps/cutlass"))
original_setup = setuptools.setup


def store_setup(**kwargs):
    kwargs["ext_modules"] = [extension for extension in kwargs["ext_modules"]
                             if extension.name == "moe_infinity._store"]
    if len(kwargs["ext_modules"]) != 1:
        raise RuntimeError("expected exactly one upstream store extension")
    return original_setup(**kwargs)


setuptools.setup = store_setup
os.chdir(here / "deps/MoE-Infinity")
setup_path = here / "deps/MoE-Infinity/setup.py"
sys.argv = [str(setup_path), "build_ext", "--inplace"]
runpy.run_path(str(setup_path), run_name="__main__")
