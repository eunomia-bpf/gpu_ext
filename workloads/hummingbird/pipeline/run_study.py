#!/usr/bin/env python3
"""Run only the separately prepared Hummingbird pipeline campaign."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    source = Path(__file__).resolve().parent / 'build/src'
    sys.path.insert(0, str(source))
    runpy.run_path(str(source / 'run_study.py'), run_name='__main__')
