#!/usr/bin/env python3
"""Stop-before-exec tenant launcher for fig13-fast.

The harness Popen()s this helper with the real tenant argv. The helper
execs its interpreter normally (so Popen returns instead of blocking on
the exec error pipe), immediately SIGSTOPs itself, and waits for
SIGCONT. When the harness resumes it, the helper execs the real tenant
image in this same process, so the PID the harness captured before any
policy starts is the PID that runs the tenant. The target must be the
/tmp symlink path (e.g. /tmp/uvmbench_high) so exec keys comm on the
symlink basename.
"""

import os
import signal
import sys


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: tenant_launcher.py TARGET [ARGS...]\n")
        return 2
    target = sys.argv[1]
    print(f"tenant_launcher: pid={os.getpid()} target={target} stopping", flush=True)
    os.kill(os.getpid(), signal.SIGSTOP)
    try:
        os.execve(target, sys.argv[1:], dict(os.environ))
    except OSError as exc:
        sys.stderr.write(f"tenant_launcher: exec {target} failed: {exc}\n")
        return 127


if __name__ == "__main__":
    sys.exit(main())
