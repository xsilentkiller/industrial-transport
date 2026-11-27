"""Wait for the running comparison (current + C) to finish, then add Option B.

The comparison is resumable: the second call skips the finished current/C
runs and adds only B's runs, then rebuilds the report with all three methods.
"""

import subprocess
import sys
import time
from pathlib import Path

log = Path("comparison/main.log")
while "Saved comparison" not in (log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""):
    time.sleep(60)
print("current + C finished; starting Option B runs", flush=True)
with open("comparison/main_b.log", "w", encoding="utf-8") as out:
    code = subprocess.call(
        [sys.executable, "-u", "-m", "experiments.compare_methods", "--name", "main",
         "--methods", "current", "B_discrete", "C_cluster"],
        stdout=out, stderr=subprocess.STDOUT,
    )
print("done, exit code", code, flush=True)
