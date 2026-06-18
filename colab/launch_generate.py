#!/usr/bin/env python3
"""Launch generation independently of the Colab CLI WebSocket."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/content/ltx_music_video")
OUTPUTS = Path("/content/outputs")
WORKER = ROOT / "colab" / "run_generate.py"
LAUNCHER = ROOT / "colab" / "launch_generate.py"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def worker_main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    console = OUTPUTS / "generation_console.log"
    with console.open("ab", buffering=0) as handle:
        completed = subprocess.run(
            [sys.executable, "-u", str(WORKER)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    report = {
        "return_code": completed.returncode,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = OUTPUTS / ".generation_exit.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUTS / "generation_exit.json")
    return completed.returncode


def main() -> int:
    if "--worker" in sys.argv:
        return worker_main()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for name in (
        "generation_console.log",
        "generation_error.log",
        "generation_exit.json",
    ):
        (OUTPUTS / name).unlink(missing_ok=True)
    worker = subprocess.Popen(
        [sys.executable, str(LAUNCHER), "--worker"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    report = {
        "pid": worker.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUTS / "generation_worker.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Detached LTX worker started with PID {worker.pid}")
    return 0


if __name__ == "__main__":
    main()
