#!/usr/bin/env python3
"""Print lightweight diagnostics from inside a Colab runtime."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    if completed.stderr:
        print(completed.stderr.rstrip(), flush=True)
    print(f"exit={completed.returncode}", flush=True)


def tail(path: Path, lines: int = 80) -> None:
    print(f"$ tail -{lines} {path}", flush=True)
    if not path.exists():
        print("missing", flush=True)
        return
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in data[-lines:]:
        print(line, flush=True)


def main() -> None:
    run(["ps", "-eo", "pid,ppid,stat,etime,%cpu,%mem,cmd"])
    run(["ls", "-lh", "/content/outputs"])
    tail(Path("/content/outputs/generation_console.log"))
    tail(Path("/content/outputs/generation_error.log"))
    run(["du", "-sh", "/content/ltx_music_video", "/content/outputs"])
    if Path("/usr/bin/nvidia-smi").exists():
        run(["nvidia-smi"])
    else:
        print("nvidia-smi missing", flush=True)
    print(f"pid={os.getpid()}", flush=True)


if __name__ == "__main__":
    main()
