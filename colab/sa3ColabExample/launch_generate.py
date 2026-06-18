#!/usr/bin/env python3
"""Launch SA3 generation independently of the Colab CLI WebSocket."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


REMOTE_COLAB_DIR = Path("/content/sa3/colab")
OUTPUT_DIR = Path("/content/outputs")
RUN_GENERATE = REMOTE_COLAB_DIR / "run_generate.py"
LAUNCH_GENERATE = REMOTE_COLAB_DIR / "launch_generate.py"
CONSOLE_LOG = OUTPUT_DIR / "generation_console.log"
EXIT_REPORT = OUTPUT_DIR / "generation_exit.json"
WORKER_REPORT = OUTPUT_DIR / "generation_worker.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_worker() -> int:
    started_at = utc_now()
    return_code = 1
    try:
        with CONSOLE_LOG.open("ab", buffering=0) as console:
            completed = subprocess.run(
                [sys.executable, str(RUN_GENERATE)],
                cwd="/content",
                stdin=subprocess.DEVNULL,
                stdout=console,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
            )
            return_code = completed.returncode
    except Exception:
        with CONSOLE_LOG.open("a", encoding="utf-8") as console:
            traceback.print_exc(file=console)
    finally:
        write_json_atomic(
            EXIT_REPORT,
            {
                "started_at": started_at,
                "finished_at": utc_now(),
                "return_code": return_code,
            },
        )
    return return_code


def launch() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RUN_GENERATE.is_file():
        raise FileNotFoundError(f"Remote generator not found: {RUN_GENERATE}")
    for path in (CONSOLE_LOG, EXIT_REPORT, WORKER_REPORT):
        path.unlink(missing_ok=True)

    worker = subprocess.Popen(
        [sys.executable, str(LAUNCH_GENERATE), "--worker"],
        cwd="/content",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        env=os.environ.copy(),
    )
    write_json_atomic(
        WORKER_REPORT,
        {
            "launched_at": utc_now(),
            "pid": worker.pid,
            "script": str(RUN_GENERATE),
            "console_log": str(CONSOLE_LOG),
            "exit_report": str(EXIT_REPORT),
        },
    )
    print(f"Detached Colab generation worker started with PID {worker.pid}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args, _ = parser.parse_known_args()
    return run_worker() if args.worker else launch()


if __name__ == "__main__":
    raise SystemExit(main())
