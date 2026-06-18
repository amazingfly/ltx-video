#!/usr/bin/env python3
"""Install the pinned official LTX-Video runtime in the active Colab VM."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REMOTE_ROOT = Path("/content/ltx_music_video")
UPSTREAM = REMOTE_ROOT / "LTX-Video"
OUTPUTS = Path("/content/outputs")
COMMIT = "4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    requirements = REMOTE_ROOT / "colab" / "requirements.txt"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(requirements),
        ]
    )
    if not (UPSTREAM / ".git").is_dir():
        run(
            [
                "git",
                "clone",
                "https://github.com/Lightricks/LTX-Video.git",
                str(UPSTREAM),
            ]
        )
    run(["git", "fetch", "--depth", "1", "origin", COMMIT], cwd=UPSTREAM)
    run(["git", "checkout", "--detach", COMMIT], cwd=UPSTREAM)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "-e",
            str(UPSTREAM),
        ]
    )
    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "upstream_commit": COMMIT,
    }
    (OUTPUTS / "setup.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("LTX-Video setup complete.", flush=True)
    return 0


if __name__ == "__main__":
    main()
