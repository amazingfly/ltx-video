#!/usr/bin/env python3
"""Cycle the active Colab CLI token to the next configured account token."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


DEFAULT_CONFIG_DIR = Path.home() / ".config" / "colab-cli"
ACTIVE_TOKEN_NAME = "token.json"
FALLBACK_TOKEN_NAME = "t.json"
STATE_FILE_NAME = ".colab-token-cycle-state.json"


def token_candidates(config_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    preferred = [config_dir / FALLBACK_TOKEN_NAME]
    for path in preferred:
        if path.is_file():
            candidates.append(path.resolve())

    extras = sorted(
        path.resolve()
        for path in config_dir.glob("token.*.json")
        if path.is_file()
        and path.name != ACTIVE_TOKEN_NAME
        and not path.name.endswith(".backup.json")
    )
    for path in extras:
        if path not in candidates:
            candidates.append(path)

    return candidates


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(source.read_bytes())
    temporary.replace(destination)


def load_state(config_dir: Path) -> str | None:
    state_path = config_dir / STATE_FILE_NAME
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    current = payload.get("current_token")
    if isinstance(current, str) and current:
        return current
    return None


def save_state(config_dir: Path, current_token: str) -> None:
    state_path = config_dir / STATE_FILE_NAME
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    temporary.write_text(
        json.dumps({"current_token": current_token}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(state_path)


def cycle_token(config_dir: Path, dry_run: bool = False) -> Path:
    active = config_dir / ACTIVE_TOKEN_NAME
    if not active.is_file():
        raise FileNotFoundError(f"Missing active Colab token: {active}")

    candidates = token_candidates(config_dir)
    if not candidates:
        raise FileNotFoundError(f"No Colab token candidates found in {config_dir}")

    current_label = load_state(config_dir)
    current_index = next(
        (index for index, path in enumerate(candidates) if path.name == current_label),
        None,
    )
    if current_index is None:
        active_bytes = read_bytes(active)
        matched_index = next(
            (
                index
                for index, path in enumerate(candidates)
                if read_bytes(path) == active_bytes
            ),
            -1,
        )
        if matched_index >= 0:
            current_index = matched_index
            current_label = candidates[current_index].name
        else:
            current_index = -1
            current_label = "<unmatched active token>"

    next_index = (current_index + 1) % len(candidates)
    next_token = candidates[next_index]

    if dry_run:
        print(
            json.dumps(
                {
                    "config_dir": str(config_dir),
                    "current": current_label,
                    "next": next_token.name,
                    "candidates": [path.name for path in candidates],
                },
                indent=2,
            )
        )
        return next_token

    copy_atomic(next_token, active)
    save_state(config_dir, next_token.name)
    print(
        json.dumps(
            {
                "config_dir": str(config_dir),
                "current": current_label,
                "next": next_token.name,
                "written": str(active),
            },
            indent=2,
        )
    )
    return next_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Colab CLI config directory containing token.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which token would be activated without changing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cycle_token(args.config_dir.expanduser().resolve(), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
