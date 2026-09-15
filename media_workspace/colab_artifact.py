#!/usr/bin/env python3
"""Recover a Colab artifact through the kernel when the Contents API is unavailable."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from colab_cli.runtime import ColabRuntime


def stream_text(outputs: list[dict]) -> str:
    chunks: list[str] = []
    for output in outputs:
        if output.get("output_type") == "stream":
            chunks.append(output.get("text", ""))
        elif output.get("output_type") == "error":
            traceback = "\n".join(output.get("traceback", []))
            raise RuntimeError(traceback or output.get("evalue", "remote execution failed"))
    return "".join(chunks).strip()


def execute(runtime: ColabRuntime, code: str, timeout: float) -> str:
    return stream_text(runtime.execute_code(code, timeout=timeout))


def main() -> int:
    from colab_cli.common import state
    from colab_cli.runtime import ColabRuntime
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_path")
    parser.add_argument("local_path", type=Path)
    parser.add_argument("--session", default="lqxl-sdxl-v1")
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    name = state.resolve_session(args.session)
    session = state.store.get(name)
    if not session:
        raise SystemExit(f"Colab session not found: {name}")

    assignment = next(
        (
            item
            for item in state.client.list_assignments()
            if item.endpoint == session.endpoint
        ),
        None,
    )
    if assignment is None:
        raise SystemExit(f"Colab assignment is no longer active: {session.endpoint}")
    session.url = assignment.runtime_proxy_info.url
    session.token = assignment.runtime_proxy_info.token
    state.store.add(session)

    def on_kernel_started(kernel_id: str) -> None:
        session.kernel_id = kernel_id
        state.store.add(session)

    def on_session_started(session_id: str) -> None:
        session.session_id = session_id
        state.store.add(session)

    runtime = ColabRuntime(
        session.url,
        session.token,
        kernel_id=session.kernel_id,
        session_id=session.session_id,
        on_kernel_started=on_kernel_started,
        on_session_started=on_session_started,
    )
    part_path = args.local_path.with_name(args.local_path.name + ".part")
    args.local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        remote_literal = json.dumps(args.remote_path)
        metadata = json.loads(
            execute(
                runtime,
                "import hashlib, json, os\n"
                f"p = {remote_literal}\n"
                "h = hashlib.sha256()\n"
                "with open(p, 'rb') as f:\n"
                "    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):\n"
                "        h.update(block)\n"
                "print(json.dumps({'size': os.path.getsize(p), 'sha256': h.hexdigest()}))",
                args.timeout,
            )
        )

        expected_size = int(metadata["size"])
        expected_sha = metadata["sha256"]
        offset = part_path.stat().st_size if part_path.exists() else 0
        if offset > expected_size:
            part_path.unlink()
            offset = 0

        chunk_size = args.chunk_mib * 1024 * 1024
        mode = "ab" if offset else "wb"
        with part_path.open(mode) as destination:
            while offset < expected_size:
                length = min(chunk_size, expected_size - offset)
                payload = execute(
                    runtime,
                    "import base64\n"
                    f"p = {remote_literal}\n"
                    f"with open(p, 'rb') as f:\n"
                    f"    f.seek({offset})\n"
                    f"    data = f.read({length})\n"
                    "print(base64.b64encode(data).decode('ascii'))",
                    args.timeout,
                )
                decoded = base64.b64decode(payload, validate=True)
                if len(decoded) != length:
                    raise RuntimeError(
                        f"short chunk at byte {offset}: expected {length}, got {len(decoded)}"
                    )
                destination.write(decoded)
                destination.flush()
                offset += length
                print(f"{args.local_path.name}: {offset}/{expected_size} bytes", flush=True)

        local_hash = hashlib.sha256()
        with part_path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                local_hash.update(block)
        digest = local_hash.hexdigest()
        if digest != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch: expected {expected_sha}, got {digest}")
        part_path.replace(args.local_path)
        print(f"Recovered {args.local_path} ({expected_sha})")
        return 0
    finally:
        runtime.stop()


if __name__ == "__main__":
    sys.exit(main())
