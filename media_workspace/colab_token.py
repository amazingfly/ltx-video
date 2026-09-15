#!/usr/bin/env python3
"""Refresh a Colab session's expiring runtime-proxy token in local state."""

from __future__ import annotations

import argparse


def refresh_session_state(state, session_name: str) -> tuple[bool, int]:
    session = state.store.get(session_name)
    if session is None:
        raise RuntimeError(f"Colab session {session_name!r} is not in local state")
    assignment = next(
        (
            item
            for item in state.client.list_assignments()
            if item.endpoint == session.endpoint
        ),
        None,
    )
    if assignment is None:
        raise RuntimeError(f"Colab session {session_name!r} is no longer assigned")
    proxy = assignment.runtime_proxy_info
    changed = session.token != proxy.token or session.url != proxy.url
    session.token = proxy.token
    session.url = proxy.url
    state.store.add(session)
    return changed, int(proxy.token_expires_in_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    from colab_cli.common import state

    changed, expires_in = refresh_session_state(state, args.session)
    action = "updated" if changed else "confirmed"
    print(
        f"[colab-token] {action} runtime file token for {args.session}; "
        f"expires in {expires_in}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
