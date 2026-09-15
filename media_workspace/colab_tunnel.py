#!/usr/bin/env python3
"""Send the Colab CLI tunnel keep-alive ping for an active session."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path




def force_ipv4_dns() -> None:
    getaddrinfo = socket.getaddrinfo

    def ipv4_getaddrinfo(
        host: str | bytes | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[int, int, int, str, tuple[object, ...]]]:
        if family in (0, socket.AF_UNSPEC):
            family = socket.AF_INET
        return getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = ipv4_getaddrinfo


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def configure_colab_state(args: argparse.Namespace):
    from colab_cli.auth import AuthProvider
    from colab_cli.common import state
    state.auth_provider = AuthProvider(args.auth_provider)
    if args.config:
        state.config_path = str(args.config)
    return state


def ping_tunnel(args: argparse.Namespace, state=None) -> dict[str, object]:
    from requests.exceptions import ReadTimeout
    state = state or configure_colab_state(args)
    session = state.store.get(args.session)
    if session is None:
        return {
            "ok": False,
            "time": utc_now(),
            "session": args.session,
            "error": "session_not_found",
        }

    url = (
        f"{state.client.colab_domain.rstrip('/')}"
        f"/tun/m/{session.endpoint}/keep-alive/"
    )
    try:
        response = state.client.session.get(
            url,
            headers={"X-Colab-Tunnel": "Google"},
            params={"authuser": str(args.authuser)},
            timeout=args.request_timeout,
        )
    except ReadTimeout:
        # Colab's tunnel frontend records activity before forwarding the
        # keep-alive request to the VM, and that forwarded request often does
        # not answer. Upstream colab-cli treats this read timeout as success.
        return {
            "ok": True,
            "time": utc_now(),
            "session": args.session,
            "endpoint": session.endpoint,
            "read_timeout": True,
        }

    report = {
        "ok": response.ok,
        "time": utc_now(),
        "session": args.session,
        "endpoint": session.endpoint,
        "status_code": response.status_code,
        "reason": response.reason,
    }
    if not response.ok:
        report["body"] = truncate(response.text)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--authuser", type=int, default=0)
    parser.add_argument("--auth-provider", choices=["oauth2", "adc"], default="oauth2")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--ipv4", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ipv4:
        force_ipv4_dns()
    while True:
        report = ping_tunnel(args)
        print(json.dumps(report, sort_keys=True), flush=True)
        if not args.loop:
            return 0 if report.get("ok") else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
