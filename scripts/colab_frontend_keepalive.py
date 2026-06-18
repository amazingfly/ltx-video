#!/usr/bin/env python3
"""Keep a CLI-created Colab runtime attached through the web frontend."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.async_api import Page, async_playwright


FIREFOX_ROOT = Path.home() / "snap/firefox/common/.mozilla/firefox"
CHROMIUM = Path("/usr/bin/chromium-browser")


def find_cookie_database() -> Path:
    databases = list(FIREFOX_ROOT.glob("*/cookies.sqlite"))
    if not databases:
        raise FileNotFoundError(
            f"No Firefox cookie database found under {FIREFOX_ROOT}"
        )
    return max(databases, key=lambda path: path.stat().st_mtime)


def read_google_cookies(database: Path) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="colab-cookies-") as temporary:
        copied = Path(temporary) / "cookies.sqlite"
        shutil.copy2(database, copied)
        wal = database.with_name(f"{database.name}-wal")
        if wal.exists():
            shutil.copy2(wal, copied.with_name(f"{copied.name}-wal"))

        connection = sqlite3.connect(copied)
        rows = connection.execute(
            """
            SELECT host, name, value, path, isSecure, isHttpOnly, sameSite,
                   lastAccessed
            FROM moz_cookies
            WHERE originAttributes = ''
              AND (
                host LIKE '%google.com'
                OR host LIKE '%googleusercontent.com'
                OR host LIKE '%googleapis.com'
              )
            ORDER BY lastAccessed
            """
        ).fetchall()
        connection.close()

    cookies: dict[tuple[str, str, str], dict[str, object]] = {}
    same_site = {0: "None", 1: "Lax", 2: "Strict"}
    for host, name, value, path, secure, http_only, same, _ in rows:
        cookie_path = path or "/"
        cookies[(host, name, cookie_path)] = {
            "name": name,
            "value": value,
            "domain": host,
            "path": cookie_path,
            "secure": bool(secure),
            "httpOnly": bool(http_only),
            "sameSite": same_site.get(same, "Lax"),
        }
    return list(cookies.values())


def select_google_account(url: str, authuser: int) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query["authuser"] = str(authuser)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


async def connect_text(page: Page) -> str | None:
    return await page.evaluate(
        """
        () => document.querySelector('colab-connect-button')
          ?.shadowRoot?.querySelector('#connect')?.innerText ?? null
        """
    )


async def click_connect(page: Page) -> None:
    await page.evaluate(
        """
        () => document.querySelector('colab-connect-button')
          ?.shadowRoot?.querySelector('#connect')?.click()
        """
    )


async def wait_until_connected(page: Page) -> str | None:
    state = await connect_text(page)
    if state == "Connect":
        await click_connect(page)
    for _ in range(30):
        await page.wait_for_timeout(2_000)
        state = await connect_text(page)
        if state not in {"Connect", "Connecting"}:
            return state
    raise RuntimeError(f"Colab frontend did not connect; final state={state!r}")


async def keep_alive(
    url: str,
    database: Path,
    ready_file: Path | None,
    authuser: int,
) -> None:
    cookies = read_google_cookies(database)
    if not cookies:
        raise RuntimeError(f"No Google cookies found in {database}")
    if not CHROMIUM.exists():
        raise FileNotFoundError(f"Chromium was not found at {CHROMIUM}")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=str(CHROMIUM),
            headless=True,
            args=["--no-sandbox"],
        )
        context = await browser.new_context()
        await context.add_cookies(cookies)
        page = await context.new_page()
        await page.goto(
            select_google_account(url, authuser),
            wait_until="domcontentloaded",
            timeout=120_000,
        )
        await page.wait_for_timeout(10_000)

        body = await page.locator("body").inner_text()
        if "Sign in" in body:
            raise RuntimeError(
                f"Firefox is not signed into Google account index {authuser}"
            )
        state = await wait_until_connected(page)
        report = {
            "authuser": authuser,
            "cookie_database": str(database),
            "page_title": await page.title(),
            "connect_state": state,
        }
        print(f"Colab frontend keepalive ready: {json.dumps(report)}", flush=True)
        if ready_file:
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.write_text(json.dumps(report) + "\n", encoding="utf-8")

        try:
            while True:
                await page.wait_for_timeout(30_000)
                if page.is_closed():
                    raise RuntimeError("The Colab keepalive page closed unexpectedly")
                state = await connect_text(page)
                if state == "Connect":
                    await wait_until_connected(page)
                print(f"Colab frontend heartbeat: state={state!r}", flush=True)
        finally:
            await browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--cookie-db", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument(
        "--authuser",
        type=int,
        default=1,
        help="Google browser account index matching the Colab CLI login",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = args.cookie_db or find_cookie_database()
    asyncio.run(keep_alive(args.url, database, args.ready_file, args.authuser))


if __name__ == "__main__":
    main()
