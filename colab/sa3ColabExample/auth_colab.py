#!/usr/bin/env python3
"""Authenticate the installed Colab CLI and verify its required OAuth scope."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from importlib import resources
from pathlib import Path

from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH
from google_auth_oauthlib.flow import InstalledAppFlow


REQUIRED_SCOPE = "https://www.googleapis.com/auth/colaboratory"


def main() -> int:
    config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
    client_config = json.loads(config_resource.read_text())
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    credentials = flow.run_local_server(
        port=8200,
        prompt="consent select_account",
        access_type="offline",
        enable_granular_consent="false",
        include_granted_scopes="true",
        authorization_prompt_message=(
            "Open this official Colab OAuth URL and approve the complete permission set:\n{url}"
        ),
    )

    token_url = (
        "https://oauth2.googleapis.com/tokeninfo?access_token="
        + urllib.parse.quote(credentials.token)
    )
    with urllib.request.urlopen(token_url) as response:
        token_info = json.load(response)
    granted_scopes = set(token_info.get("scope", "").split())
    if REQUIRED_SCOPE not in granted_scopes:
        raise RuntimeError(
            "Google did not grant the required Colab scope. "
            "Choose an account with Google Colab access."
        )

    token_path = Path(TOKEN_CONFIG_PATH)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    token_path.chmod(0o600)
    print(f"Verified Colab OAuth token saved to {token_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
