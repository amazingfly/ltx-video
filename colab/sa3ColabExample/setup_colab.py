#!/usr/bin/env python3
"""Install the reproducible SA3 runtime into the active Colab VM."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REMOTE_ROOT = Path("/content/sa3")
REMOTE_COLAB_DIR = REMOTE_ROOT / "colab"
OUTPUT_DIR = Path("/content/outputs")
HF_TOKEN_PATH = REMOTE_ROOT / "hf_token"
REQUIREMENTS_CANDIDATES = (
    REMOTE_COLAB_DIR / "requirements.txt",
    Path("/content/requirements.txt"),
)


def find_requirements() -> Path:
    for path in REQUIREMENTS_CANDIDATES:
        if path.is_file():
            return path
    checked = ", ".join(str(path) for path in REQUIREMENTS_CANDIDATES)
    raise FileNotFoundError(f"requirements.txt not found; checked: {checked}")


def configure_hugging_face() -> str:
    """Persist an optional Colab secret without requiring it for the public mirror."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    source = "environment"
    if not token and HF_TOKEN_PATH.is_file():
        token = HF_TOKEN_PATH.read_text(encoding="utf-8").strip()
        HF_TOKEN_PATH.unlink(missing_ok=True)
        source = "uploaded local token"
    if not token:
        try:
            from google.colab import userdata  # type: ignore

            token = userdata.get("HF_TOKEN")
            source = "Colab secret"
        except Exception:
            token = None

    if not token:
        return "not configured (public mirror remains usable)"

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    os.environ["HF_TOKEN"] = token
    return f"configured from {source}"


def main() -> int:
    REMOTE_COLAB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    requirements = find_requirements()

    print(f"Python: {sys.version}")
    print(f"Installing: {requirements}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--prefer-binary",
            "--no-cache-dir",
            "-r",
            str(requirements),
        ],
        check=True,
    )

    # Colab preinstalls torchvision. Replacing Torch can leave its compiled ops
    # ABI-incompatible, and Transformers imports it while loading T5Gemma even
    # though SA3 does not use vision features.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"],
        check=True,
    )

    config_path = REMOTE_COLAB_DIR / "generation_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = config.get("model", {})
        if model.get("name") == "medium":
            import torch

            props = torch.cuda.get_device_properties(0)
            allow_turing = bool(model.get("allow_turing_sdpa_fallback", False))
            if props.major >= 8:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-cache-dir",
                        "--no-deps",
                        (
                            "https://github.com/Dao-AILab/flash-attention/releases/"
                            "download/v2.8.3/"
                            "flash_attn-2.8.3%2Bcu12torch2.7cxx11abiFALSE-"
                            "cp312-cp312-linux_x86_64.whl"
                        ),
                    ],
                    check=True,
                )
            elif not allow_turing:
                raise RuntimeError(
                    "Medium requires compute capability >= 8.0 unless "
                    "allow_turing_sdpa_fallback is enabled"
                )

    token_status = configure_hugging_face()
    result = {
        "python": sys.version,
        "requirements": str(requirements),
        "hf_token": token_status,
        "output_dir": str(OUTPUT_DIR),
        "restart_kernel_required": True,
    }
    (OUTPUT_DIR / "setup.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print("Setup complete. Restart the Colab kernel before probing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
