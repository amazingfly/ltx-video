#!/usr/bin/env python3
"""Probe the Colab GPU runtime and selected SA3 model without downloading weights."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(
    os.environ.get("SA3_CONFIG", "/content/sa3/colab/generation_config.json")
)
OUTPUT_DIR = Path(os.environ.get("SA3_OUTPUT_DIR", "/content/outputs"))
REQUIRED_IMPORTS = (
    "torch",
    "torchaudio",
    "stable_audio_3",
    "transformers",
    "huggingface_hub",
    "soundfile",
)


def distribution_version(import_name: str) -> str:
    package_name = {
        "stable_audio_3": "stable-audio-3",
        "huggingface_hub": "huggingface-hub",
    }.get(import_name, import_name)
    return importlib.metadata.version(package_name)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = config.get("model", {})
    if model.get("name") not in {"small-music", "medium"}:
        raise ValueError("model.name must be 'small-music' or 'medium'")
    if not model.get("repo"):
        raise ValueError("model.repo is required")
    return config


def main() -> int:
    config = load_config()
    model_config = config["model"]
    model_name = model_config["name"]
    model_repo = model_config["repo"]
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "config_path": str(CONFIG_PATH),
        "model_name": model_name,
        "model_repo": model_repo,
        "imports": {},
        "checks": {},
    }
    errors: list[str] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_test = OUTPUT_DIR / ".write-test"
    write_test.write_text("ok\n", encoding="utf-8")
    write_test.unlink()
    report["checks"]["output_path"] = f"writable: {OUTPUT_DIR}"

    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
            report["imports"][name] = distribution_version(name)
        except Exception as exc:
            message = f"{name}: {exc.__class__.__name__}: {exc}"
            report["imports"][name] = message
            errors.append(message)

    try:
        from transformers import T5GemmaEncoderModel  # noqa: F401

        report["checks"]["t5gemma_import"] = "ok"
    except Exception as exc:
        message = f"T5GemmaEncoderModel: {exc.__class__.__name__}: {exc}"
        report["checks"]["t5gemma_import"] = message
        errors.append(message)

    if model_name == "medium":
        report["checks"]["medium_attention_backend"] = "pending GPU check"

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "compiled_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
        if not torch.cuda.is_available():
            errors.append("CUDA is not available; a Colab GPU runtime is required")
        else:
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gib": round(props.total_memory / 1024**3, 2),
            }
            if model_name == "medium":
                if props.major >= 8:
                    try:
                        import flash_attn
                        from flash_attn import flash_attn_func  # noqa: F401

                        report["imports"]["flash_attn"] = flash_attn.__version__
                        report["checks"]["medium_attention_backend"] = "flash-attn 2"
                    except Exception as exc:
                        message = f"flash_attn: {exc.__class__.__name__}: {exc}"
                        report["checks"]["medium_attention_backend"] = message
                        errors.append(message)
                elif model_config.get("allow_turing_sdpa_fallback", False):
                    report["checks"]["medium_attention_backend"] = (
                        "PyTorch SDPA plus chunked-halo sliding-window fallback"
                    )
                else:
                    errors.append(
                        "Medium on Turing requires allow_turing_sdpa_fallback=true"
                    )
            allocation = torch.empty((256, 1024), device="cuda")
            report["checks"]["cuda_allocation"] = {
                "device": str(allocation.device),
                "shape": list(allocation.shape),
            }
            del allocation
    except Exception as exc:
        errors.append(f"torch/CUDA probe: {exc.__class__.__name__}: {exc}")

    try:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().model_info(model_repo, files_metadata=True)
        siblings = {item.rfilename: item for item in (info.siblings or [])}
        required_files = ("model_config.json", "model.safetensors")
        missing = [name for name in required_files if name not in siblings]
        if missing:
            raise FileNotFoundError(f"Missing model files: {missing}")

        config_path = hf_hub_download(model_repo, "model_config.json")
        checkpoint_size = siblings["model.safetensors"].size
        report["model"] = {
            "available": True,
            "private": info.private,
            "gated": info.gated,
            "revision": info.sha,
            "config_path": config_path,
            "checkpoint_size_bytes": checkpoint_size,
            "checkpoint_size_gib": (
                round(checkpoint_size / 1024**3, 3) if checkpoint_size else None
            ),
        }
    except Exception as exc:
        errors.append(f"model availability: {exc.__class__.__name__}: {exc}")
        report["model"] = {
            "available": False,
            "error": f"{exc.__class__.__name__}: {exc}",
        }

    report["ok"] = not errors
    report["errors"] = errors
    report_path = OUTPUT_DIR / "probe.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Probe report: {report_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
