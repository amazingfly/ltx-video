#!/usr/bin/env python3
"""Generate and validate one or more FLAC files with one loaded SA3 model."""

from __future__ import annotations

import hashlib
import gc
import json
import math
import os
import sys
import threading
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(
    os.environ.get("SA3_CONFIG", "/content/sa3/colab/generation_config.json")
)
OUTPUT_DIR = Path(os.environ.get("SA3_OUTPUT_DIR", "/content/outputs"))


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = config.get("model", {})
    if model.get("name") not in {"small-music", "medium"}:
        raise ValueError("model.name must be 'small-music' or 'medium'")
    if not model.get("repo"):
        raise ValueError("model.repo is required")

    if "generations" in config:
        generations = config["generations"]
        if not isinstance(generations, list) or not generations:
            raise ValueError("generations must be a non-empty list")
    else:
        generation = config.get("generation")
        if not isinstance(generation, dict):
            raise ValueError("generation must be an object")
        generations = [generation]

    max_duration = float(model.get("max_duration_seconds", 0))
    output_names = set()
    for index, generation in enumerate(generations, start=1):
        if not isinstance(generation, dict):
            raise ValueError(f"generations[{index - 1}] must be an object")
        duration = float(generation.get("duration_seconds", 0))
        if not 0 < duration <= max_duration:
            raise ValueError(
                f"generation {index} duration_seconds must be between "
                f"0 and {max_duration:g}"
            )
        if not str(generation.get("prompt") or "").strip():
            raise ValueError(f"generation {index} prompt is required")
        if int(generation.get("steps", 8)) < 1:
            raise ValueError(f"generation {index} steps must be positive")
        output_name = str(generation.get("output_name", ""))
        if Path(output_name).name != output_name:
            raise ValueError(f"generation {index} output_name must be a filename")
        if Path(output_name).suffix.lower() != ".flac":
            raise ValueError(f"generation {index} output_name must end in .flac")
        if output_name in output_names:
            raise ValueError(f"Duplicate generation output_name: {output_name}")
        output_names.add(output_name)
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_model_config_resolver(
    model_name: str,
    model_repo: str,
    conditioning_repo: str,
    work_dir: Path,
) -> None:
    """Point the selected checkpoint and text conditioner at configured repos."""
    from huggingface_hub import hf_hub_download
    import stable_audio_3.model as model_module
    import stable_audio_3.model_configs as model_configs

    class PatchedModelConfig:
        def resolve(self) -> tuple[str, str]:
            config_path = hf_hub_download(model_repo, "model_config.json")
            checkpoint_path = hf_hub_download(model_repo, "model.safetensors")
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))

            conditioning = config.get("model", {}).get("conditioning", {})
            for item in conditioning.get("configs", []):
                item_config = item.get("config", {})
                if item_config.get("repo_id"):
                    item_config["repo_id"] = conditioning_repo

            patched_path = work_dir / f"{model_name}-patched-model_config.json"
            patched_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            return str(patched_path), checkpoint_path

    patched = PatchedModelConfig()
    model_configs.all_models[model_name] = patched
    model_module.all_models[model_name] = patched


def install_streaming_medium_loader() -> None:
    """Build in FP16 and stream weights to avoid medium-model host RAM spikes."""
    import torch
    from safetensors import safe_open
    from stable_audio_3.factory import create_diffusion_cond_from_config
    import stable_audio_3.model as model_module

    def remap_key(key: str, model_state: dict[str, torch.Tensor]) -> str:
        if key in model_state:
            return key
        parts = key.split(".")
        for index in range(1, len(parts)):
            candidate = ".".join(parts[:index] + parts[index + 1 :])
            if candidate in model_state:
                return candidate
        return key

    def streaming_load_diffusion_cond(
        model_config: dict[str, Any],
        checkpoint_path: str,
        device: str = "cuda",
        model_half: bool = False,
    ):
        if not model_half:
            raise RuntimeError("The low-memory Medium loader requires model_half=true")

        print("Constructing Medium model in FP16 on CPU", flush=True)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            model = create_diffusion_cond_from_config(model_config)
        finally:
            torch.set_default_dtype(previous_dtype)

        model.eval().requires_grad_(False)
        model.to(torch.float16)

        print(f"Moving empty model to {device}", flush=True)
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache()

        model_state = model.state_dict()
        loaded = 0
        skipped = 0
        print("Streaming checkpoint tensors directly into GPU parameters", flush=True)
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            for index, source_key in enumerate(keys, start=1):
                target_key = remap_key(source_key, model_state)
                target = model_state.get(target_key)
                source = checkpoint.get_tensor(source_key)
                if target is None or source.shape != target.shape:
                    skipped += 1
                else:
                    with torch.no_grad():
                        target.copy_(source, non_blocking=False)
                    loaded += 1
                del source
                if index % 100 == 0 or index == len(keys):
                    print(
                        f"Loaded {index}/{len(keys)} checkpoint tensors",
                        flush=True,
                    )

        del model_state
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"Streaming load complete: loaded={loaded}, skipped={skipped}", flush=True
        )
        if skipped:
            raise RuntimeError(f"Streaming checkpoint load skipped {skipped} tensors")
        return model

    model_module.load_diffusion_cond = streaming_load_diffusion_cond


def install_sequential_chunked_decoder(
    model,
    *,
    chunk_size: int = 64,
    overlap: int = 32,
) -> None:
    """Decode one chunk at a time so long tracks do not retain every chunk in VRAM."""
    import torch

    if chunk_size < 1:
        raise ValueError("decode chunk_size must be positive")
    if overlap < 0:
        raise ValueError("decode overlap cannot be negative")

    pretransform = model.same
    autoencoder = pretransform.model
    original_decode_audio = autoencoder.decode_audio

    def decode_audio_sequential(
        self,
        latents,
        chunked=False,
        overlap=overlap,
        chunk_size=chunk_size,
        **kwargs,
    ):
        if not chunked:
            return original_decode_audio(
                latents,
                chunked=False,
                overlap=overlap,
                chunk_size=chunk_size,
                **kwargs,
            )

        chunk_size = min(int(chunk_size), 64)
        overlap = min(int(overlap), chunk_size // 2)
        if latents.shape[-1] < chunk_size:
            return self.decode(latents, **kwargs)

        samples_per_latent = int(self.downsampling_ratio)
        hop_latents = chunk_size - overlap
        total_latents = latents.shape[-1]
        chunk_starts = list(range(0, total_latents - chunk_size + 1, hop_latents))
        if chunk_starts[-1] != total_latents - chunk_size:
            chunk_starts.append(total_latents - chunk_size)

        total_samples = total_latents * samples_per_latent
        chunk_size_samples = chunk_size * samples_per_latent
        half_overlap_samples = (overlap // 2) * samples_per_latent
        output = None
        num_chunks = len(chunk_starts)
        print(
            f"Diffusion complete; decoding audio in {num_chunks} sequential "
            f"chunks (chunk_size={chunk_size}, overlap={overlap})",
            flush=True,
        )

        for index, start_latent in enumerate(chunk_starts, start=1):
            chunk = self.decode(
                latents[..., start_latent : start_latent + chunk_size],
                **kwargs,
            )
            if output is None:
                output = latents.new_zeros(
                    *chunk.shape[:-1],
                    total_samples,
                )
            is_first = index == 1
            is_last = index == num_chunks
            out_start = (
                total_samples - chunk_size_samples
                if is_last
                else start_latent * samples_per_latent
            )
            left = 0 if is_first else half_overlap_samples
            right = (
                chunk_size_samples
                if is_last
                else chunk_size_samples - half_overlap_samples
            )
            output[..., out_start + left : out_start + right] = chunk[..., left:right]
            del chunk
            if index == 1 or index % 10 == 0 or is_last:
                print(
                    f"Audio decode chunk {index}/{num_chunks}",
                    flush=True,
                )
            torch.cuda.empty_cache()

        print("Audio decode complete", flush=True)
        return output

    autoencoder.decode_audio = types.MethodType(
        decode_audio_sequential,
        autoencoder,
    )


def validate_audio(path: Path, expected_duration: float) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise RuntimeError(f"Invalid audio metadata: {info}")
    if info.format != "FLAC":
        raise RuntimeError(f"Expected FLAC output, got {info.format}")

    actual_duration = info.frames / info.samplerate
    if abs(actual_duration - expected_duration) > 0.1:
        raise RuntimeError(
            f"Unexpected duration {actual_duration:.3f}s; expected {expected_duration:.3f}s"
        )

    peak = 0.0
    rms_sum = 0.0
    sample_count = 0
    with sf.SoundFile(str(path)) as audio_file:
        while True:
            block = audio_file.read(65536, dtype="float32", always_2d=True)
            if not len(block):
                break
            if not np.isfinite(block).all():
                raise RuntimeError("Generated audio contains NaN or infinite samples")
            peak = max(peak, float(np.max(np.abs(block))))
            rms_sum += float(np.square(block, dtype=np.float64).sum())
            sample_count += block.size

    rms = math.sqrt(rms_sum / sample_count) if sample_count else 0.0
    if peak <= 1e-5 or rms <= 1e-6:
        raise RuntimeError(
            f"Generated audio is effectively silent: peak={peak}, rms={rms}"
        )

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_seconds": round(actual_duration, 6),
        "peak": round(peak, 8),
        "rms": round(rms, 8),
        "sha256": sha256_file(path),
    }


def write_item_report(output_name: str, result: dict[str, Any]) -> Path:
    report_path = OUTPUT_DIR / Path(output_name).with_suffix(".json")
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    return report_path


def start_memory_monitor():
    import psutil
    import torch

    process = psutil.Process(os.getpid())
    stop = threading.Event()
    stats = {
        "peak_process_rss_bytes": process.memory_info().rss,
        "min_system_available_bytes": psutil.virtual_memory().available,
        "samples": 0,
    }

    def monitor() -> None:
        while not stop.wait(0.25):
            stats["peak_process_rss_bytes"] = max(
                stats["peak_process_rss_bytes"], process.memory_info().rss
            )
            stats["min_system_available_bytes"] = min(
                stats["min_system_available_bytes"],
                psutil.virtual_memory().available,
            )
            stats["samples"] += 1

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def finish() -> dict[str, Any]:
        stop.set()
        thread.join()
        stats["peak_process_rss_bytes"] = max(
            stats["peak_process_rss_bytes"], process.memory_info().rss
        )
        stats["min_system_available_bytes"] = min(
            stats["min_system_available_bytes"], psutil.virtual_memory().available
        )
        return {
            "peak_process_rss_gib": round(stats["peak_process_rss_bytes"] / 1024**3, 3),
            "min_system_available_gib": round(
                stats["min_system_available_bytes"] / 1024**3, 3
            ),
            "peak_cuda_memory_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "samples": stats["samples"],
        }

    return finish


def main() -> int:
    import torch
    import soundfile as sf
    from stable_audio_3 import StableAudioModel

    config = load_config()
    model_config = config["model"]
    queue_mode = "generations" in config
    generations = config["generations"] if queue_mode else [config["generation"]]
    model_name = model_config["name"]
    model_repo = model_config["repo"]
    conditioning_repo = model_config.get("conditioning_repo", model_repo)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; run this script in a Colab GPU session"
        )
    props = torch.cuda.get_device_properties(0)
    allow_turing_fallback = bool(model_config.get("allow_turing_sdpa_fallback", False))
    if model_name == "medium" and props.major < 8 and not allow_turing_fallback:
        raise RuntimeError("Medium on Turing requires allow_turing_sdpa_fallback=true")
    attention_backend = "default"
    if model_name == "medium" and props.major < 8 and allow_turing_fallback:
        import stable_audio_3.models.transformer as transformer_module

        # T4 cannot compile PyTorch's flex-attention Triton kernels. Skip the
        # noisy failed autotune and use SA3's chunked-halo SDPA path directly.
        transformer_module.flex_attention_compiled = None
        attention_backend = "chunked-halo-sdpa"
    if model_name == "medium" and props.major >= 8:
        import flash_attn  # noqa: F401

        attention_backend = "flash-attn-2"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = OUTPUT_DIR / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "generation.json"
    error_path = OUTPUT_DIR / "generation_error.log"
    error_path.unlink(missing_ok=True)

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    patch_model_config_resolver(
        model_name,
        model_repo,
        conditioning_repo,
        work_dir,
    )
    if model_name == "medium" and model_config.get("low_memory_load", True):
        install_streaming_medium_loader()
    finish_memory_monitor = start_memory_monitor()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name} from {model_repo}")
    print(f"Queue items: {len(generations)}")
    print("Loading model once for the complete queue", flush=True)
    model_load_started = time.monotonic()
    model = StableAudioModel.from_pretrained(
        model_name,
        device="cuda",
        model_half=bool(model_config.get("model_half", True)),
    )
    if model_name == "medium" and model_config.get("chunked_decode", True):
        install_sequential_chunked_decoder(
            model,
            chunk_size=int(model_config.get("decode_chunk_size", 64)),
            overlap=int(model_config.get("decode_overlap", 32)),
        )
    model_load_seconds = time.monotonic() - model_load_started
    sample_rate = int(model.model.sample_rate)
    model_sample_size = int(model.model_config.get("sample_size", 16_777_216))
    results = []
    error_messages = []

    for index, generation in enumerate(generations, start=1):
        item_started = time.monotonic()
        duration_seconds = float(generation["duration_seconds"])
        steps = int(generation.get("steps", 8))
        cfg_scale = float(generation.get("cfg_scale", 1.0))
        apg_scale = float(generation.get("apg_scale", 1.0))
        duration_padding = float(generation.get("duration_padding_seconds", 6.0))
        seed = int(generation.get("seed", -1))
        output_path = OUTPUT_DIR / generation["output_name"]
        requested_sample_size = int(
            (duration_seconds + duration_padding + 12.0) * sample_rate
        )
        sample_size = min(model_sample_size, requested_sample_size)
        torch.cuda.reset_peak_memory_stats()
        print(
            f"Queue item {index}/{len(generations)}: "
            f"{generation['output_name']}, {duration_seconds:g}s, "
            f"steps={steps}, cfg={cfg_scale}, apg={apg_scale}, seed={seed}",
            flush=True,
        )

        try:
            if output_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite remote output: {output_path}"
                )
            audio = model.generate(
                prompt=generation["prompt"],
                negative_prompt=generation.get("negative_prompt"),
                duration=duration_seconds,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                batch_size=1,
                chunked_decode=bool(model_config.get("chunked_decode", True)),
                apg_scale=apg_scale,
                duration_padding_sec=duration_padding,
                sample_size=sample_size,
                sampler_type=generation.get("sampler_type", "pingpong"),
            )
            print(
                f"Queue item {index}/{len(generations)} decode returned; "
                "moving audio to CPU",
                flush=True,
            )
            audio = audio.detach().to(dtype=torch.float32, device="cpu")
            if audio.dim() == 3:
                audio = audio[0]
            if audio.dim() != 2:
                raise RuntimeError(
                    f"Unexpected audio tensor shape: {tuple(audio.shape)}"
                )

            target_samples = round(duration_seconds * sample_rate)
            audio = audio[:, :target_samples].clamp(-1, 1).contiguous()
            sf.write(
                str(output_path),
                audio.transpose(0, 1).numpy(),
                sample_rate,
                format="FLAC",
                subtype=generation.get("flac_subtype", "PCM_24"),
            )
            print(
                f"Queue item {index}/{len(generations)} FLAC written; validating audio",
                flush=True,
            )
            validation = validate_audio(output_path, duration_seconds)
            result = {
                "ok": True,
                "index": index,
                "id": generation.get("id") or f"item-{index}",
                "elapsed_seconds": round(time.monotonic() - item_started, 3),
                "prompt": generation["prompt"],
                "negative_prompt": generation.get("negative_prompt"),
                "settings": {
                    "duration_seconds": duration_seconds,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "apg_scale": apg_scale,
                    "seed": seed,
                    "sampler_type": generation.get("sampler_type", "pingpong"),
                    "sample_size": sample_size,
                    "flac_subtype": generation.get("flac_subtype", "PCM_24"),
                    "peak_cuda_memory_gib": round(
                        torch.cuda.max_memory_allocated() / 1024**3, 3
                    ),
                },
                "audio": validation,
            }
            print(f"Generated audio: {output_path}", flush=True)
        except Exception:
            error = traceback.format_exc()
            error_messages.append(
                f"Queue item {index} ({generation['output_name']}):\n{error}"
            )
            result = {
                "ok": False,
                "index": index,
                "id": generation.get("id") or f"item-{index}",
                "elapsed_seconds": round(time.monotonic() - item_started, 3),
                "prompt": generation["prompt"],
                "output_name": generation["output_name"],
                "error": error,
            }
            print(error, file=sys.stderr, flush=True)
        finally:
            if "audio" in locals():
                del audio
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result)
        item_report_path = write_item_report(
            generation["output_name"],
            result,
        )
        print(f"Queue item report: {item_report_path}", flush=True)

    elapsed = time.monotonic() - started
    memory = finish_memory_monitor()
    common = {
        "ok": all(result["ok"] for result in results),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "model_load_count": 1,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config_path": str(CONFIG_PATH),
        "model_name": model_name,
        "model_repo": model_repo,
        "conditioning_repo": conditioning_repo,
        "model_settings": {
            "model_half": bool(model_config.get("model_half", True)),
            "low_memory_load": bool(model_config.get("low_memory_load", True)),
            "chunked_decode": bool(model_config.get("chunked_decode", True)),
            "decode_chunk_size": int(model_config.get("decode_chunk_size", 64)),
            "decode_overlap": int(model_config.get("decode_overlap", 32)),
            "allow_turing_sdpa_fallback": allow_turing_fallback,
            "attention_backend": attention_backend,
        },
        "memory": memory,
    }

    if queue_mode:
        manifest = {
            **common,
            "queue": True,
            "count": len(results),
            "succeeded": sum(result["ok"] for result in results),
            "failed": sum(not result["ok"] for result in results),
            "results": results,
        }
    else:
        result = results[0]
        manifest = {
            **common,
            "prompt": result["prompt"],
            "negative_prompt": result.get("negative_prompt"),
            "settings": {
                **result.get("settings", {}),
                **common["model_settings"],
            },
            "audio": result.get("audio"),
        }
        if not result["ok"]:
            manifest["error"] = result["error"]

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if error_messages:
        error_path.write_text("\n\n".join(error_messages), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not queue_mode and not manifest["ok"]:
        raise RuntimeError("Generation failed; see generation_error.log")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        error = traceback.format_exc()
        (OUTPUT_DIR / "generation_error.log").write_text(error, encoding="utf-8")
        print(error, file=sys.stderr)
        raise
