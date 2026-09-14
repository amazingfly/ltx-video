#!/usr/bin/env python3
"""Generate a queue of image-to-video clips while loading LTX-Video once."""

from __future__ import annotations

import hashlib
import json
import os
import traceback
import gc
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.inference import (
    calculate_padding,
    prepare_conditioning,
)
from ltx_video.models.autoencoders.vae_encode import vae_decode
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    LTXMultiScalePipeline,
    LTXVideoPipeline,
    adain_filter_latent,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy


ROOT = Path("/content/ltx_music_video")
JOB_ROOT = ROOT / "job"
JOB_PATH = JOB_ROOT / "job.json"
CHUNKS_ROOT = ROOT / "chunks"
OUTPUTS = Path("/content/outputs")
PROMPT_CACHE = JOB_ROOT / "prompt_embeddings.pt"
PROMPT_CACHE_OUTPUT = OUTPUTS / "prompt_embeddings.pt"
PROMPT_CACHE_REPORT = OUTPUTS / "prompt_embeddings.json"


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def decode_latents_with_cpu_vae(
    base: LTXVideoPipeline,
    latents: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    vae_per_channel_normalize: bool,
    decode_timestep: float | list[float],
    decode_noise_scale: float | list[float] | None,
    tone_map_compression_ratio: float,
) -> torch.Tensor:
    vae_dtype = next(base.vae.parameters()).dtype
    latents = latents.detach().cpu().float()
    base.vae = base.vae.to(device="cpu", dtype=torch.float32)
    release_cuda_memory()
    try:
        if base.vae.decoder.timestep_conditioning:
            noise = torch.randn_like(latents)
            timestep_values = decode_timestep
            noise_scale_values = (
                decode_timestep if decode_noise_scale is None else decode_noise_scale
            )
            if not isinstance(timestep_values, list):
                timestep_values = [float(timestep_values)] * latents.shape[0]
            if not isinstance(noise_scale_values, list):
                noise_scale_values = [float(noise_scale_values)] * latents.shape[0]
            timestep_tensor = torch.tensor(timestep_values, device=latents.device)
            noise_scale_tensor = torch.tensor(
                noise_scale_values, device=latents.device
            )[:, None, None, None, None]
            latents = latents * (1 - noise_scale_tensor) + noise * noise_scale_tensor
        else:
            timestep_tensor = None

        latents = base.tone_map_latents(latents, tone_map_compression_ratio)
        images = vae_decode(
            latents,
            base.vae,
            is_video=True,
            vae_per_channel_normalize=vae_per_channel_normalize,
            timestep=timestep_tensor,
        )
        images = base.image_processor.postprocess(images, output_type="pt")
        if images.shape[3] != target_height or images.shape[4] != target_width:
            frame_count = images.shape[2]
            videos = rearrange(images, "b c f h w -> (b f) c h w")
            videos = F.interpolate(
                videos,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            images = rearrange(
                videos, "(b f) c h w -> b c f h w", f=frame_count
            )
        return images
    finally:
        base.vae = base.vae.to(device="cuda", dtype=vae_dtype)
        release_cuda_memory()


class T4MultiScalePipeline(LTXMultiScalePipeline):
    """Keep both denoising passes on GPU, then offload before VAE decode."""

    def __call__(
        self,
        downscale_factor: float,
        first_pass: dict[str, Any],
        second_pass: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_kwargs = kwargs.copy()
        original_output_type = kwargs["output_type"]
        original_width = kwargs["width"]
        original_height = kwargs["height"]
        execution_device = kwargs.get("device", "cuda")

        # A previous clip may have offloaded the transformer after decoding.
        self.video_pipeline.transformer = self.video_pipeline.transformer.to(
            execution_device
        )

        x_width = int(kwargs["width"] * downscale_factor)
        downscaled_width = x_width - (x_width % self.video_pipeline.vae_scale_factor)
        x_height = int(kwargs["height"] * downscale_factor)
        downscaled_height = x_height - (
            x_height % self.video_pipeline.vae_scale_factor
        )

        kwargs["output_type"] = "latent"
        kwargs["width"] = downscaled_width
        kwargs["height"] = downscaled_height
        kwargs["offload_to_cpu"] = False
        kwargs.update(**first_pass)
        result = self.video_pipeline(*args, **kwargs)
        latents = result.images

        upsampled_latents = self._upsample_latents(self.latent_upsampler, latents)
        upsampled_latents = adain_filter_latent(
            latents=upsampled_latents, reference_latents=latents
        )

        kwargs = original_kwargs
        kwargs["latents"] = upsampled_latents
        kwargs["output_type"] = original_output_type
        kwargs["width"] = downscaled_width * 2
        kwargs["height"] = downscaled_height * 2
        kwargs.update(**second_pass)

        result = self.video_pipeline(*args, **kwargs)
        if original_output_type != "latent":
            num_frames = result.images.shape[2]
            videos = rearrange(result.images, "b c f h w -> (b f) c h w")
            videos = F.interpolate(
                videos,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            result.images = rearrange(
                videos, "(b f) c h w -> b c f h w", f=num_frames
            )

        return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def skip_strategy(name: str) -> SkipLayerStrategy:
    options = {
        "attention_values": SkipLayerStrategy.AttentionValues,
        "attention_skip": SkipLayerStrategy.AttentionSkip,
        "residual": SkipLayerStrategy.Residual,
        "transformer_block": SkipLayerStrategy.TransformerBlock,
    }
    return options[name.lower()]


def save_prompt_cache(
    cache_key: str,
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache_data = {
        "version": 1,
        "cache_key": cache_key,
        "encoded": encoded,
    }
    temporary = PROMPT_CACHE_OUTPUT.with_suffix(".partial.pt")
    torch.save(cache_data, temporary)
    os.replace(temporary, PROMPT_CACHE_OUTPUT)
    atomic_json(
        PROMPT_CACHE_REPORT,
        {
            "cache_key": cache_key,
            "bytes": PROMPT_CACHE_OUTPUT.stat().st_size,
            "entries": len(encoded) - 1,
        },
    )


def encode_prompts(
    settings: dict[str, Any],
    clips: list[dict[str, Any]],
    cache_key: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if PROMPT_CACHE.is_file():
        try:
            cached = torch.load(
                PROMPT_CACHE,
                map_location="cpu",
                weights_only=True,
            )
            if cached.get("cache_key") != cache_key:
                raise ValueError("prompt cache key does not match this batch")
            encoded = cached["encoded"]
            required = {"__negative__", *(clip["id"] for clip in clips)}
            if not required.issubset(encoded):
                raise ValueError("prompt cache is missing required embeddings")
            encoded = {key: encoded[key] for key in required}
            save_prompt_cache(cache_key, encoded)
            print(
                f"Loaded cached prompt embeddings for {len(clips)} clip(s)",
                flush=True,
            )
            return encoded
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            print(f"Ignoring unusable prompt cache: {exc}", flush=True)

    text_encoder_repo = settings.get(
        "text_encoder_repo", "city96/t5-v1_1-xxl-encoder-bf16"
    )
    tokenizer_repo = settings.get("tokenizer_repo", text_encoder_repo)
    dtype = torch.bfloat16

    print(f"Loading text encoder: {text_encoder_repo}", flush=True)
    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_repo,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    text_encoder.eval()
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_repo)
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        negative = tokenizer(
            settings["negative_prompt"],
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        negative_mask = negative.attention_mask.to("cuda")
        negative_embeds = text_encoder(
            negative.input_ids.to("cuda"), attention_mask=negative_mask
        )[0]
        encoded["__negative__"] = (
            negative_embeds.cpu(),
            negative_mask.cpu(),
        )
        for index, clip in enumerate(clips, start=1):
            if index == 1 or index % 10 == 0 or index == len(clips):
                print(
                    f"Encoding prompt embeddings {index}/{len(clips)}",
                    flush=True,
                )
            inputs = tokenizer(
                clip["prompt"],
                padding="max_length",
                max_length=256,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            attention_mask = inputs.attention_mask.to("cuda")
            embeddings = text_encoder(
                inputs.input_ids.to("cuda"), attention_mask=attention_mask
            )[0]
            encoded[clip["id"]] = (
                embeddings.cpu(),
                attention_mask.cpu(),
            )
    del text_encoder
    release_cuda_memory()
    save_prompt_cache(cache_key, encoded)
    print(
        f"Saved prompt embedding cache for {len(clips)} clip(s)",
        flush=True,
    )
    return encoded


def load_pipeline(settings: dict[str, Any]):
    repo = settings["model_repo"]
    checkpoint = hf_hub_download(repo_id=repo, filename=settings["checkpoint"])
    upscaler = hf_hub_download(repo_id=repo, filename=settings["spatial_upscaler"])
    config_path = ROOT / "LTX-Video" / "configs" / "ltxv-2b-0.9.8-distilled.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dtype = torch.bfloat16

    with safe_open(checkpoint, framework="pt") as handle:
        metadata = handle.metadata() or {}
    checkpoint_config = json.loads(metadata.get("config", "{}"))

    print("Loading LTX transformer and VAE", flush=True)
    transformer = Transformer3DModel.from_pretrained(
        checkpoint,
        low_cpu_mem_usage=True,
    ).to(device="cuda", dtype=dtype)
    vae = CausalVideoAutoencoder.from_pretrained(checkpoint).to(
        device="cuda", dtype=dtype
    )
    scheduler = RectifiedFlowScheduler.from_pretrained(checkpoint)
    base = LTXVideoPipeline(
        transformer=transformer,
        patchifier=SymmetricPatchifier(patch_size=1),
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
        vae=vae,
        prompt_enhancer_image_caption_model=None,
        prompt_enhancer_image_caption_processor=None,
        prompt_enhancer_llm_model=None,
        prompt_enhancer_llm_tokenizer=None,
        allowed_inference_steps=checkpoint_config.get("allowed_inference_steps"),
    )
    # Scheduler arithmetic promotes first-pass latents to FP32. The upstream
    # loader keeps the upscaler in FP32, then the second pass casts as needed.
    latent_upsampler = LatentUpsampler.from_pretrained(upscaler).to("cuda")
    latent_upsampler.eval()
    print(
        "Model dtypes: "
        f"transformer={transformer.dtype}, vae={vae.dtype}, "
        f"upscaler={next(latent_upsampler.parameters()).dtype}",
        flush=True,
    )
    return base, T4MultiScalePipeline(base, latent_upsampler), config


def conditioning_correlation(
    generated_frame: np.ndarray,
    conditioning,
    padding: tuple[int, int, int, int],
) -> float:
    left, right, top, bottom = padding
    source = conditioning[0].media_item[0, :, 0]
    bottom_index = -bottom if bottom else source.shape[1]
    right_index = -right if right else source.shape[2]
    source = source[:, top:bottom_index, left:right_index]
    source = torch.nn.functional.interpolate(
        source.unsqueeze(0),
        size=generated_frame.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0]
    source = ((source.permute(1, 2, 0).cpu().float().numpy() + 1.0) / 2.0).clip(
        0.0, 1.0
    )
    generated = generated_frame.astype(np.float32) / 255.0
    source_gray = source.mean(axis=2).reshape(-1)
    generated_gray = generated.mean(axis=2).reshape(-1)
    if source_gray.std() < 0.03 or generated_gray.std() < 0.03:
        return 1.0 - float(np.abs(source_gray - generated_gray).mean())
    return float(np.corrcoef(source_gray, generated_gray)[0, 1])


def frame_motion_metrics(
    frames: np.ndarray,
    *,
    pixel_delta: int,
) -> dict[str, float | int]:
    grayscale = (
        frames[..., 0].astype(np.float32) * 0.299
        + frames[..., 1].astype(np.float32) * 0.587
        + frames[..., 2].astype(np.float32) * 0.114
    )
    differences = np.abs(np.diff(grayscale, axis=0))
    pair_mad = differences.mean(axis=(1, 2))
    pair_changed = (differences >= pixel_delta).mean(axis=(1, 2)) * 100.0
    return {
        "frame_count": int(len(frames)),
        "mean_pair_mad_0_255": float(pair_mad.mean()),
        "median_pair_mad_0_255": float(np.median(pair_mad)),
        "maximum_pair_mad_0_255": float(pair_mad.max()),
        "mean_changed_pixels_percent": float(pair_changed.mean()),
        "maximum_changed_pixels_percent": float(pair_changed.max()),
        "pixel_delta": pixel_delta,
    }


def generate_one(
    base,
    pipeline,
    pipeline_config: dict[str, Any],
    settings: dict[str, Any],
    clip: dict[str, Any],
    prompt_data: tuple[torch.Tensor, torch.Tensor],
    negative_prompt_data: tuple[torch.Tensor, torch.Tensor],
) -> tuple[Path, dict[str, Any]]:
    width = int(settings["width"])
    height = int(settings["height"])
    num_frames = int(settings["num_frames"])
    fps = int(settings["fps"])
    padded_height = ((height - 1) // 32 + 1) * 32
    padded_width = ((width - 1) // 32 + 1) * 32
    padded_frames = ((num_frames - 2) // 8 + 1) * 8 + 1
    padding = calculate_padding(height, width, padded_height, padded_width)
    image_path = JOB_ROOT / clip["image_path"]
    anchor_mode = settings.get("conditioning_anchor_frames", "start")
    if anchor_mode == "start":
        anchor_frames = [0]
        anchor_strengths = [1.0]
    elif anchor_mode == "start_middle_end":
        anchor_frames = [0, (num_frames - 1) // 2, num_frames - 1]
        anchor_strengths = [
            1.0,
            float(settings.get("middle_anchor_strength", 0.55)),
            1.0,
        ]
    else:
        raise ValueError(f"Unknown conditioning anchor mode: {anchor_mode}")
    conditioning = prepare_conditioning(
        conditioning_media_paths=[str(image_path)] * len(anchor_frames),
        conditioning_strengths=anchor_strengths,
        conditioning_start_frames=anchor_frames,
        height=height,
        width=width,
        num_frames=num_frames,
        padding=padding,
        pipeline=base,
    )
    dynamic_config = {
        key: value
        for key, value in pipeline_config.items()
        if key
        not in {
            "checkpoint_path",
            "pipeline_type",
            "precision",
            "prompt_enhancement_words_threshold",
            "prompt_enhancer_image_caption_model_name_or_path",
            "prompt_enhancer_llm_model_name_or_path",
            "sampler",
            "spatial_upscaler_model_path",
            "stg_mode",
            "text_encoder_model_name_or_path",
        }
    }
    prompt_embeds, prompt_attention_mask = (tensor.to("cuda") for tensor in prompt_data)
    negative_prompt_embeds, negative_prompt_attention_mask = (
        tensor.to("cuda") for tensor in negative_prompt_data
    )

    generation_attempts = int(settings.get("generation_attempts_per_clip", 4))
    seed_stride = int(settings.get("motion_seed_stride", 104729))
    seed_attempt_offset = int(clip.get("seed_attempt_offset", 0))
    base_noise_scale = float(settings.get("image_cond_noise_scale", 0.15))
    noise_increment = float(settings.get("motion_retry_noise_increment", 0.05))
    maximum_noise_scale = float(
        settings.get("maximum_image_cond_noise_scale", 0.30)
    )
    minimum_correlation = float(settings.get("minimum_anchor_correlation", 0.5))
    minimum_motion_mad = float(settings.get("minimum_motion_mad", 0.5))
    minimum_changed_percent = float(
        settings.get("minimum_changed_percent", 2.0)
    )
    motion_pixel_delta = int(settings.get("motion_pixel_delta", 3))
    last_failure = ""

    for local_attempt in range(generation_attempts):
        seed_attempt = seed_attempt_offset + local_attempt
        effective_seed = (
            int(clip["seed"]) + seed_attempt * seed_stride
        ) % (2**31)
        noise_scale = min(
            maximum_noise_scale,
            base_noise_scale + local_attempt * noise_increment,
        )
        print(
            f"Motion attempt {local_attempt + 1}/{generation_attempts}: "
            f"seed={effective_seed}, image_cond_noise_scale={noise_scale:.3f}",
            flush=True,
        )
        generator = torch.Generator(device="cuda").manual_seed(effective_seed)

        def run_pipeline(
            output_type: str,
            conditioning_items=conditioning,
        ) -> torch.Tensor:
            return pipeline(
                **dynamic_config,
                skip_layer_strategy=skip_strategy(
                    pipeline_config.get("stg_mode", "attention_values")
                ),
                generator=generator,
                output_type=output_type,
                callback_on_step_end=None,
                height=padded_height,
                width=padded_width,
                num_frames=padded_frames,
                frame_rate=fps,
                prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt=None,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                media_items=None,
                conditioning_items=conditioning_items,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=noise_scale,
                mixed_precision=False,
                offload_to_cpu=True,
                device="cuda",
                enhance_prompt=False,
            ).images

        try:
            result = run_pipeline("pt")
        except torch.OutOfMemoryError as exc:
            last_failure = f"cuda_out_of_memory={exc}"
            release_cuda_memory()
            if os.environ.get("LTX_CPU_VAE_FALLBACK", "0") != "1":
                print(
                    f"CUDA OOM during {clip['output_name']} attempt "
                    f"{local_attempt + 1}/{generation_attempts}; "
                    "aborting this worker so the local runner can resume "
                    "on a fresh runtime.",
                    flush=True,
                )
                del conditioning
                raise
            print(
                f"CUDA OOM during {clip['output_name']} attempt "
                f"{local_attempt + 1}/{generation_attempts}; "
                "retrying the same attempt with CPU VAE decode.",
                flush=True,
            )
            result = pipeline(
                **dynamic_config,
                skip_layer_strategy=skip_strategy(
                    pipeline_config.get("stg_mode", "attention_values")
                ),
                generator=torch.Generator(device="cuda").manual_seed(effective_seed),
                output_type="latent",
                callback_on_step_end=None,
                height=padded_height,
                width=padded_width,
                num_frames=padded_frames,
                frame_rate=fps,
                prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt=None,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                media_items=None,
                conditioning_items=conditioning,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=noise_scale,
                mixed_precision=False,
                offload_to_cpu=True,
                device="cuda",
                enhance_prompt=False,
            ).images
            result = decode_latents_with_cpu_vae(
                base,
                result,
                target_height=padded_height,
                target_width=padded_width,
                vae_per_channel_normalize=True,
                decode_timestep=dynamic_config.get("decode_timestep", 0.0),
                decode_noise_scale=dynamic_config.get("decode_noise_scale"),
                tone_map_compression_ratio=dynamic_config.get(
                    "tone_map_compression_ratio", 0.0
                ),
            )
        left, right, top, bottom = padding
        bottom_index = -bottom if bottom else result.shape[3]
        right_index = -right if right else result.shape[4]
        result = result[:, :, :num_frames, top:bottom_index, left:right_index]
        frames = result[0].permute(1, 2, 3, 0).cpu().float().numpy()
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        anchor_correlations = {
            str(frame_number): conditioning_correlation(
                frames[frame_number], conditioning, padding
            )
            for frame_number in anchor_frames
        }
        motion_metrics = frame_motion_metrics(
            frames,
            pixel_delta=motion_pixel_delta,
        )
        anchors_passed = all(
            np.isfinite(correlation) and correlation >= minimum_correlation
            for correlation in anchor_correlations.values()
        )
        motion_passed = (
            motion_metrics["mean_pair_mad_0_255"] >= minimum_motion_mad
            and motion_metrics["mean_changed_pixels_percent"]
            >= minimum_changed_percent
        )
        print(
            f"Validation for {clip['output_name']}: "
            f"anchor={anchor_correlations}, "
            f"motion_mad={motion_metrics['mean_pair_mad_0_255']:.4f}, "
            f"changed={motion_metrics['mean_changed_pixels_percent']:.3f}%, "
            f"passed={anchors_passed and motion_passed}",
            flush=True,
        )
        if anchors_passed and motion_passed:
            output = OUTPUTS / clip["output_name"]
            temporary = output.with_name(f".{output.name}.partial.mp4")
            temporary.unlink(missing_ok=True)
            with imageio.get_writer(
                temporary,
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            os.replace(temporary, output)
            details = {
                "anchor_correlations": anchor_correlations,
                "effective_seed": effective_seed,
                "generation_attempt": local_attempt + 1,
                "seed_attempt": seed_attempt,
                "image_cond_noise_scale": noise_scale,
                "motion_validation": {
                    **motion_metrics,
                    "minimum_motion_mad": minimum_motion_mad,
                    "minimum_changed_percent": minimum_changed_percent,
                    "passed": True,
                },
            }
            del result, frames, conditioning
            release_cuda_memory()
            return output, details

        last_failure = (
            f"anchors_passed={anchors_passed}, motion_passed={motion_passed}, "
            f"anchor_correlations={anchor_correlations}, "
            f"motion_metrics={motion_metrics}"
        )
        del result, frames
        release_cuda_memory()

    del conditioning
    release_cuda_memory()
    raise RuntimeError(
        f"{clip['output_name']} failed {generation_attempts} motion attempts: "
        f"{last_failure}"
    )


def clip_report(
    clip: dict[str, Any],
    output: Path,
    generation_details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": clip["id"],
        "completed_at": now(),
        "output_name": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "prompt_sha256": hashlib.sha256(clip["prompt"].encode("utf-8")).hexdigest(),
        "generation_sha256": clip["generation_sha256"],
        "anchor_correlations": generation_details["anchor_correlations"],
        "seed": clip["seed"],
        "effective_seed": generation_details["effective_seed"],
        "generation_attempt": generation_details["generation_attempt"],
        "seed_attempt": generation_details["seed_attempt"],
        "image_cond_noise_scale": generation_details["image_cond_noise_scale"],
        "motion_validation": generation_details["motion_validation"],
    }


def generate_clip_sequence(
    *,
    base,
    pipeline,
    pipeline_config: dict[str, Any],
    settings: dict[str, Any],
    clips: list[dict[str, Any]],
    prompt_data: dict[str, tuple[torch.Tensor, torch.Tensor]],
    summary: dict[str, Any],
    start_index: int,
    total_count: int,
) -> int:
    for offset, clip in enumerate(clips, start=0):
        print(
            f"[{start_index + offset}/{total_count}] "
            f"Generating {clip['output_name']}",
            flush=True,
        )
        output, generation_details = generate_one(
            base,
            pipeline,
            pipeline_config,
            settings,
            clip,
            prompt_data[clip["id"]],
            prompt_data["__negative__"],
        )
        report = clip_report(clip, output, generation_details)
        atomic_json(OUTPUTS / f"{output.stem}.json", report)
        summary["clips"].append(report)
        atomic_json(OUTPUTS / "generation.json", summary)
        release_cuda_memory()
    return len(clips)


def wait_for_stream_chunk(chunk_index: int) -> dict[str, Any]:
    ready = CHUNKS_ROOT / f"chunk-{chunk_index:04d}.ready.json"
    archive_path = CHUNKS_ROOT / f"chunk-{chunk_index:04d}.tar.gz"
    last_notice_at = 0.0
    while not ready.is_file() or not archive_path.is_file():
        now_monotonic = time.monotonic()
        if now_monotonic - last_notice_at >= 30:
            print(
                f"Waiting for streamed image chunk {chunk_index:04d}",
                flush=True,
            )
            last_notice_at = now_monotonic
        time.sleep(5)

    shutil.rmtree(JOB_ROOT / "inputs", ignore_errors=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(JOB_ROOT, filter="data")
    chunk = json.loads((JOB_ROOT / "chunk.json").read_text(encoding="utf-8"))
    print(
        f"Received streamed image chunk {chunk_index:04d} with "
        f"{len(chunk['clip_ids'])} clip(s)",
        flush=True,
    )
    ready.unlink(missing_ok=True)
    archive_path.unlink(missing_ok=True)
    return chunk


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    job = json.loads(JOB_PATH.read_text(encoding="utf-8"))
    settings = job["settings"]
    streaming = bool(job.get("streaming"))
    summary: dict[str, Any] = {
        "started_at": now(),
        "status": "running",
        "model_load_count": 0,
        "streaming": streaming,
        "clips": [],
    }
    atomic_json(OUTPUTS / "generation.json", summary)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; a Colab GPU runtime is required")
        prompt_data = encode_prompts(
            settings,
            job["clips"],
            job["prompt_cache_key"],
        )
        base, pipeline, pipeline_config = load_pipeline(settings)
        summary["model_load_count"] = 1
        if streaming:
            stream_batch_size = int(job["stream_batch_size"])
            if stream_batch_size <= 0:
                raise RuntimeError("stream_batch_size must be positive")
            clips_by_id = {clip["id"]: clip for clip in job["clips"]}
            total_chunks = (len(job["clips"]) + stream_batch_size - 1) // stream_batch_size
            generated_count = 0
            for chunk_index in range(total_chunks):
                chunk = wait_for_stream_chunk(chunk_index)
                chunk_clips = [clips_by_id[clip_id] for clip_id in chunk["clip_ids"]]
                generated_count += generate_clip_sequence(
                    base=base,
                    pipeline=pipeline,
                    pipeline_config=pipeline_config,
                    settings=settings,
                    clips=chunk_clips,
                    prompt_data=prompt_data,
                    summary=summary,
                    start_index=generated_count + 1,
                    total_count=len(job["clips"]),
                )
                atomic_json(
                    OUTPUTS / f"chunk-{chunk_index:04d}.done.json",
                    {
                        "chunk_index": chunk_index,
                        "completed_at": now(),
                        "clip_ids": chunk["clip_ids"],
                    },
                )
        else:
            generate_clip_sequence(
                base=base,
                pipeline=pipeline,
                pipeline_config=pipeline_config,
                settings=settings,
                clips=job["clips"],
                prompt_data=prompt_data,
                summary=summary,
                start_index=1,
                total_count=len(job["clips"]),
            )
        summary["status"] = "complete"
        summary["completed_at"] = now()
        atomic_json(OUTPUTS / "generation.json", summary)
        return 0
    except BaseException:
        summary["status"] = "failed"
        summary["completed_at"] = now()
        summary["error"] = traceback.format_exc()
        atomic_json(OUTPUTS / "generation.json", summary)
        (OUTPUTS / "generation_error.log").write_text(
            summary["error"], encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
