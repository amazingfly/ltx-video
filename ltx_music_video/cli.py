from __future__ import annotations

import argparse
import json
import random
import secrets
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .colab import generate_pending, generated_clip_is_valid, generation_sha256
from .gemma import FALLBACK_PROMPT, GemmaClient, build_ltx_prompt
from .manifest import load_manifest, save_manifest, utc_now
from .media import (
    DEFAULT_MINIMUM_CHANGED_PERCENT,
    DEFAULT_MINIMUM_MOTION_MAD,
    DEFAULT_MOTION_PIXEL_DELTA,
    assemble_video,
    assemble_video_with_transitions,
    crossfade_timeline_duration,
    duration_seconds,
    evenly_spaced_indices,
    list_audio,
    list_images,
    normalize_clip,
    required_clip_count,
    required_crossfade_clip_count,
    resolve_transition_output_fps,
    video_is_valid,
)


DEFAULT_IMAGE_DIR = Path("/mnt/storage/projects/agentic/images/scripts/outputs")
IMAGE_DIR_FALLBACK = Path("/mnt/storage/projects/agentic/images/scripts/output")
DEFAULT_MUSIC_DIR = Path("/home/derek/projects/agentic/sa3/musicLibrary/ogg/yes")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEMMA_START = PROJECT_ROOT / "scripts" / "start_gemma_vision.sh"
DEFAULT_GEMMA_STOP = PROJECT_ROOT / "scripts" / "stop_gemma_vision.sh"
MOTION_PROMPT_CONTRACT_VERSION = 3
DEFAULT_TRANSITIONS = (
    "fade",
    "dissolve",
    "fadeblack",
    "fadewhite",
    "hblur",
    "fadegrays",
    "fadefast",
    "fadeslow",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a music video from still images with Gemma 4 and LTX-Video."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Select media and ask Gemma for prompts"
    )
    _add_prepare_arguments(prepare)

    generate = subparsers.add_parser("generate", help="Generate pending clips on Colab")
    _add_generate_arguments(generate)

    assemble = subparsers.add_parser("assemble", help="Join generated clips and music")
    _add_assemble_arguments(assemble)

    assemble_transitions = subparsers.add_parser(
        "assemble-transitions",
        help="Build a slow-playback test video with randomized crossfades",
    )
    _add_transition_assemble_arguments(assemble_transitions)

    all_command = subparsers.add_parser(
        "all", help="Run prepare, generate, and assemble"
    )
    _add_prepare_arguments(all_command)
    _add_colab_options(all_command)
    all_command.add_argument(
        "--output", type=Path, help="Final MP4 path (defaults inside the run directory)"
    )
    return parser


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest", type=Path, help="Manifest path to create or resume"
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--music-dir", type=Path, default=DEFAULT_MUSIC_DIR)
    parser.add_argument(
        "--music", type=Path, help="Use this track instead of a random track"
    )
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--gemma-url", default="http://127.0.0.1:8080")
    parser.add_argument("--gemma-start", type=Path, default=DEFAULT_GEMMA_START)
    parser.add_argument("--gemma-stop", type=Path, default=DEFAULT_GEMMA_STOP)
    parser.add_argument(
        "--keep-gemma-running",
        action="store_true",
        help="Do not stop Gemma when this command started it",
    )
    parser.add_argument(
        "--strict-prompts",
        action="store_true",
        help="Stop instead of using a conservative fallback prompt after Gemma errors",
    )
    parser.add_argument(
        "--regenerate-prompts",
        action="store_true",
        help="Replace every existing prompt and regenerate its clip",
    )
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=49)


def _add_colab_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", default="ltx-music-video")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument(
        "--colab-authuser",
        type=int,
        default=1,
        help="Google browser account index used by the Colab frontend keepalive",
    )
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Maximum clips sent to each Colab runtime; 0 sends all remaining "
            "clips so the loaded LTX model is reused for the whole pending set"
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument(
        "--no-open-frontend",
        action="store_true",
        help="Disable the authenticated Colab frontend keepalive",
    )


def _add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    _add_colab_options(parser)


def _add_assemble_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)


def _add_transition_assemble_arguments(parser: argparse.ArgumentParser) -> None:
    _add_assemble_arguments(parser)
    parser.add_argument("--playback-fps", type=int, default=12)
    parser.add_argument(
        "--output-fps",
        type=int,
        help=(
            "FPS for the encoded transition video after frame interpolation; "
            "defaults to the manifest/source FPS"
        ),
    )
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    parser.add_argument("--transition-seed", type=int)
    parser.add_argument(
        "--transitions",
        default=",".join(DEFAULT_TRANSITIONS),
        help="Comma-separated FFmpeg xfade transition names",
    )


def resolve_image_directory(path: Path) -> Path:
    if path.is_dir():
        return path.resolve()
    if path == DEFAULT_IMAGE_DIR and IMAGE_DIR_FALLBACK.is_dir():
        print(
            f"Image directory {DEFAULT_IMAGE_DIR} was not found; "
            f"using {IMAGE_DIR_FALLBACK}"
        )
        return IMAGE_DIR_FALLBACK.resolve()
    raise FileNotFoundError(f"Image directory does not exist: {path}")


def default_manifest_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (Path("outputs") / timestamp / "manifest.json").resolve()


def choose_repeated(items: list[Path], count: int, rng: random.Random) -> list[Path]:
    if not items:
        raise ValueError("No images are available")
    selected: list[Path] = []
    while len(selected) < count:
        batch = items.copy()
        rng.shuffle(batch)
        selected.extend(batch[: count - len(selected)])
    return selected


def apply_motion_generation_contract(manifest: dict[str, Any]) -> None:
    ltx_settings = manifest["settings"]["ltx"]
    previous_prompt_contract = int(
        ltx_settings.get("prompt_contract_version", 0)
    )
    ltx_settings.update(
        {
            "conditioning_anchor_frames": "start",
            "image_cond_noise_scale": 0.15,
            "motion_retry_noise_increment": 0.05,
            "maximum_image_cond_noise_scale": 0.30,
            "generation_attempts_per_clip": 4,
            "motion_seed_stride": 104729,
            "minimum_anchor_correlation": 0.5,
            "minimum_motion_mad": DEFAULT_MINIMUM_MOTION_MAD,
            "minimum_changed_percent": DEFAULT_MINIMUM_CHANGED_PERCENT,
            "motion_pixel_delta": DEFAULT_MOTION_PIXEL_DELTA,
            "prompt_contract_version": MOTION_PROMPT_CONTRACT_VERSION,
            "negative_prompt": (
                "worst quality, frozen frame, static image, no motion, "
                "inconsistent motion, blurry, jittery, distorted, abrupt "
                "movement, scene change, camera movement, panning, zooming, "
                "reframing, identity change, morphing"
            ),
        }
    )
    if previous_prompt_contract < MOTION_PROMPT_CONTRACT_VERSION:
        for clip in manifest["clips"]:
            motion_prompt = clip.get("motion_prompt")
            if motion_prompt:
                clip["prompt"] = build_ltx_prompt(motion_prompt)
                if clip.get("status") == "generated":
                    clip["status"] = "prompted"
    generation_signature = generation_sha256(ltx_settings)
    for clip in manifest["clips"]:
        clip["generation_sha256"] = generation_signature


def prepare_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    manifest_path = (args.manifest or default_manifest_path()).resolve()
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        print(f"Resuming manifest: {manifest_path}")
    else:
        image_dir = resolve_image_directory(args.image_dir)
        images = list_images(image_dir)
        if not images:
            raise RuntimeError(f"No supported images found in {image_dir}")

        if args.music:
            music = args.music.resolve()
            if not music.is_file():
                raise FileNotFoundError(f"Music track does not exist: {music}")
        else:
            tracks = list_audio(args.music_dir.resolve())
            if not tracks:
                raise RuntimeError(f"No supported music found in {args.music_dir}")
            music = secrets.choice(tracks)

        selection_seed = (
            args.selection_seed
            if args.selection_seed is not None
            else secrets.randbits(63)
        )
        rng = random.Random(selection_seed)
        audio_duration = duration_seconds(music)
        count = required_clip_count(audio_duration, args.clip_seconds)
        selected = choose_repeated(images, count, rng)
        run_dir = manifest_path.parent
        clips_dir = run_dir / "clips"

        manifest = {
            "version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "preparing",
            "selection_seed": selection_seed,
            "music": {
                "path": str(music),
                "duration_seconds": audio_duration,
            },
            "settings": {
                "clip_seconds": args.clip_seconds,
                "gemma_url": args.gemma_url,
                "ltx": {
                    "model_repo": "Lightricks/LTX-Video",
                    "checkpoint": "ltxv-2b-0.9.8-distilled.safetensors",
                    "spatial_upscaler": "ltxv-spatial-upscaler-0.9.8.safetensors",
                    "text_encoder_repo": "city96/t5-v1_1-xxl-encoder-bf16",
                    "tokenizer_repo": "city96/t5-v1_1-xxl-encoder-bf16",
                    "upstream_commit": "4b2d053057623ddd4d0a1d3e9cd28890e9ef487f",
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "num_frames": args.num_frames,
                    "conditioning_anchor_frames": "start",
                    "image_cond_noise_scale": 0.15,
                    "motion_retry_noise_increment": 0.05,
                    "maximum_image_cond_noise_scale": 0.30,
                    "generation_attempts_per_clip": 4,
                    "motion_seed_stride": 104729,
                    "minimum_anchor_correlation": 0.5,
                    "minimum_motion_mad": DEFAULT_MINIMUM_MOTION_MAD,
                    "minimum_changed_percent": DEFAULT_MINIMUM_CHANGED_PERCENT,
                    "motion_pixel_delta": DEFAULT_MOTION_PIXEL_DELTA,
                    "prompt_contract_version": MOTION_PROMPT_CONTRACT_VERSION,
                    "negative_prompt": (
                        "worst quality, frozen frame, static image, no motion, "
                        "inconsistent motion, blurry, jittery, distorted, abrupt "
                        "movement, scene change, camera movement, panning, zooming, "
                        "reframing, identity change, morphing"
                    ),
                },
            },
            "clips": [
                {
                    "id": f"clip_{index:04d}",
                    "index": index,
                    "image_path": str(image),
                    "motion_prompt": None,
                    "prompt": None,
                    "prompt_source": None,
                    "seed": rng.randrange(0, 2**31),
                    "clip_path": str((clips_dir / f"clip_{index:04d}.mp4").resolve()),
                    "status": "selected",
                }
                for index, image in enumerate(selected)
            ],
            "final_video": None,
        }
        save_manifest(manifest_path, manifest)
        print(
            f"Selected {count} image(s) for {audio_duration:.3f}s of music: {music.name}"
        )

    apply_motion_generation_contract(manifest)

    if args.regenerate_prompts:
        for clip in manifest["clips"]:
            clip["motion_prompt"] = None
            clip["prompt"] = None
            clip["prompt_source"] = None
            clip["status"] = "selected"
            clip.pop("prompt_error", None)
        manifest["status"] = "preparing"
        manifest["final_video"] = None
        save_manifest(manifest_path, manifest)

    missing_prompts = [
        clip
        for clip in manifest["clips"]
        if not clip.get("motion_prompt") or not clip.get("prompt")
    ]
    if missing_prompts:
        client = GemmaClient(
            base_url=manifest["settings"].get("gemma_url", args.gemma_url),
            start_command=args.gemma_start,
        )
        client.ensure_ready()
        try:
            for position, clip in enumerate(missing_prompts, start=1):
                print(
                    f"Gemma prompt {position}/{len(missing_prompts)}: "
                    f"{Path(clip['image_path']).name}"
                )
                try:
                    motion_prompt = client.describe_motion(Path(clip["image_path"]))
                    clip["motion_prompt"] = motion_prompt
                    clip["prompt"] = build_ltx_prompt(motion_prompt)
                    clip["prompt_source"] = "gemma4-e4b"
                    clip["status"] = "prompted"
                    clip.pop("prompt_error", None)
                except Exception as exc:
                    if args.strict_prompts:
                        raise
                    clip["motion_prompt"] = FALLBACK_PROMPT
                    clip["prompt"] = build_ltx_prompt(FALLBACK_PROMPT)
                    clip["prompt_source"] = "fallback"
                    clip["prompt_error"] = str(exc)
                    clip["status"] = "prompted"
                    print(f"Gemma failed; using fallback prompt: {exc}")
                save_manifest(manifest_path, manifest)
        finally:
            if (
                client.started_server
                and not args.keep_gemma_running
                and args.gemma_stop.is_file()
            ):
                subprocess.run([str(args.gemma_stop)], check=False)

    manifest["status"] = "prepared"
    save_manifest(manifest_path, manifest)
    print(f"Prepared manifest: {manifest_path}")
    return manifest_path, manifest


def run_generate(args: argparse.Namespace, manifest_path: Path | None = None) -> None:
    path = (manifest_path or args.manifest).resolve()
    manifest = load_manifest(path)
    apply_motion_generation_contract(manifest)
    manifest["status"] = "generating"
    save_manifest(path, manifest)
    generate_pending(
        path,
        manifest,
        session=args.session,
        gpu=args.gpu,
        colab_authuser=args.colab_authuser,
        max_attempts=args.max_attempts,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        open_frontend=not args.no_open_frontend,
    )
    manifest["status"] = "generated"
    save_manifest(path, manifest)
    print(f"All clips generated: {path.parent / 'clips'}")


def run_assemble(args: argparse.Namespace, manifest_path: Path | None = None) -> Path:
    path = (manifest_path or args.manifest).resolve()
    manifest = load_manifest(path)
    ltx = manifest["settings"]["ltx"]
    clip_seconds = float(manifest["settings"]["clip_seconds"])
    normalized_dir = path.parent / "normalized"
    normalized: list[Path] = []
    for clip in manifest["clips"]:
        source = Path(clip["clip_path"])
        if not generated_clip_is_valid(clip):
            raise RuntimeError(
                f"Generated clip failed identity, integrity, or motion validation: "
                f"{source}"
            )
        destination = normalized_dir / source.name
        if (
            not video_is_valid(destination, minimum_duration=clip_seconds - 0.1)
            or destination.stat().st_mtime < source.stat().st_mtime
        ):
            print(f"Normalizing {source.name}")
            normalize_clip(
                source,
                destination,
                duration=clip_seconds,
                width=int(ltx["width"]),
                height=int(ltx["height"]),
                fps=int(ltx["fps"]),
            )
        normalized.append(destination)

    output = (
        args.output.resolve()
        if args.output
        else (path.parent / "music_video.mp4").resolve()
    )
    assemble_video(
        normalized,
        Path(manifest["music"]["path"]),
        output,
        audio_duration=float(manifest["music"]["duration_seconds"]),
    )
    manifest["status"] = "complete"
    manifest["final_video"] = str(output)
    save_manifest(path, manifest)
    print(f"Final video: {output}")
    return output


def run_transition_assemble(args: argparse.Namespace) -> Path:
    path = args.manifest.resolve()
    manifest = load_manifest(path)
    ltx = manifest["settings"]["ltx"]
    source_fps = int(ltx["fps"])
    playback_fps = int(args.playback_fps)
    output_fps = resolve_transition_output_fps(
        source_fps,
        playback_fps,
        args.output_fps,
    )
    source_clip_seconds = float(manifest["settings"]["clip_seconds"])
    transition_seconds = float(args.transition_seconds)
    audio_duration = float(manifest["music"]["duration_seconds"])
    playback_clip_seconds = source_clip_seconds * source_fps / playback_fps
    required_count = required_crossfade_clip_count(
        audio_duration,
        playback_clip_seconds,
        transition_seconds,
    )
    clips = manifest["clips"]
    if required_count > len(clips):
        raise RuntimeError(
            f"The transition assembly needs {required_count} clips, "
            f"but the manifest contains only {len(clips)}"
        )

    selected_indices = evenly_spaced_indices(len(clips), required_count)
    normalized_dir = path.parent / "normalized"
    normalized: list[Path] = []
    selected_ids: list[str] = []
    for index in selected_indices:
        clip = clips[index]
        source = Path(clip["clip_path"])
        if not generated_clip_is_valid(clip):
            raise RuntimeError(
                f"Generated clip failed identity, integrity, or motion validation: "
                f"{source}"
            )
        destination = normalized_dir / source.name
        if (
            not video_is_valid(destination, minimum_duration=source_clip_seconds - 0.1)
            or destination.stat().st_mtime < source.stat().st_mtime
        ):
            print(f"Normalizing {source.name}")
            normalize_clip(
                source,
                destination,
                duration=source_clip_seconds,
                width=int(ltx["width"]),
                height=int(ltx["height"]),
                fps=source_fps,
            )
        normalized.append(destination)
        selected_ids.append(str(clip["id"]))

    transition_pool = [
        transition.strip()
        for transition in str(args.transitions).split(",")
        if transition.strip()
    ]
    if not transition_pool:
        raise ValueError("At least one transition must be supplied")
    transition_seed = (
        args.transition_seed
        if args.transition_seed is not None
        else int(manifest.get("selection_seed", 0))
    )
    rng = random.Random(transition_seed)
    transitions = [
        rng.choice(transition_pool) for _ in range(len(normalized) - 1)
    ]

    output = (
        args.output.resolve()
        if args.output
        else (path.parent / "music_video_12fps_transitions.mp4").resolve()
    )
    available_duration = crossfade_timeline_duration(
        required_count,
        playback_clip_seconds,
        transition_seconds,
    )
    print(
        f"Selected {required_count}/{len(clips)} clips: "
        f"{playback_clip_seconds:.3f}s playback with "
        f"{transition_seconds:.3f}s overlaps provides "
        f"{available_duration:.3f}s for {audio_duration:.3f}s of music; "
        f"interpolating {playback_fps}fps playback to {output_fps}fps output"
    )
    assemble_video_with_transitions(
        normalized,
        Path(manifest["music"]["path"]),
        output,
        audio_duration=audio_duration,
        source_clip_seconds=source_clip_seconds,
        source_fps=source_fps,
        playback_fps=playback_fps,
        output_fps=output_fps,
        transition_seconds=transition_seconds,
        transitions=transitions,
    )
    plan_path = output.with_suffix(".json")
    plan_path.write_text(
        json.dumps(
            {
                "manifest": str(path),
                "output": str(output),
                "music": manifest["music"],
                "source_fps": source_fps,
                "playback_fps": playback_fps,
                "output_fps": output_fps,
                "frame_interpolation": output_fps > playback_fps,
                "source_clip_seconds": source_clip_seconds,
                "playback_clip_seconds": playback_clip_seconds,
                "transition_seconds": transition_seconds,
                "transition_seed": transition_seed,
                "available_video_seconds_before_trim": available_duration,
                "selected_clip_count": required_count,
                "selected_clip_indices": selected_indices,
                "selected_clip_ids": selected_ids,
                "transitions": transitions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Transition test video: {output}")
    print(f"Transition plan: {plan_path}")
    return output


def validate_generation_settings(args: argparse.Namespace) -> None:
    if args.width % 32 or args.height % 32:
        raise ValueError("LTX width and height must be divisible by 32")
    if (args.num_frames - 1) % 8:
        raise ValueError("LTX num-frames must satisfy 8n+1")
    if args.clip_seconds <= 0:
        raise ValueError("--clip-seconds must be positive")
    if getattr(args, "batch_size", 0) < 0:
        raise ValueError("--batch-size cannot be negative")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            validate_generation_settings(args)
            prepare_manifest(args)
        elif args.command == "generate":
            run_generate(args)
        elif args.command == "assemble":
            run_assemble(args)
        elif args.command == "assemble-transitions":
            run_transition_assemble(args)
        elif args.command == "all":
            validate_generation_settings(args)
            manifest_path, _ = prepare_manifest(args)
            run_generate(args, manifest_path)
            run_assemble(args, manifest_path)
        else:
            parser.error(f"Unknown command: {args.command}")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
