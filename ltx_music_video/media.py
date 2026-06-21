from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
DEFAULT_MINIMUM_MOTION_MAD = 0.5
DEFAULT_MINIMUM_CHANGED_PERCENT = 2.0
DEFAULT_MOTION_PIXEL_DELTA = 3


def require_program(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required program is not on PATH: {name}")
    return path


def ffprobe(path: Path) -> dict[str, Any]:
    require_program("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def duration_seconds(path: Path) -> float:
    probe = ffprobe(path)
    duration = probe.get("format", {}).get("duration")
    if duration is None:
        for stream in probe.get("streams", []):
            if stream.get("duration") is not None:
                duration = stream["duration"]
                break
    if duration is None:
        raise RuntimeError(f"Could not determine media duration: {path}")
    return float(duration)


def required_clip_count(audio_duration: float, clip_seconds: float) -> int:
    if audio_duration <= 0 or clip_seconds <= 0:
        raise ValueError("Audio duration and clip duration must be positive")
    return math.ceil(audio_duration / clip_seconds)


def crossfade_timeline_duration(
    clip_count: int,
    clip_seconds: float,
    transition_seconds: float,
) -> float:
    if clip_count <= 0:
        raise ValueError("Clip count must be positive")
    if clip_seconds <= 0:
        raise ValueError("Clip duration must be positive")
    if transition_seconds < 0 or transition_seconds >= clip_seconds:
        raise ValueError("Transition duration must be non-negative and shorter than a clip")
    return clip_count * clip_seconds - (clip_count - 1) * transition_seconds


def required_crossfade_clip_count(
    audio_duration: float,
    clip_seconds: float,
    transition_seconds: float,
) -> int:
    if audio_duration <= 0:
        raise ValueError("Audio duration must be positive")
    crossfade_timeline_duration(1, clip_seconds, transition_seconds)
    if audio_duration <= clip_seconds:
        return 1
    usable_seconds = clip_seconds - transition_seconds
    return math.ceil((audio_duration - transition_seconds) / usable_seconds)


def resolve_transition_output_fps(
    source_fps: int,
    playback_fps: int,
    output_fps: int | None,
) -> int:
    if source_fps <= 0 or playback_fps <= 0:
        raise ValueError("Source and playback FPS must be positive")
    resolved = source_fps if output_fps is None else output_fps
    if resolved <= 0:
        raise ValueError("Output FPS must be positive")
    if resolved < playback_fps:
        raise ValueError("Output FPS cannot be lower than playback FPS")
    return resolved


def evenly_spaced_indices(item_count: int, selected_count: int) -> list[int]:
    if item_count <= 0:
        raise ValueError("Item count must be positive")
    if selected_count <= 0 or selected_count > item_count:
        raise ValueError("Selected count must be between one and the item count")
    if selected_count == 1:
        return [0]
    return [
        round(position * (item_count - 1) / (selected_count - 1))
        for position in range(selected_count)
    ]


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_audio(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def video_is_valid(path: Path, minimum_duration: float = 0.5) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        probe = ffprobe(path)
        has_video = any(
            stream.get("codec_type") == "video" for stream in probe.get("streams", [])
        )
        duration = float(probe.get("format", {}).get("duration", 0))
        return has_video and duration >= minimum_duration
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        return False


def video_decodes_cleanly(path: Path, minimum_duration: float = 0.5) -> bool:
    if not video_is_valid(path, minimum_duration=minimum_duration):
        return False
    try:
        subprocess.run(
            [
                require_program("ffmpeg"),
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-an",
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def video_motion_metrics(
    path: Path,
    *,
    analysis_width: int = 192,
    pixel_delta: int = DEFAULT_MOTION_PIXEL_DELTA,
) -> dict[str, float | int]:
    if analysis_width < 32:
        raise ValueError("Motion analysis width must be at least 32")
    if pixel_delta < 1:
        raise ValueError("Motion pixel delta must be positive")
    probe = ffprobe(path)
    stream = next(
        (
            item
            for item in probe.get("streams", [])
            if item.get("codec_type") == "video"
        ),
        None,
    )
    if not stream:
        raise RuntimeError(f"No video stream found: {path}")
    source_width = int(stream["width"])
    source_height = int(stream["height"])
    analysis_height = max(
        2,
        round(source_height * analysis_width / source_width / 2) * 2,
    )
    result = subprocess.run(
        [
            require_program("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(path),
            "-an",
            "-vf",
            f"scale={analysis_width}:{analysis_height},format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    frame_bytes = analysis_width * analysis_height
    if len(result.stdout) % frame_bytes:
        raise RuntimeError(f"Unexpected decoded frame size for {path}")
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        -1,
        analysis_height,
        analysis_width,
    )
    if len(frames) < 2:
        raise RuntimeError(f"Not enough frames for motion validation: {path}")
    differences = np.abs(np.diff(frames.astype(np.int16), axis=0))
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


def video_has_motion(
    path: Path,
    *,
    minimum_motion_mad: float = DEFAULT_MINIMUM_MOTION_MAD,
    minimum_changed_percent: float = DEFAULT_MINIMUM_CHANGED_PERCENT,
    pixel_delta: int = DEFAULT_MOTION_PIXEL_DELTA,
) -> bool:
    if not video_is_valid(path):
        return False
    try:
        metrics = video_motion_metrics(path, pixel_delta=pixel_delta)
        return (
            float(metrics["mean_pair_mad_0_255"]) >= minimum_motion_mad
            and float(metrics["mean_changed_pixels_percent"])
            >= minimum_changed_percent
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return False


def conditioned_video_is_valid(
    video: Path,
    image: Path,
    *,
    minimum_correlation: float = 0.5,
    sample_positions: tuple[float, ...] = (0.0, 0.5, 0.95),
) -> bool:
    if not video_is_valid(video) or not image.is_file():
        return False
    if not sample_positions or any(
        position < 0.0 or position > 1.0 for position in sample_positions
    ):
        raise ValueError("Conditioning sample positions must be between zero and one")
    try:
        duration = duration_seconds(video)
        timestamps = tuple(duration * position for position in sample_positions)
        source_original = Image.open(image).convert("RGB")
        similarities: list[float] = []
        for timestamp in timestamps:
            result = subprocess.run(
                [
                    require_program("ffmpeg"),
                    "-v",
                    "error",
                    "-i",
                    str(video),
                    "-ss",
                    f"{timestamp:.6f}",
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ],
                check=True,
                capture_output=True,
            )
            generated = Image.open(BytesIO(result.stdout)).convert("RGB")
            source = source_original.copy()
            target_ratio = generated.width / generated.height
            source_ratio = source.width / source.height
            if source_ratio > target_ratio:
                width = round(source.height * target_ratio)
                left = (source.width - width) // 2
                source = source.crop((left, 0, left + width, source.height))
            else:
                height = round(source.width / target_ratio)
                top = (source.height - height) // 2
                source = source.crop((0, top, source.width, top + height))
            source = source.resize(generated.size, Image.Resampling.LANCZOS)

            source_gray = np.asarray(source, dtype=np.float32).mean(axis=2).reshape(-1)
            generated_gray = (
                np.asarray(generated, dtype=np.float32).mean(axis=2).reshape(-1)
            )
            if source_gray.std() < 7.5 or generated_gray.std() < 7.5:
                similarity = 1.0 - float(
                    np.abs(source_gray - generated_gray).mean() / 255.0
                )
            else:
                similarity = float(np.corrcoef(source_gray, generated_gray)[0, 1])
            similarities.append(similarity)
        return all(
            math.isfinite(similarity) and similarity >= minimum_correlation
            for similarity in similarities
        )
    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ):
        return False


def normalize_clip(
    source: Path,
    destination: Path,
    *,
    duration: float,
    width: int,
    height: int,
    fps: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial.mp4")
    temporary.unlink(missing_ok=True)
    filter_graph = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1"
    )
    subprocess.run(
        [
            require_program("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-t",
            f"{duration:.6f}",
            "-vf",
            filter_graph,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(destination)


def assemble_video(
    clips: list[Path],
    music: Path,
    output: Path,
    *,
    audio_duration: float,
) -> None:
    if not clips:
        raise ValueError("No clips were supplied for assembly")
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_path = output.parent / "concat.txt"
    concat_path.write_text(
        "".join(f"file '{_ffconcat_escape(path.resolve())}'\n" for path in clips),
        encoding="utf-8",
    )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial.mp4")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        [
            require_program("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-i",
            str(music),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{audio_duration:.6f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    minimum_output_duration = max(0.5, audio_duration - 0.5)
    if not video_decodes_cleanly(
        temporary,
        minimum_duration=minimum_output_duration,
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Assembled video failed decode validation: {temporary}")
    temporary.replace(output)


def assemble_video_with_transitions(
    clips: Sequence[Path],
    music: Path,
    output: Path,
    *,
    audio_duration: float,
    source_clip_seconds: float,
    source_fps: int,
    playback_fps: int,
    output_fps: int | None,
    transition_seconds: float,
    transitions: Sequence[str],
) -> None:
    if not clips:
        raise ValueError("No clips were supplied for assembly")
    resolved_output_fps = resolve_transition_output_fps(
        source_fps,
        playback_fps,
        output_fps,
    )
    if playback_fps > source_fps:
        raise ValueError("Playback FPS cannot exceed source FPS for slow playback")
    if len(transitions) != len(clips) - 1:
        raise ValueError("One transition is required between each pair of clips")

    playback_factor = source_fps / playback_fps
    playback_clip_seconds = source_clip_seconds * playback_factor
    available_duration = crossfade_timeline_duration(
        len(clips),
        playback_clip_seconds,
        transition_seconds,
    )
    if available_duration + 1e-6 < audio_duration:
        raise ValueError(
            f"Selected clips cover only {available_duration:.3f}s, "
            f"shorter than the {audio_duration:.3f}s soundtrack"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    filter_path = output.with_suffix(".filter_complex.txt")
    if resolved_output_fps == playback_fps:
        clip_fps_filter = f"fps={playback_fps}"
    else:
        clip_fps_filter = (
            f"minterpolate=fps={resolved_output_fps}:"
            "mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        )
    filter_lines = [
        (
            f"[{index}:v]setpts={playback_factor:.9f}*PTS,"
            f"{clip_fps_filter},settb=AVTB,format=yuv420p[v{index}]"
        )
        for index in range(len(clips))
    ]
    current_label = "v0"
    for index, transition in enumerate(transitions, start=1):
        output_label = f"x{index}"
        offset = index * (playback_clip_seconds - transition_seconds)
        filter_lines.append(
            f"[{current_label}][v{index}]"
            f"xfade=transition={transition}:duration={transition_seconds:.9f}:"
            f"offset={offset:.9f}[{output_label}]"
        )
        current_label = output_label
    filter_lines.append(
        f"[{current_label}]trim=duration={audio_duration:.9f},"
        "setpts=PTS-STARTPTS[vout]"
    )
    filter_path.write_text(";\n".join(filter_lines) + "\n", encoding="utf-8")

    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial.mp4")
    temporary.unlink(missing_ok=True)
    command = [require_program("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        command.extend(["-threads", "1", "-i", str(clip)])
    command.extend(
        [
            "-i",
            str(music),
            "-filter_complex_script",
            str(filter_path),
            "-map",
            "[vout]",
            "-map",
            f"{len(clips)}:a:0",
            "-t",
            f"{audio_duration:.9f}",
            "-r",
            str(resolved_output_fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    subprocess.run(command, check=True)
    minimum_output_duration = max(0.5, audio_duration - 0.5)
    if not video_decodes_cleanly(
        temporary,
        minimum_duration=minimum_output_duration,
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Transition video failed decode validation: {temporary}")
    temporary.replace(output)


def _ffconcat_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")
