#!/usr/bin/env python3
"""Measure real frame-to-frame motion in an LTX music-video run."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np


STATIC_MAD_THRESHOLD = 0.25
STATIC_CHANGED_PERCENT_THRESHOLD = 0.5
MINIMAL_MAD_THRESHOLD = 0.75
MINIMAL_CHANGED_PERCENT_THRESHOLD = 2.0


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {path}")
    return streams[0]


def parse_rate(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def decode_grayscale(path: Path, analysis_width: int) -> tuple[np.ndarray, float]:
    probe = probe_video(path)
    source_width = int(probe["width"])
    source_height = int(probe["height"])
    analysis_height = max(
        2,
        round(source_height * analysis_width / source_width / 2) * 2,
    )
    result = subprocess.run(
        [
            "ffmpeg",
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
    return frames, parse_rate(probe["avg_frame_rate"])


def classify_motion(pair_mad: float, changed_percent: float) -> str:
    if (
        pair_mad < STATIC_MAD_THRESHOLD
        and changed_percent < STATIC_CHANGED_PERCENT_THRESHOLD
    ):
        return "effectively_static"
    if (
        pair_mad < MINIMAL_MAD_THRESHOLD
        and changed_percent < MINIMAL_CHANGED_PERCENT_THRESHOLD
    ):
        return "minimal_motion"
    return "visible_motion"


def analyze_frames(frames: np.ndarray, fps: float) -> dict[str, Any]:
    if len(frames) < 2:
        raise ValueError("At least two frames are required for motion analysis")
    signed = frames.astype(np.int16)
    differences = np.abs(np.diff(signed, axis=0))
    pair_mad = differences.mean(axis=(1, 2))
    pair_changed = (differences >= 3).mean(axis=(1, 2)) * 100.0
    exact_duplicate_pairs = sum(
        np.array_equal(frames[index], frames[index + 1])
        for index in range(len(frames) - 1)
    )
    average_mad = float(pair_mad.mean())
    average_changed = float(pair_changed.mean())
    return {
        "frame_count": int(len(frames)),
        "fps": fps,
        "duration_seconds": float(len(frames) / fps),
        "mean_pair_mad_0_255": average_mad,
        "median_pair_mad_0_255": float(np.median(pair_mad)),
        "maximum_pair_mad_0_255": float(pair_mad.max()),
        "mean_changed_pixels_ge_3_percent": average_changed,
        "maximum_changed_pixels_ge_3_percent": float(pair_changed.max()),
        "first_last_mad_0_255": float(
            np.abs(signed[-1] - signed[0]).mean()
        ),
        "mean_temporal_stddev_0_255": float(
            signed.astype(np.float32).std(axis=0).mean()
        ),
        "exact_duplicate_pairs": exact_duplicate_pairs,
        "pair_count": int(len(frames) - 1),
        "classification": classify_motion(average_mad, average_changed),
    }


def analyze_video(path: Path, analysis_width: int) -> dict[str, Any]:
    frames, fps = decode_grayscale(path, analysis_width)
    return {
        "path": str(path.resolve()),
        **analyze_frames(frames, fps),
    }


def analyze_final_segments(
    path: Path,
    *,
    clip_seconds: float,
    expected_segments: int,
    analysis_width: int,
) -> dict[str, Any]:
    frames, fps = decode_grayscale(path, analysis_width)
    frames_per_segment = round(clip_seconds * fps)
    segments = []
    for index in range(expected_segments):
        start = index * frames_per_segment
        end = min(start + frames_per_segment, len(frames))
        if end - start < 2:
            break
        segments.append(
            {
                "index": index,
                **analyze_frames(frames[start:end], fps),
            }
        )
    return {
        "path": str(path.resolve()),
        "frame_count": int(len(frames)),
        "fps": fps,
        "duration_seconds": float(len(frames) / fps),
        "segment_seconds": clip_seconds,
        "segments": segments,
        "summary": summarize(segments),
    }


def summarize(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(results)
    if not items:
        return {"count": 0}
    classifications: dict[str, int] = {}
    for item in items:
        classification = item["classification"]
        classifications[classification] = classifications.get(classification, 0) + 1
    mean_mads = [float(item["mean_pair_mad_0_255"]) for item in items]
    changed = [
        float(item["mean_changed_pixels_ge_3_percent"]) for item in items
    ]
    return {
        "count": len(items),
        "classifications": classifications,
        "mean_of_pair_mad_0_255": mean(mean_mads),
        "median_of_pair_mad_0_255": median(mean_mads),
        "mean_changed_pixels_ge_3_percent": mean(changed),
        "static_percent": classifications.get("effectively_static", 0)
        / len(items)
        * 100.0,
    }


def analyze_many(
    paths: list[Path],
    *,
    analysis_width: int,
    workers: int,
) -> list[dict[str, Any]]:
    def analyze(path: Path) -> dict[str, Any]:
        try:
            return analyze_video(path, analysis_width)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            return {"path": str(path.resolve()), "error": str(exc)}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(analyze, paths))


def valid_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if "error" not in result]


def build_report(
    manifest_path: Path,
    *,
    analysis_width: int,
    workers: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    raw_paths = [Path(clip["clip_path"]) for clip in manifest["clips"]]
    normalized_paths = [
        run_dir / "normalized" / path.name for path in raw_paths
    ]
    normalized_paths = [path for path in normalized_paths if path.is_file()]

    print(f"Analyzing {len(raw_paths)} raw Colab clips")
    raw_results = analyze_many(
        raw_paths,
        analysis_width=analysis_width,
        workers=workers,
    )
    print(f"Analyzing {len(normalized_paths)} normalized clips")
    normalized_results = analyze_many(
        normalized_paths,
        analysis_width=analysis_width,
        workers=workers,
    )

    final_results: dict[str, Any] = {}
    final_path_value = manifest.get("final_video")
    if final_path_value:
        final_path = Path(final_path_value)
        if final_path.is_file():
            print(f"Analyzing final video segments: {final_path.name}")
            final_results["manifest_final"] = analyze_final_segments(
                final_path,
                clip_seconds=float(manifest["settings"]["clip_seconds"]),
                expected_segments=len(raw_paths),
                analysis_width=analysis_width,
            )

    transition_path = run_dir / "music_video_12fps_transitions.mp4"
    if transition_path.is_file():
        print(f"Analyzing transition video overall: {transition_path.name}")
        final_results["transition_video_overall"] = analyze_video(
            transition_path,
            analysis_width,
        )

    raw_valid = valid_results(raw_results)
    normalized_valid = valid_results(normalized_results)
    return {
        "manifest": str(manifest_path.resolve()),
        "thresholds": {
            "effectively_static": {
                "mean_pair_mad_0_255_below": STATIC_MAD_THRESHOLD,
                "mean_changed_pixels_ge_3_percent_below": (
                    STATIC_CHANGED_PERCENT_THRESHOLD
                ),
            },
            "minimal_motion": {
                "mean_pair_mad_0_255_below": MINIMAL_MAD_THRESHOLD,
                "mean_changed_pixels_ge_3_percent_below": (
                    MINIMAL_CHANGED_PERCENT_THRESHOLD
                ),
            },
        },
        "provenance": {
            "raw_clip_count": len(raw_paths),
            "raw_clips_from_manifest": True,
            "normalized_clip_directory": str((run_dir / "normalized").resolve()),
            "manifest_final_video": final_path_value,
        },
        "raw_clips": {
            "summary": summarize(raw_valid),
            "results": raw_results,
        },
        "normalized_clips": {
            "summary": summarize(normalized_valid),
            "results": normalized_results,
        },
        "final_videos": final_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure frame-to-frame changes in raw LTX clips, normalized clips, "
            "and assembled videos."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analysis-width", type=int, default=192)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    if args.analysis_width < 32:
        raise SystemExit("error: --analysis-width must be at least 32")
    if args.workers < 1:
        raise SystemExit("error: --workers must be positive")
    report = build_report(
        manifest_path,
        analysis_width=args.analysis_width,
        workers=args.workers,
    )
    output = (
        args.output.resolve()
        if args.output
        else manifest_path.parent / "motion_analysis.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raw = report["raw_clips"]["summary"]
    normalized = report["normalized_clips"]["summary"]
    print(
        f"Raw clips: {raw['static_percent']:.1f}% effectively static; "
        f"mean frame MAD {raw['mean_of_pair_mad_0_255']:.4f}/255"
    )
    print(
        f"Normalized clips: {normalized['static_percent']:.1f}% effectively static; "
        f"mean frame MAD {normalized['mean_of_pair_mad_0_255']:.4f}/255"
    )
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
