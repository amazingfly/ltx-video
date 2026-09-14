from __future__ import annotations

import argparse
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from antigravityPicker.analysis import MusicVideoAnalyzer

CTA_FONT_PATHS = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
)


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def fraction_float(value: str) -> float:
    parsed = nonnegative_float(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Music Video Clipper: find the best sections of music videos for short-form uploads (e.g. YouTube Shorts)."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the outputs directory containing run folders (default: ./outputs)."
    )
    parser.add_argument(
        "--window",
        type=float,
        default=60.0,
        help="Duration of each candidate clip in seconds for the fixed fallback (default: 60.0)."
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Sliding window step duration in seconds for the fixed fallback (default: 1.0)."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of top clips to select per video (default: 5)."
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=15.0,
        help="Soft separation gap in seconds; closer non-overlapping clips are penalized, not discarded (default: 15.0)."
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=45.0,
        help="Minimum clip length for dynamic selection (default: 45.0)."
    )
    parser.add_argument(
        "--target-min-length",
        type=float,
        default=55.0,
        help="Target minimum clip length for dynamic selection (default: 55.0)."
    )
    parser.add_argument(
        "--target-max-length",
        type=float,
        default=65.0,
        help="Target maximum clip length for dynamic selection (default: 65.0)."
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=75.0,
        help="Maximum clip length for dynamic selection (default: 75.0)."
    )
    parser.add_argument(
        "--overlap-penalty",
        type=nonnegative_float,
        default=0.75,
        help="Soft penalty for overlapping selected clips, scaled by top score and overlap ratio (default: 0.75)."
    )
    parser.add_argument(
        "--gap-penalty",
        type=nonnegative_float,
        default=0.15,
        help="Soft penalty for clips inside --min-gap but not overlapping, scaled by top score (default: 0.15)."
    )
    parser.add_argument(
        "--duplicate-overlap-threshold",
        type=fraction_float,
        default=0.92,
        help="Suppress near-duplicate windows when overlap over the shorter clip is at least this ratio (default: 0.92)."
    )
    parser.add_argument(
        "--end-search-radius",
        type=positive_float,
        default=4.0,
        help="Seconds around each nominal end to search for a cleaner audio boundary (default: 4.0)."
    )
    parser.add_argument(
        "--duration-step",
        type=positive_float,
        default=4.0,
        help="Candidate duration grid step in seconds for dynamic selection (default: 4.0)."
    )
    parser.add_argument(
        "--max-dynamic-anchors",
        type=nonnegative_int,
        default=80,
        help="Keep only the top N audio anchors before scoring dynamic windows; 0 disables the cap (default: 80)."
    )
    parser.add_argument(
        "--no-refine-endings",
        action="store_true",
        help="Disable beat/novelty/release-based end-boundary refinement."
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use traditional fixed-length sliding windows instead of dynamic length."
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable exporting clip video files via ffmpeg."
    )
    parser.add_argument(
        "--fast-export",
        action="store_true",
        help="Use stream copy (-c copy) instead of re-encoding when exporting clips (faster, but may have keyframe issues)."
    )
    parser.add_argument(
        "--cta-duration",
        type=positive_float,
        default=5.0,
        help="Seconds to show the end-of-clip CTA (default: 5)."
    )
    parser.add_argument(
        "--cta-scale",
        type=positive_float,
        default=2.0,
        help="CTA text-size multiplier; 1.0 is the original size (default: 2.0)."
    )
    parser.add_argument(
        "--no-cta",
        action="store_true",
        help="Disable the end-of-clip FULL TRACK BELOW overlay."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analysis even if selections.json already exists for a run."
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Process only the newest music_video.mp4 found under --outputs-dir."
    )
    return parser.parse_args(argv)


@lru_cache(maxsize=16)
def probe_video_dimensions(video_path: str) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width_text, height_text = result.stdout.strip().split("x", maxsplit=1)
    return int(width_text), int(height_text)


def cta_timing(clip_duration: float, requested_duration: float) -> tuple[float, float]:
    duration = min(clip_duration, requested_duration)
    return max(0.0, clip_duration - duration), duration


def build_cta_filter(
    width: int,
    height: int,
    start_time: float,
    duration: float,
    size_scale: float = 2.0,
) -> str:
    font_size = max(14, round(min(width, height) * 0.055 * size_scale))
    triangle_size = max(18, round(font_size * 1.35))
    border_width = max(2, round(font_size * 0.07))
    shadow_offset = max(2, round(font_size * 0.05))
    box_padding = max(6, round(font_size * 0.30))
    text_y = round(height * 0.52)
    second_line_y = text_y + round(font_size * 1.10)
    triangle_y = text_y + round(font_size * 2.35)
    end_time = start_time + duration
    enable = f"between(t\\,{start_time:.3f}\\,{end_time:.3f})"
    font_path = next((path for path in CTA_FONT_PATHS if path.is_file()), None)
    font_option = f"fontfile='{font_path}':" if font_path else "font='Sans':"

    def text_filter(text: str, y: int) -> str:
        return (
            f"drawtext={font_option}text='{text}':"
            "fontcolor=white:"
            f"fontsize={font_size}:"
            "bordercolor=black@0.95:"
            f"borderw={border_width}:"
            "shadowcolor=black@0.70:"
            f"shadowx={shadow_offset}:shadowy={shadow_offset}:"
            "box=1:boxcolor=black@0.24:"
            f"boxborderw={box_padding}:"
            "x=(w-text_w)/2:"
            f"y={y}:enable='{enable}'"
        )

    triangle_filter = (
        f"drawtext={font_option}text='▼':"
        "fontcolor=white:"
        f"fontsize={triangle_size}:"
        "bordercolor=black@0.95:"
        f"borderw={border_width}:"
        "shadowcolor=black@0.70:"
        f"shadowx={shadow_offset}:shadowy={shadow_offset}:"
        "x=(w-text_w)/2:"
        f"y={triangle_y}:"
        f"alpha='0.72+0.18*sin(2*PI*1.4*(t-{start_time:.3f}))':"
        f"enable='{enable}'"
    )
    return ",".join(
        (
            text_filter("FULL TRACK", text_y),
            text_filter("BELOW", second_line_y),
            triangle_filter,
        )
    )


def export_clip(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    fast: bool = False,
    cta_enabled: bool = True,
    cta_duration: float = 5.0,
    cta_scale: float = 2.0,
) -> bool:
    """
    Cuts the video file using ffmpeg.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_time - start_time
    cta_start, applied_cta_duration = cta_timing(duration, cta_duration)

    # Base command
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(video_path)
    ]

    if cta_enabled:
        try:
            width, height = probe_video_dimensions(str(video_path.resolve()))
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            print(f"Error probing video dimensions for CTA: {exc}", file=sys.stderr)
            print("     CTA applied: no (video probe failed)")
            return False
        cmd.extend([
            "-vf",
            build_cta_filter(
                width,
                height,
                cta_start,
                applied_cta_duration,
                size_scale=cta_scale,
            ),
        ])
        if fast:
            print("     Fast export disabled for this clip because CTA compositing requires video encoding.")

    if fast and not cta_enabled:
        cmd.extend(["-c", "copy"])
    else:
        # Re-encode for precise cuts at keyframe boundaries
        cmd.extend([
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "22"
        ])

    cmd.append(str(output_path))

    try:
        # Run silently
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )
        if cta_enabled:
            print(
                f"     CTA applied: yes (start: {cta_start:.2f}s, "
                f"duration: {applied_cta_duration:.2f}s, scale: {cta_scale:.2f}x)"
            )
        else:
            print("     CTA applied: no (--no-cta)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error exporting clip: {e.stderr.decode().strip()}", file=sys.stderr)
        print("     CTA applied: no (export failed)")
        return False


def process_video(
    video_path: Path,
    args: argparse.Namespace
) -> dict[str, any] | None:
    run_dir = video_path.parent
    shorts_dir = run_dir / "shorts"
    selections_json_path = shorts_dir / "selections.json"

    if selections_json_path.is_file() and not args.force:
        print(f"-> Found existing selections for {run_dir.name}, skipping (use --force to overwrite).")
        try:
            return json.loads(selections_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    print(f"\n==================================================")
    print(f"Analyzing: {video_path}")
    print(f"==================================================")

    try:
        analyzer = MusicVideoAnalyzer(str(video_path))
    except Exception as e:
        print(f"Failed to load/initialize audio analyzer for {video_path}: {e}", file=sys.stderr)
        return None

    print(f"Duration: {analyzer.duration:.2f}s")
    print("Running feature extraction (librosa & essentia)...")
    features = analyzer.analyze()
    print(f"BPM: {features['bpm']:.1f} (Confidence: {features['beats_confidence']:.2f})")

    print("Scoring candidate windows...")
    candidates = analyzer.score_windows(
        features,
        dynamic=not args.fixed,
        min_length=args.min_length,
        target_min_length=args.target_min_length,
        target_max_length=args.target_max_length,
        max_length=args.max_length,
        window_duration=args.window,
        step_duration=args.step,
        refine_endings=not args.no_refine_endings,
        end_search_radius=args.end_search_radius,
        duration_step=args.duration_step,
        max_dynamic_anchors=args.max_dynamic_anchors,
    )
    print(f"Scored {len(candidates)} candidate windows.")

    print("Selecting top clips (soft overlap penalty; near-duplicates suppressed)...")
    selected_clips = analyzer.select_top_clips(
        candidates,
        num_clips=args.count,
        min_gap=args.min_gap,
        overlap_penalty=args.overlap_penalty,
        gap_penalty=args.gap_penalty,
        duplicate_overlap_threshold=args.duplicate_overlap_threshold,
    )

    # Print results summary
    print(f"\nTop {len(selected_clips)} Recommended Shorts Selections:")
    for rank, clip in enumerate(selected_clips, 1):
        print(f"\n  #{rank} Selection: {clip['start_time']}s to {clip['end_time']}s (Duration: {clip['duration']}s)")
        print(f"    Combined Score: {clip['combined_score']:.4f} [Core: {clip['core_score']:.4f}, Advanced: {clip['advanced_score']:.4f}]")
        print(f"    Selection Score: {clip.get('selection_score', clip['combined_score']):.4f} [Penalty: {clip.get('selection_penalty', 0.0):.4f}, Overlap: {clip.get('selection_overlap_ratio', 0.0):.2f}]")
        print(f"    Score Breakdown:")

        # Display full breakdown of score terms
        bd = clip["breakdown"]
        print(f"      - Loudness/RMS: {bd['loudness']:.2f} | Bass: {bd['bass']:.2f} | Onset: {bd['punchiness']:.2f} | Rhythmic density: {bd['beat_density']:.2f}")
        print(f"      - Drop Likelihood: {bd['drop_likelihood']:.2f} | Chorus Hook: {bd['chorus_hook']:.2f} | Section Novelty: {bd['novelty']:.2f}")
        print(f"      - Start Impact: {bd['start_impact']:.2f} | Ending Cleanliness: {bd['ending_cleanliness']:.2f} | Energy Density: {bd['energy_density']:.2f}")
        print(f"      - End Beat: {bd.get('end_beat_alignment', 0.0):.2f} | End Novelty: {bd.get('end_novelty', 0.0):.2f} | End Release: {bd.get('end_energy_release', 0.0):.2f}")
        print(f"      - Arc/Payoff: {bd['arc_payoff']:.2f} | Phrase Boundary: {bd['phrase_boundary']:.2f} | Short-form Suitability: {bd['short_form_suitability']:.2f}")
        if bd.get("duration_bonus", 0) > 0:
            print(f"      [Bonus] Duration: +{bd['duration_bonus']:.2f}")

        if bd["repetition_penalty"] > 0:
            print(f"      [Penalty] Repetition: -{bd['repetition_penalty']:.2f}")
        if bd["padding_penalty"] > 0:
            print(f"      [Penalty] Padding/Length: -{bd['padding_penalty']:.2f}")
        if bd["silence_penalty"] > 0:
            print(f"      [Penalty] Silence/Low-Energy: -{bd['silence_penalty']:.2f}")
        if bd["intro_outro_penalty"] > 0:
            print(f"      [Penalty] Intro/Outro: -{bd['intro_outro_penalty']:.2f}")

    # Save JSON report
    report = {
        "video_path": str(video_path.resolve()),
        "bpm": round(float(features["bpm"]), 2),
        "duration": round(float(analyzer.duration), 2),
        "selections": selected_clips
    }

    shorts_dir.mkdir(parents=True, exist_ok=True)
    selections_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved selections metadata to: {selections_json_path}")

    # Export videos
    if not args.no_export:
        print("\nExporting clip videos...")
        for stale_clip in shorts_dir.glob("short_*.mp4"):
            stale_clip.unlink()

        for rank, clip in enumerate(selected_clips, 1):
            clip_name = f"short_{rank}_ss{clip['start_time']}_to_{clip['end_time']}.mp4"
            clip_path = shorts_dir / clip_name
            print(f"  -> Exporting {clip_name}...")
            success = export_clip(
                video_path,
                clip_path,
                clip["start_time"],
                clip["end_time"],
                fast=args.fast_export,
                cta_enabled=not args.no_cta,
                cta_duration=args.cta_duration,
                cta_scale=args.cta_scale,
            )
            if success:
                print(f"     Successfully exported: {clip_path}")
                clip["exported_path"] = str(clip_path.resolve())
                clip_duration = clip["end_time"] - clip["start_time"]
                cta_start, applied_duration = cta_timing(clip_duration, args.cta_duration)
                clip["cta"] = {
                    "applied": not args.no_cta,
                    "start_time": round(cta_start, 3) if not args.no_cta else None,
                    "duration": round(applied_duration, 3) if not args.no_cta else 0.0,
                    "scale": args.cta_scale if not args.no_cta else None,
                }
            else:
                print(f"     Failed to export: {clip_path}")

        # Re-save report with exported paths
        selections_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        print(f"Error: Outputs directory not found at: {outputs_dir}", file=sys.stderr)
        return 1

    print(f"Scanning '{outputs_dir}' for music_video.mp4 files...")

    # Find all music_video.mp4 files recursively under outputs/
    video_files = sorted(list(outputs_dir.glob("**/music_video.mp4")))
    if not video_files:
        print(f"No music_video.mp4 files found under {outputs_dir}.")
        return 0
    if args.latest_only:
        video_files = [max(video_files, key=lambda path: path.stat().st_mtime)]
        print(f"Latest-only mode selected: {video_files[0]}")

    print(f"Found {len(video_files)} music video(s) to process.")

    results = {}
    for idx, video in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing run directory: {video.parent.name}")
        report = process_video(video, args)
        if report:
            results[video.parent.name] = report

    print("\n==================================================")
    print("All processing complete!")
    print("==================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
