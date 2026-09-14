from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from antigravityPicker.analysis_v3 import (
    ALGORITHM_VERSION,
    SCORING_MODEL_VERSION,
    MusicVideoAnalyzerV3,
)
from antigravityPicker.cli import (
    cta_timing,
    export_clip,
    nonnegative_int,
)
from antigravityPicker.preferences_v3 import PreferenceProfile


SCHEMA_VERSION = 3


def positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite number")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def finite_fraction_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite number")
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def relative_subdir(value: str) -> Path:
    path = Path(value)
    if (
        not value.strip()
        or path.is_absolute()
        or ".." in path.parts
        or path == Path(".")
        or path.parts[:2] != ("shorts", "v3")
    ):
        raise argparse.ArgumentTypeError(
            "must be shorts/v3 or a directory below shorts/v3"
        )
    return path


def _safe_output_dir(run_dir: Path, subdir: Path) -> Path:
    relative = relative_subdir(str(subdir))
    run_root = run_dir.resolve()
    output_dir = (run_root / relative).resolve(strict=False)
    if not output_dir.is_relative_to(run_root):
        raise ValueError(
            f"V3 output directory resolves outside the run directory: {output_dir}"
        )
    return output_dir


def _safe_descendant(root: Path, relative: Path) -> Path:
    root = root.resolve(strict=False)
    path = (root / relative).resolve(strict=False)
    if not path.is_relative_to(root):
        raise ValueError(f"V3 artifact resolves outside its output directory: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Antigravity Picker V3: quality-first, phrase-aligned music highlight "
            "selection with one champion per musical event."
        )
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-subdir",
        type=relative_subdir,
        default=Path("shorts/v3"),
        help="Per-run V3 output directory (default: shorts/v3).",
    )
    parser.add_argument("--count", type=nonnegative_int, default=5)
    parser.add_argument("--min-length", type=positive_finite_float, default=45.0)
    parser.add_argument("--target-min-length", type=positive_finite_float, default=55.0)
    parser.add_argument("--target-max-length", type=positive_finite_float, default=65.0)
    parser.add_argument("--max-length", type=positive_finite_float, default=75.0)
    parser.add_argument(
        "--boundary-search-radius",
        type=positive_finite_float,
        default=4.0,
        help="Seconds around both nominal cuts to search for phrase boundaries.",
    )
    parser.add_argument(
        "--max-nuclei",
        type=nonnegative_int,
        default=36,
        help="Maximum compact highlight events expanded into full clips.",
    )
    parser.add_argument(
        "--quality-floor",
        type=finite_fraction_float,
        default=0.94,
        help="Minimum quality relative to the song champion (default: 0.94).",
    )
    parser.add_argument(
        "--quality-floor-absolute",
        type=finite_fraction_float,
        default=None,
        help="Optional absolute score floor in addition to --quality-floor.",
    )
    parser.add_argument(
        "--preference-history",
        type=Path,
        default=None,
        help="Optional outputs directory containing historical shorts/1This choices.",
    )
    parser.add_argument(
        "--preference-weight",
        type=finite_fraction_float,
        default=0.15,
        help="Maximum validated preference-model contribution (default: 0.15).",
    )
    parser.add_argument("--preference-min-tracks", type=nonnegative_int, default=10)
    parser.add_argument("--preference-ridge", type=positive_finite_float, default=4.0)
    parser.add_argument("--no-candidate-report", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--fast-export", action="store_true")
    parser.add_argument("--cta-duration", type=positive_finite_float, default=5.0)
    parser.add_argument("--cta-scale", type=positive_finite_float, default=2.0)
    parser.add_argument("--no-cta", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--latest-only", action="store_true")
    args = parser.parse_args(argv)

    lengths = (
        args.min_length,
        args.target_min_length,
        args.target_max_length,
        args.max_length,
    )
    if tuple(sorted(lengths)) != lengths:
        parser.error(
            "clip lengths must satisfy --min-length <= --target-min-length <= "
            "--target-max-length <= --max-length"
        )
    if args.max_nuclei < 1:
        parser.error("--max-nuclei must be at least 1")
    if args.preference_min_tracks < 1:
        parser.error("--preference-min-tracks must be at least 1")
    return args


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _archive_artifact(path: Path, output_dir: Path, generation_id: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    history_dir = _safe_descendant(output_dir, Path("history") / generation_id)
    history_dir.mkdir(parents=True, exist_ok=False)
    os.replace(path, history_dir / path.name)


def _music_path(run_dir: Path) -> str | None:
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    music = manifest.get("music")
    if not isinstance(music, dict):
        return None
    value = music.get("path")
    return value if isinstance(value, str) and value else None


def _preference_profile(
    args: argparse.Namespace,
    current_track: str | None,
) -> PreferenceProfile | None:
    if args.preference_history is None:
        return None
    return PreferenceProfile.from_history(
        args.preference_history.resolve(),
        current_track=current_track,
        min_tracks=args.preference_min_tracks,
        ridge=args.preference_ridge,
    )


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "count": args.count,
        "min_length": args.min_length,
        "target_min_length": args.target_min_length,
        "target_max_length": args.target_max_length,
        "max_length": args.max_length,
        "boundary_search_radius": args.boundary_search_radius,
        "max_nuclei": args.max_nuclei,
        "quality_floor": args.quality_floor,
        "quality_floor_absolute": args.quality_floor_absolute,
        "preference_weight": args.preference_weight,
        "cta_enabled": not args.no_cta,
        "cta_duration": args.cta_duration,
        "cta_scale": args.cta_scale,
        "export_enabled": not args.no_export,
        "fast_export": args.fast_export,
    }


def _candidate_diagnostics(
    analyzer: MusicVideoAnalyzerV3,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    quality_floor_ratio: float,
    quality_floor_absolute: float | None,
) -> tuple[list[dict[str, Any]], dict[str, int | float | None]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item["quality_score"]),
            float(item["start_time"]),
            float(item["end_time"]),
        ),
    )
    clusters = analyzer.cluster_events(ordered)
    event_by_candidate: dict[str, tuple[str, bool]] = {}
    for event_index, cluster in enumerate(clusters, 1):
        event_id = f"event_{event_index:03d}"
        for candidate_index, candidate in enumerate(cluster):
            event_by_candidate[str(candidate.get("candidate_id", ""))] = (
                event_id,
                candidate_index == 0,
            )

    selected_ranks = {
        str(item.get("candidate_id", "")): int(item["selection_rank"])
        for item in selected
    }
    best_quality = float(ordered[0]["quality_score"]) if ordered else 0.0
    floor = best_quality * quality_floor_ratio
    if quality_floor_absolute is not None:
        floor = max(floor, quality_floor_absolute)

    diagnostics: list[dict[str, Any]] = []
    event_champions = 0
    above_floor = 0
    for quality_rank, candidate in enumerate(ordered, 1):
        item = dict(candidate)
        item["breakdown"] = dict(candidate.get("breakdown", {}))
        event_id, is_event_champion = event_by_candidate.get(
            str(candidate.get("candidate_id", "")),
            ("", False),
        )
        item["quality_rank"] = quality_rank
        item["event_id"] = event_id
        item["is_event_champion"] = is_event_champion
        item["selection_rank"] = selected_ranks.get(
            str(candidate.get("candidate_id", ""))
        )
        if is_event_champion:
            event_champions += 1
            if float(candidate["quality_score"]) >= floor:
                above_floor += 1
        if item["selection_rank"] is not None:
            item["rejection_reason"] = None
        elif not is_event_champion:
            item["rejection_reason"] = "alternate_edit_of_stronger_event_champion"
        elif float(candidate["quality_score"]) < floor:
            item["rejection_reason"] = "below_quality_floor"
        else:
            item["rejection_reason"] = "outside_requested_count"
        diagnostics.append(item)

    summary: dict[str, int | float | None] = {
        "generated": len(candidates),
        "event_clusters": len(clusters),
        "event_champions": event_champions,
        "event_champions_above_floor": above_floor,
        "selected": len(selected),
        "quality_floor_score": round(floor, 4) if ordered else None,
    }
    return diagnostics, summary


def process_video(video_path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    run_dir = video_path.parent
    try:
        output_dir = _safe_output_dir(run_dir, args.output_subdir)
        selections_path = _safe_descendant(output_dir, Path("selections.json"))
        candidates_path = _safe_descendant(output_dir, Path("candidates.json"))
    except (argparse.ArgumentTypeError, OSError, ValueError) as exc:
        print(f"Unsafe V3 output path: {exc}", file=sys.stderr)
        return None

    if selections_path.is_file() and not args.force:
        print(
            f"-> Found existing V3 report for {run_dir.name}; skipping (use --force)."
        )
        try:
            return json.loads(selections_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass

    print("\n==================================================")
    print(f"V3 analysis: {video_path}")
    print("==================================================")
    try:
        analyzer = MusicVideoAnalyzerV3(str(video_path))
    except Exception as exc:
        print(f"Failed to initialize V3 analyzer: {exc}", file=sys.stderr)
        return None

    print(f"Duration: {analyzer.duration:.2f}s")
    print("Extracting audio and structural features...")
    features = analyzer.analyze()
    print(
        f"BPM: {float(features['bpm']):.1f} | beats: {len(features['beats'])} "
        f"| source: {features['beat_source']} | coverage: {features['beat_coverage']:.1%}"
    )

    current_track = _music_path(run_dir)
    preference_profile = _preference_profile(args, current_track)
    if preference_profile is not None:
        if preference_profile.enabled:
            print(
                "Preference reranker enabled: "
                f"{preference_profile.track_count} tracks, validation "
                f"{preference_profile.validation_accuracy:.3f}."
            )
        else:
            print(f"Preference reranker disabled: {preference_profile.reason}.")

    print("Finding highlight nuclei and phrase-aligned edits...")
    candidates = analyzer.generate_candidates(
        features,
        min_length=args.min_length,
        target_min_length=args.target_min_length,
        target_max_length=args.target_max_length,
        max_length=args.max_length,
        boundary_search_radius=args.boundary_search_radius,
        max_nuclei=args.max_nuclei,
        preference_profile=preference_profile,
        preference_weight=args.preference_weight,
    )
    selected = analyzer.select_top_clips(
        candidates,
        num_clips=args.count,
        quality_floor_ratio=args.quality_floor,
        quality_floor_absolute=args.quality_floor_absolute,
    )
    candidate_report, candidate_summary = _candidate_diagnostics(
        analyzer,
        candidates,
        selected,
        args.quality_floor,
        args.quality_floor_absolute,
    )

    print(
        f"Generated {len(candidates)} edits across "
        f"{candidate_summary['event_clusters']} musical events."
    )
    print(f"Selected {len(selected)} recommendation(s) in strict quality order:")
    for item in selected:
        label = "CHAMPION" if item["is_song_champion"] else f"#{item['selection_rank']}"
        print(
            f"  {label}: {item['start_time']:.1f}s to {item['end_time']:.1f}s "
            f"({item['duration']:.1f}s) | quality {item['quality_score']:.4f} "
            f"| confidence {item['confidence']:.2f} | raw rank {item['quality_rank']}"
        )
        print(
            f"       nucleus {item['nucleus_start']:.1f}s to "
            f"{item['nucleus_end']:.1f}s | timeline position {item['timeline_order']}"
        )

    generated_time = datetime.now(timezone.utc)
    generated_at = generated_time.isoformat().replace("+00:00", "Z")
    generation_id = generated_time.strftime("%Y%m%dT%H%M%S.%fZ")
    video_stat = video_path.stat()
    preference_metadata: dict[str, Any]
    if preference_profile is None:
        preference_metadata = {
            "enabled": False,
            "reason": "not requested; pass --preference-history to validate history",
        }
    else:
        preference_metadata = preference_profile.to_dict()

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "scoring_model_version": SCORING_MODEL_VERSION,
        "generated_at": generated_at,
        "generation_id": generation_id,
        "video": {
            "path": str(video_path.resolve()),
            "size_bytes": video_stat.st_size,
            "mtime_ns": video_stat.st_mtime_ns,
        },
        "music_path": current_track,
        "audio": {
            "duration": round(float(analyzer.duration), 3),
            "bpm": round(float(features["bpm"]), 3),
            "beat_count": len(features["beats"]),
            "beat_source": features["beat_source"],
            "beat_detected_coverage": features["beat_detected_coverage"],
            "beat_coverage": features["beat_coverage"],
            "beat_grid_extended": features["beat_grid_extended"],
            "beats_confidence": (
                round(float(features["beats_confidence"]), 4)
                if features.get("beats_confidence") is not None
                else None
            ),
            "beats_confidence_source": features.get("beats_confidence_source"),
            "essentia_beats_confidence": (
                round(float(features["essentia_beats_confidence"]), 4)
                if features.get("essentia_beats_confidence") is not None
                else None
            ),
        },
        "settings": _settings(args),
        "preference_model": preference_metadata,
        "candidate_summary": candidate_summary,
        "champion_candidate_id": selected[0].get("candidate_id") if selected else None,
        "selections": selected,
    }
    export_failures: list[dict[str, Any]] = []
    if not args.no_export:
        try:
            export_dir = _safe_descendant(output_dir, Path("exports") / generation_id)
            export_dir.mkdir(parents=True, exist_ok=False)
        except (OSError, ValueError) as exc:
            print(
                f"Failed to create isolated V3 export directory: {exc}", file=sys.stderr
            )
            return None
        print(f"Exporting V3 clips under {export_dir}...")
        for item in selected:
            rank = int(item["selection_rank"])
            champion = "_champion" if item["is_song_champion"] else ""
            clip_name = f"short_{rank}{champion}_ss{item['start_time']}_to_{item['end_time']}.mp4"
            clip_path = export_dir / clip_name
            success = export_clip(
                video_path,
                clip_path,
                float(item["start_time"]),
                float(item["end_time"]),
                fast=args.fast_export,
                cta_enabled=not args.no_cta,
                cta_duration=args.cta_duration,
                cta_scale=args.cta_scale,
            )
            item["export"] = {"status": "failed"}
            if success:
                cta_start, cta_duration = cta_timing(
                    float(item["duration"]),
                    args.cta_duration,
                )
                item["export"] = {
                    "status": "exported",
                    "path": str(clip_path.resolve()),
                    "cta": {
                        "applied": not args.no_cta,
                        "start_time": round(cta_start, 3) if not args.no_cta else None,
                        "duration": round(cta_duration, 3) if not args.no_cta else 0.0,
                        "scale": args.cta_scale if not args.no_cta else None,
                    },
                }
            else:
                export_failures.append(
                    {
                        "candidate_id": item.get("candidate_id"),
                        "selection_rank": rank,
                        "path": str(clip_path),
                    }
                )

        if export_failures:
            failure_path = export_dir / "export_failure.json"
            _atomic_write_json(
                failure_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "algorithm_version": ALGORITHM_VERSION,
                    "generated_at": generated_at,
                    "video_path": str(video_path.resolve()),
                    "failures": export_failures,
                },
            )
            print(
                f"V3 export failed for {len(export_failures)} clip(s); "
                f"published reports were left unchanged. Details: {failure_path}",
                file=sys.stderr,
            )
            return None

    candidate_payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "scoring_model_version": SCORING_MODEL_VERSION,
        "generated_at": generated_at,
        "generation_id": generation_id,
        "video_path": str(video_path.resolve()),
        "summary": candidate_summary,
        "candidates": candidate_report,
    }
    try:
        if args.no_candidate_report:
            _archive_artifact(candidates_path, output_dir, generation_id)
        else:
            _atomic_write_json(candidates_path, candidate_payload)
        # selections.json is the commit marker for a completed V3 run. Publish it last.
        _atomic_write_json(selections_path, report)
    except (OSError, ValueError) as exc:
        print(f"Failed to publish V3 reports: {exc}", file=sys.stderr)
        return None

    print(f"Saved V3 selections: {selections_path}")
    if not args.no_candidate_report:
        print(f"Saved V3 candidate diagnostics: {candidates_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        print(f"Outputs directory not found: {outputs_dir}", file=sys.stderr)
        return 1

    videos = sorted(outputs_dir.glob("**/music_video.mp4"))
    if not videos:
        print(f"No music_video.mp4 files found under {outputs_dir}.")
        return 0
    if args.latest_only:
        videos = [max(videos, key=lambda path: path.stat().st_mtime)]

    failures = 0
    for index, video_path in enumerate(videos, 1):
        print(f"\n[{index}/{len(videos)}] {video_path.parent.name}")
        if process_video(video_path, args) is None:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
