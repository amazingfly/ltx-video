from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SHORT_INDEX_RE = re.compile(r"(?:^|_)short_(\d+)(?:_|\.|$)")


@dataclass(frozen=True)
class EvaluationRow:
    run: str
    track: str
    chosen_start: float
    chosen_end: float
    matched_event_rank: int | None
    closest_event_rank: int
    closest_distance: float
    champion_start: float
    champion_end: float
    champion_nucleus_start: float
    champion_nucleus_end: float
    selected_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V3 event ranking against historical shorts/1This choices."
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--event-tolerance",
        type=float,
        default=2.0,
        help="Seconds outside the chosen interval still counted as an event match.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _track_id(run_dir: Path) -> str:
    manifest = _load_json(run_dir / "manifest.json") or {}
    music = manifest.get("music", {})
    if isinstance(music, dict) and isinstance(music.get("path"), str):
        return str(Path(music["path"]).expanduser())
    return run_dir.name


def _chosen_index(run_dir: Path) -> int | None:
    chosen_dir = run_dir / "shorts" / "1This"
    if not chosen_dir.is_dir():
        return None
    indices = set()
    for path in chosen_dir.glob("*.mp4"):
        match = SHORT_INDEX_RE.search(path.name)
        if match:
            indices.add(int(match.group(1)) - 1)
    return next(iter(indices)) if len(indices) == 1 else None


def _distance_to_interval(point: float, start: float, end: float) -> float:
    if point < start:
        return start - point
    if point > end:
        return point - end
    return 0.0


def _paired_report_error(
    selections: dict[str, Any], candidates: dict[str, Any]
) -> str | None:
    for key in (
        "schema_version",
        "algorithm_version",
        "scoring_model_version",
        "generated_at",
        "generation_id",
    ):
        selection_value = selections.get(key)
        candidate_value = candidates.get(key)
        if selection_value is None or candidate_value is None:
            return f"missing paired-report metadata: {key}"
        if selection_value != candidate_value:
            return f"paired-report metadata mismatch: {key}"

    selection_video = selections.get("video")
    selection_video_path = (
        selection_video.get("path") if isinstance(selection_video, dict) else None
    )
    candidate_video_path = candidates.get("video_path")
    if not isinstance(selection_video_path, str) or not isinstance(
        candidate_video_path, str
    ):
        return "missing paired-report video identity"
    if (
        Path(selection_video_path).expanduser()
        != Path(candidate_video_path).expanduser()
    ):
        return "paired-report video mismatch"
    return None


def evaluate_run_detailed(
    run_dir: Path, tolerance: float
) -> tuple[EvaluationRow | None, str | None]:
    legacy = _load_json(run_dir / "shorts" / "selections.json")
    v3 = _load_json(run_dir / "shorts" / "v3" / "selections.json")
    candidate_report = _load_json(run_dir / "shorts" / "v3" / "candidates.json")
    chosen_index = _chosen_index(run_dir)
    if chosen_index is None:
        return None, "no unambiguous historical selection"
    if legacy is None:
        return None, "missing or invalid legacy selections report"
    if v3 is None:
        return None, "missing or invalid V3 selections report"
    if candidate_report is None:
        return None, "missing or invalid V3 candidates report"
    pair_error = _paired_report_error(v3, candidate_report)
    if pair_error is not None:
        return None, pair_error

    legacy_selections = legacy.get("selections")
    v3_selections = v3.get("selections")
    candidates = candidate_report.get("candidates")
    if not isinstance(legacy_selections, list) or chosen_index >= len(
        legacy_selections
    ):
        return None, "historical selection index is outside the legacy report"
    if not isinstance(v3_selections, list) or not v3_selections:
        return None, "V3 report has no selections"
    if not isinstance(candidates, list):
        return None, "V3 candidate list is missing"

    chosen = legacy_selections[chosen_index]
    if not isinstance(chosen, dict):
        return None, "historical selection is malformed"
    try:
        chosen_start = float(chosen["start_time"])
        chosen_end = float(chosen["end_time"])
    except (KeyError, TypeError, ValueError):
        return None, "historical selection boundaries are malformed"
    if not all(math.isfinite(value) for value in (chosen_start, chosen_end)):
        return None, "historical selection boundaries are non-finite"
    event_champions = [
        item
        for item in candidates
        if isinstance(item, dict) and bool(item.get("is_event_champion"))
    ]
    try:
        event_champions.sort(
            key=lambda item: (
                -float(item["quality_score"]),
                int(item.get("quality_rank", 10**9)),
            )
        )
    except (KeyError, TypeError, ValueError):
        return None, "event champion scores are malformed"
    if not event_champions:
        return None, "V3 candidate report has no event champions"

    ranked_distances = []
    try:
        for rank, event in enumerate(event_champions, 1):
            center = (float(event["nucleus_start"]) + float(event["nucleus_end"])) / 2.0
            if not math.isfinite(center):
                return None, "event champion nucleus is non-finite"
            ranked_distances.append(
                (rank, _distance_to_interval(center, chosen_start, chosen_end))
            )
    except (KeyError, TypeError, ValueError):
        return None, "event champion nucleus is malformed"
    matched_ranks = [
        rank for rank, distance in ranked_distances if distance <= tolerance
    ]
    closest_rank, closest_distance = min(
        ranked_distances, key=lambda item: (item[1], item[0])
    )
    champion = event_champions[0]
    try:
        row = EvaluationRow(
            run=run_dir.name,
            track=_track_id(run_dir),
            chosen_start=chosen_start,
            chosen_end=chosen_end,
            matched_event_rank=min(matched_ranks) if matched_ranks else None,
            closest_event_rank=closest_rank,
            closest_distance=closest_distance,
            champion_start=float(champion["start_time"]),
            champion_end=float(champion["end_time"]),
            champion_nucleus_start=float(champion["nucleus_start"]),
            champion_nucleus_end=float(champion["nucleus_end"]),
            selected_count=len(v3_selections),
        )
    except (KeyError, TypeError, ValueError):
        return None, "song champion is malformed"
    return row, None


def evaluate_run(run_dir: Path, tolerance: float) -> EvaluationRow | None:
    row, _reason = evaluate_run_detailed(run_dir, tolerance)
    return row


def summarize(rows: list[EvaluationRow]) -> dict[str, Any]:
    if not rows:
        return {
            "unique_tracks": 0,
            "event_hit_at_1": 0.0,
            "event_hit_at_3": 0.0,
            "event_hit_at_5": 0.0,
            "mean_reciprocal_event_rank": 0.0,
        }
    ranks = [row.matched_event_rank for row in rows]
    reciprocal = [1.0 / rank if rank is not None else 0.0 for rank in ranks]
    return {
        "unique_tracks": len(rows),
        "event_hit_at_1": sum(rank is not None and rank <= 1 for rank in ranks)
        / len(rows),
        "event_hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks)
        / len(rows),
        "event_hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks)
        / len(rows),
        "mean_reciprocal_event_rank": sum(reciprocal) / len(rows),
        "median_closest_nucleus_distance_seconds": statistics.median(
            row.closest_distance for row in rows
        ),
        "mean_selected_count": statistics.mean(row.selected_count for row in rows),
    }


def main() -> int:
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir() or not math.isfinite(args.event_tolerance):
        return 1

    rows_by_track: dict[str, EvaluationRow] = {}
    skipped: list[dict[str, str]] = []
    labeled_run_dirs = sorted(
        path.parents[1] for path in outputs_dir.glob("*/shorts/1This") if path.is_dir()
    )
    for run_dir in labeled_run_dirs:
        row, reason = evaluate_run_detailed(run_dir, max(0.0, args.event_tolerance))
        if row is not None:
            rows_by_track.setdefault(row.track, row)
        else:
            skipped.append(
                {"run": run_dir.name, "reason": reason or "unknown evaluation error"}
            )
    rows = list(rows_by_track.values())
    summary = summarize(rows)
    summary.update(
        {
            "historically_labeled_runs": len(labeled_run_dirs),
            "evaluated_runs": len(labeled_run_dirs) - len(skipped),
            "evaluated_unique_tracks": len(rows),
            "duplicate_track_runs": max(
                0, len(labeled_run_dirs) - len(skipped) - len(rows)
            ),
            "skipped_runs": len(skipped),
        }
    )

    if args.as_json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "tracks": [row.__dict__ for row in rows],
                    "skipped": skipped,
                },
                indent=2,
            )
        )
        return 0

    for row in rows:
        match = (
            str(row.matched_event_rank)
            if row.matched_event_rank is not None
            else "miss"
        )
        print(
            f"{row.run}: chosen={row.chosen_start:.1f}-{row.chosen_end:.1f}s "
            f"event_rank={match} closest={row.closest_event_rank} "
            f"distance={row.closest_distance:.1f}s champion="
            f"{row.champion_start:.1f}-{row.champion_end:.1f}s"
        )
    for item in skipped:
        print(f"{item['run']}: skipped ({item['reason']})")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
