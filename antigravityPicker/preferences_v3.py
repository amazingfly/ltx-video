from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_VERSION = "pairwise-ridge-v1"
VALIDATION_THRESHOLD = 0.55

# These fields exist in both the older short reports and the current analyzer.
# Deliberate aliases such as flux/energy_density are omitted so they do not get
# counted twice in a very small preference dataset.
FEATURE_NAMES = (
    "loudness",
    "bass",
    "punchiness",
    "beat_density",
    "dynamic_contrast",
    "drop_likelihood",
    "chorus_hook",
    "novelty",
    "vocal_presence",
    "melody_salience",
    "repeatability",
    "contrast_from_prev",
    "start_impact",
    "ending_cleanliness",
    "arc_payoff",
    "phrase_boundary",
    "duration_bonus",
    "repetition_penalty",
    "padding_penalty",
    "silence_penalty",
    "intro_outro_penalty",
    "short_form_suitability",
)

_SHORT_INDEX_RE = re.compile(r"(?:^|_)short_(\d+)(?:_|\.|$)")


@dataclass
class _HistoryDiagnostics:
    reports_found: int = 0
    reports_used: int = 0
    reports_excluded_current_track: int = 0
    reports_skipped_missing_manifest_track: int = 0
    reports_skipped_invalid_json: int = 0
    reports_skipped_ambiguous_choice: int = 0
    reports_skipped_invalid_selections: int = 0
    duplicate_pairwise_examples: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "reports_found": self.reports_found,
            "reports_used": self.reports_used,
            "reports_excluded_current_track": self.reports_excluded_current_track,
            "reports_skipped_missing_manifest_track": self.reports_skipped_missing_manifest_track,
            "reports_skipped_invalid_json": self.reports_skipped_invalid_json,
            "reports_skipped_ambiguous_choice": self.reports_skipped_ambiguous_choice,
            "reports_skipped_invalid_selections": self.reports_skipped_invalid_selections,
            "duplicate_pairwise_examples": self.duplicate_pairwise_examples,
        }


@dataclass
class PreferenceProfile:
    enabled: bool
    reason: str | None
    history_dir: str
    current_track: str | None
    min_tracks: int
    ridge: float
    track_count: int
    pair_count: int
    validation_accuracy: float | None
    history_fingerprint: str
    weights: np.ndarray | None = field(default=None, repr=False)
    diagnostics: _HistoryDiagnostics = field(
        default_factory=_HistoryDiagnostics,
        repr=False,
    )

    @classmethod
    def from_history(
        cls,
        history_dir: Path,
        current_track: str | None = None,
        min_tracks: int = 10,
        ridge: float = 4.0,
    ) -> "PreferenceProfile":
        if min_tracks < 1:
            raise ValueError("min_tracks must be at least 1")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be a finite number greater than zero")

        history_dir = Path(history_dir)
        normalized_current_track = (
            _normalize_track_id(current_track) if current_track else None
        )
        diagnostics = _HistoryDiagnostics()

        if not history_dir.is_dir():
            return cls._disabled(
                history_dir,
                normalized_current_track,
                min_tracks,
                ridge,
                reason=f"history directory not found: {history_dir}",
                diagnostics=diagnostics,
            )

        comparisons_by_track: dict[str, list[np.ndarray]] = {}
        fingerprints_by_track: dict[str, set[tuple[float, ...]]] = {}

        report_paths = sorted(history_dir.glob("*/shorts/selections.json"))
        diagnostics.reports_found = len(report_paths)
        for report_path in report_paths:
            run_dir = report_path.parent.parent
            track_id = _manifest_track_id(run_dir / "manifest.json")
            if track_id is None:
                diagnostics.reports_skipped_missing_manifest_track += 1
                continue
            if normalized_current_track and track_id == normalized_current_track:
                diagnostics.reports_excluded_current_track += 1
                continue

            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                diagnostics.reports_skipped_invalid_json += 1
                continue

            chosen_index = _chosen_selection_index(run_dir / "shorts" / "1This")
            if chosen_index is None:
                diagnostics.reports_skipped_ambiguous_choice += 1
                continue

            comparisons = _pairwise_comparisons(report, chosen_index)
            if not comparisons:
                diagnostics.reports_skipped_invalid_selections += 1
                continue

            track_comparisons = comparisons_by_track.setdefault(track_id, [])
            seen = fingerprints_by_track.setdefault(track_id, set())
            for comparison in comparisons:
                fingerprint = tuple(np.round(comparison, decimals=8))
                if fingerprint in seen:
                    diagnostics.duplicate_pairwise_examples += 1
                    continue
                seen.add(fingerprint)
                track_comparisons.append(comparison)
            diagnostics.reports_used += 1

        comparisons_by_track = {
            track_id: comparisons
            for track_id, comparisons in comparisons_by_track.items()
            if comparisons
        }
        track_count = len(comparisons_by_track)
        pair_count = sum(len(rows) for rows in comparisons_by_track.values())
        history_fingerprint = _history_fingerprint(comparisons_by_track)

        if track_count < min_tracks:
            return cls._disabled(
                history_dir,
                normalized_current_track,
                min_tracks,
                ridge,
                reason=(
                    f"requires at least {min_tracks} unique preference tracks; "
                    f"found {track_count}"
                ),
                diagnostics=diagnostics,
                track_count=track_count,
                pair_count=pair_count,
                history_fingerprint=history_fingerprint,
            )
        if track_count < 2:
            return cls._disabled(
                history_dir,
                normalized_current_track,
                min_tracks,
                ridge,
                reason=(
                    "leave-one-track-out validation requires at least 2 unique "
                    f"preference tracks; found {track_count}"
                ),
                diagnostics=diagnostics,
                track_count=track_count,
                pair_count=pair_count,
                history_fingerprint=history_fingerprint,
            )

        validation_accuracy = _leave_one_track_out_accuracy(
            comparisons_by_track,
            ridge,
        )
        if validation_accuracy <= VALIDATION_THRESHOLD:
            return cls._disabled(
                history_dir,
                normalized_current_track,
                min_tracks,
                ridge,
                reason=(
                    "leave-one-track-out pairwise accuracy "
                    f"{validation_accuracy:.3f} did not exceed "
                    f"{VALIDATION_THRESHOLD:.3f}"
                ),
                diagnostics=diagnostics,
                track_count=track_count,
                pair_count=pair_count,
                validation_accuracy=validation_accuracy,
                history_fingerprint=history_fingerprint,
            )

        weights = _fit_pairwise_ridge(comparisons_by_track, ridge)
        return cls(
            enabled=True,
            reason=None,
            history_dir=str(history_dir.resolve()),
            current_track=normalized_current_track,
            min_tracks=min_tracks,
            ridge=float(ridge),
            track_count=track_count,
            pair_count=pair_count,
            validation_accuracy=validation_accuracy,
            history_fingerprint=history_fingerprint,
            weights=weights,
            diagnostics=diagnostics,
        )

    @classmethod
    def _disabled(
        cls,
        history_dir: Path,
        current_track: str | None,
        min_tracks: int,
        ridge: float,
        *,
        reason: str,
        diagnostics: _HistoryDiagnostics,
        track_count: int = 0,
        pair_count: int = 0,
        validation_accuracy: float | None = None,
        history_fingerprint: str = "",
    ) -> "PreferenceProfile":
        return cls(
            enabled=False,
            reason=reason,
            history_dir=str(history_dir.resolve()),
            current_track=current_track,
            min_tracks=min_tracks,
            ridge=float(ridge),
            track_count=track_count,
            pair_count=pair_count,
            validation_accuracy=validation_accuracy,
            history_fingerprint=history_fingerprint,
            weights=None,
            diagnostics=diagnostics,
        )

    def score_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        candidate_list = list(candidates)
        if not candidate_list:
            return np.empty(0, dtype=float)
        if not self.enabled or self.weights is None:
            return np.full(len(candidate_list), 0.5, dtype=float)

        matrix = np.vstack(
            [_permissive_feature_vector(item) for item in candidate_list]
        )
        matrix = _impute_nonfinite_columns(matrix)
        normalized = _z_normalize(matrix)
        utility = normalized @ self.weights
        utility = np.clip(utility, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-utility))

    def to_dict(self) -> dict[str, Any]:
        serialized_weights: dict[str, float] = {}
        if self.weights is not None:
            serialized_weights = {
                name: round(float(weight), 8)
                for name, weight in zip(FEATURE_NAMES, self.weights)
            }
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "model_version": MODEL_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "history_dir": self.history_dir,
            "current_track": self.current_track,
            "min_tracks": self.min_tracks,
            "ridge": self.ridge,
            "track_count": self.track_count,
            "pair_count": self.pair_count,
            "validation": {
                "method": "leave-one-track-out",
                "pairwise_accuracy": (
                    round(float(self.validation_accuracy), 6)
                    if self.validation_accuracy is not None
                    else None
                ),
                "minimum_exclusive": VALIDATION_THRESHOLD,
            },
            "history_fingerprint": self.history_fingerprint,
            "weights": serialized_weights,
            "diagnostics": self.diagnostics.to_dict(),
        }


def _normalize_track_id(track: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.expanduser(track)))


def _manifest_track_id(manifest_path: Path) -> str | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    music = manifest.get("music")
    if not isinstance(music, Mapping):
        return None
    path = music.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    return _normalize_track_id(path)


def _chosen_selection_index(chosen_dir: Path) -> int | None:
    if not chosen_dir.is_dir():
        return None
    indices: set[int] = set()
    for path in chosen_dir.iterdir():
        if not path.is_file():
            continue
        match = _SHORT_INDEX_RE.search(path.name)
        if match:
            indices.add(int(match.group(1)) - 1)
    if len(indices) != 1:
        return None
    chosen_index = next(iter(indices))
    return chosen_index if chosen_index >= 0 else None


def _strict_feature_vector(candidate: Mapping[str, Any]) -> np.ndarray | None:
    breakdown = candidate.get("breakdown")
    if not isinstance(breakdown, Mapping):
        return None
    values: list[float] = []
    for name in FEATURE_NAMES:
        value = breakdown.get(name)
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        values.append(number)
    return np.asarray(values, dtype=float)


def _permissive_feature_vector(candidate: Mapping[str, Any]) -> np.ndarray:
    breakdown = candidate.get("breakdown")
    if not isinstance(breakdown, Mapping):
        breakdown = {}
    values: list[float] = []
    for name in FEATURE_NAMES:
        value = breakdown.get(name)
        if isinstance(value, bool):
            values.append(float("nan"))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        values.append(number if math.isfinite(number) else float("nan"))
    return np.asarray(values, dtype=float)


def _pairwise_comparisons(
    report: Mapping[str, Any],
    chosen_index: int,
) -> list[np.ndarray]:
    selections = report.get("selections")
    if not isinstance(selections, list) or len(selections) < 2:
        return []
    if chosen_index >= len(selections):
        return []

    feature_rows: list[np.ndarray] = []
    for candidate in selections:
        if not isinstance(candidate, Mapping):
            return []
        features = _strict_feature_vector(candidate)
        if features is None:
            return []
        feature_rows.append(features)

    normalized = _z_normalize(np.vstack(feature_rows))
    chosen = normalized[chosen_index]
    comparisons: list[np.ndarray] = []
    for index, rejected in enumerate(normalized):
        if index == chosen_index:
            continue
        difference = chosen - rejected
        if np.linalg.norm(difference) > 1e-12:
            comparisons.append(difference)
    return comparisons


def _z_normalize(matrix: np.ndarray) -> np.ndarray:
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    safe_scales = np.where(scales > 1e-8, scales, 1.0)
    return (matrix - means) / safe_scales


def _impute_nonfinite_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype=float, copy=True)
    for column_index in range(result.shape[1]):
        column = result[:, column_index]
        finite = np.isfinite(column)
        replacement = float(np.median(column[finite])) if np.any(finite) else 0.0
        column[~finite] = replacement
    return result


def _fit_pairwise_ridge(
    comparisons_by_track: Mapping[str, Sequence[np.ndarray]],
    ridge: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    sample_weights: list[float] = []
    for track_id in sorted(comparisons_by_track):
        comparisons = comparisons_by_track[track_id]
        if not comparisons:
            continue
        weight = 1.0 / len(comparisons)
        rows.extend(comparisons)
        sample_weights.extend([weight] * len(comparisons))

    if not rows:
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    matrix = np.vstack(rows)
    square_root_weights = np.sqrt(np.asarray(sample_weights, dtype=float))
    weighted_matrix = matrix * square_root_weights[:, np.newaxis]
    targets = square_root_weights
    system = weighted_matrix.T @ weighted_matrix
    system += float(ridge) * np.eye(system.shape[0], dtype=float)
    right_hand_side = weighted_matrix.T @ targets
    try:
        return np.linalg.solve(system, right_hand_side)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(system) @ right_hand_side


def _leave_one_track_out_accuracy(
    comparisons_by_track: Mapping[str, Sequence[np.ndarray]],
    ridge: float,
) -> float:
    correct = 0.0
    total = 0
    for held_out_track in sorted(comparisons_by_track):
        training = {
            track_id: comparisons
            for track_id, comparisons in comparisons_by_track.items()
            if track_id != held_out_track
        }
        weights = _fit_pairwise_ridge(training, ridge)
        for comparison in comparisons_by_track[held_out_track]:
            margin = float(comparison @ weights)
            correct += 1.0 if margin > 0 else 0.5 if abs(margin) <= 1e-12 else 0.0
            total += 1
    return correct / total if total else 0.0


def _history_fingerprint(
    comparisons_by_track: Mapping[str, Sequence[np.ndarray]],
) -> str:
    if not comparisons_by_track:
        return ""
    digest = hashlib.sha256()
    for track_id in sorted(comparisons_by_track):
        digest.update(track_id.encode("utf-8"))
        digest.update(b"\0")
        rows = sorted(
            (
                np.asarray(row, dtype="<f8").tobytes()
                for row in comparisons_by_track[track_id]
            )
        )
        for row in rows:
            digest.update(row)
        digest.update(b"\0")
    return digest.hexdigest()
