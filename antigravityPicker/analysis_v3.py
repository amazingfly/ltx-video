from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import librosa
import numpy as np

from antigravityPicker.analysis import MusicVideoAnalyzer


ALGORITHM_VERSION = "3.0.0"
SCORING_MODEL_VERSION = "event-and-edit-quality-v3"


def _clip01(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, 0.0, 1.0)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return default
    return result if math.isfinite(result) else default


def _safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    result = float(np.nanmean(values))
    return result if math.isfinite(result) else default


def _robust_scale(
    values: np.ndarray, low: float = 10.0, high: float = 90.0
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array.copy()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)
    lower, upper = np.percentile(finite, [low, high])
    if upper - lower < 1e-8:
        return np.zeros_like(array)
    return np.asarray(
        _clip01((np.nan_to_num(array, nan=lower) - lower) / (upper - lower))
    )


def _interval_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    overlap = max(
        0.0,
        min(float(left["end_time"]), float(right["end_time"]))
        - max(float(left["start_time"]), float(right["start_time"])),
    )
    if overlap <= 0.0:
        return 0.0
    shorter = max(
        1e-6,
        min(
            float(
                left.get(
                    "duration", float(left["end_time"]) - float(left["start_time"])
                )
            ),
            float(
                right.get(
                    "duration", float(right["end_time"]) - float(right["start_time"])
                )
            ),
        ),
    )
    return overlap / shorter


class MusicVideoAnalyzerV3(MusicVideoAnalyzer):
    """Quality-first music highlight picker.

    V3 reuses the proven audio decoding and low-level feature extraction from the
    original analyzer, but candidate generation and set selection are independent.
    """

    algorithm_version = ALGORITHM_VERSION

    def analyze(self) -> dict[str, Any]:
        features = super().analyze()
        self._repair_incomplete_beats(features)

        ds_fps = float(features["ds_fps"])
        for name in (
            "rms",
            "bass",
            "onset",
            "centroid",
            "flatness",
            "vocal",
            "melody_salience",
            "repeatability",
            "novelty",
        ):
            features[f"v3_{name}"] = _robust_scale(
                np.asarray(features[name], dtype=float)
            )

        features["sequence_repeatability"] = self._compute_sequence_repeatability(
            np.asarray(features["chroma"], dtype=float),
            ds_fps,
        )
        features["pulse_clarity"] = self._compute_pulse_clarity(features)
        features["boundary_strength"] = self._compute_boundary_strength(features)
        downbeats, phrase_boundaries = self._infer_metric_grid(features)
        features["downbeats"] = downbeats
        features["phrase_boundaries"] = phrase_boundaries

        hook_curve = (
            0.40 * features["sequence_repeatability"]
            + 0.30 * features["v3_melody_salience"]
            + 0.15 * features["pulse_clarity"]
            + 0.15 * features["v3_rms"]
        )
        features["hook_curve"] = np.asarray(_clip01(hook_curve), dtype=float)
        features["analysis_version"] = self.algorithm_version
        return features

    def _repair_incomplete_beats(self, features: dict[str, Any]) -> None:
        duration = float(features["duration"])
        beats = np.asarray(features.get("beats", []), dtype=float)
        beats = beats[np.isfinite(beats)]
        beats = np.unique(beats[(beats >= 0.0) & (beats <= duration)])
        essentia_confidence = features.get("beats_confidence")

        coverage = 0.0
        if beats.size >= 2 and duration > 0:
            coverage = max(0.0, float(beats[-1] - beats[0]) / duration)
        complete = bool(
            beats.size >= 4
            and beats[0] <= min(10.0, duration * 0.10)
            and beats[-1] >= max(0.0, duration - 10.0)
            and coverage >= 0.85
        )

        source = "essentia"
        if not complete:
            hop_length = 512
            onset_envelope = librosa.onset.onset_strength(
                y=self.audio,
                sr=self.target_sr,
                hop_length=hop_length,
            )
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=self.target_sr,
                hop_length=hop_length,
            )
            repaired = librosa.frames_to_time(
                np.asarray(beat_frames),
                sr=self.target_sr,
                hop_length=hop_length,
            )
            repaired = repaired[np.isfinite(repaired)]
            repaired = np.unique(repaired[(repaired >= 0.0) & (repaired <= duration)])
            repaired_coverage = (
                max(0.0, float(repaired[-1] - repaired[0]) / duration)
                if repaired.size >= 2 and duration > 0
                else 0.0
            )
            minimum_repair_coverage = max(0.60, coverage + 0.05)
            if repaired.size >= 4 and repaired_coverage >= minimum_repair_coverage:
                beats = repaired
                features["bpm"] = _finite_float(
                    tempo, _finite_float(features.get("bpm"))
                )
                source = "librosa_coverage_fallback"

        detected_coverage = (
            max(0.0, float(beats[-1] - beats[0]) / duration)
            if beats.size >= 2 and duration > 0
            else 0.0
        )
        grid_extended = False
        if beats.size >= 4:
            needs_grid_extension = bool(
                beats[0] > min(10.0, duration * 0.10)
                or beats[-1] < max(0.0, duration - 10.0)
                or (beats[-1] - beats[0]) / max(duration, 1e-6) < 0.85
            )
            if needs_grid_extension:
                extended = self._complete_beat_grid(beats, duration)
                if extended.size > beats.size:
                    beats = extended
                    source = f"{source}_grid_extended"
                    grid_extended = True

        if beats.size >= 2 and duration > 0:
            coverage = max(0.0, float(beats[-1] - beats[0]) / duration)
        else:
            coverage = 0.0

        features["beats"] = beats
        features["beat_source"] = source
        features["essentia_beats_confidence"] = essentia_confidence
        if source.startswith("librosa"):
            # Librosa does not expose a confidence value comparable to Essentia's.
            features["beats_confidence"] = None
            features["beats_confidence_source"] = "unavailable_for_librosa"
        else:
            features["beats_confidence"] = essentia_confidence
            features["beats_confidence_source"] = "essentia_detected_beats"
        features["beat_detected_coverage"] = round(detected_coverage, 4)
        features["beat_coverage"] = round(coverage, 4)
        features["beat_grid_extended"] = grid_extended
        features["beat_density"] = self._beat_density_curve(
            beats,
            np.asarray(features["time_axis"], dtype=float),
        )

    @staticmethod
    def _complete_beat_grid(beats: np.ndarray, duration: float) -> np.ndarray:
        # A short local pulse can be regular but is not enough evidence to project
        # a metrical grid over an entire song.
        if beats.size < 12 or float(beats[-1] - beats[0]) < 4.0:
            return beats
        intervals = np.diff(beats)
        plausible = intervals[(intervals >= 0.20) & (intervals <= 2.50)]
        if plausible.size < max(4, int(math.ceil(0.60 * intervals.size))):
            return beats
        interval = float(np.median(plausible))
        completed = list(float(value) for value in beats)

        value = float(beats[0]) - interval
        while value >= 0.0:
            completed.append(value)
            value -= interval
        value = float(beats[-1]) + interval
        while value <= duration:
            completed.append(value)
            value += interval

        original = sorted(completed)
        filled: list[float] = [original[0]]
        for right in original[1:]:
            left = filled[-1]
            gap = right - left
            if gap > 1.75 * interval:
                missing = max(0, int(round(gap / interval)) - 1)
                filled.extend(
                    left + interval * index for index in range(1, missing + 1)
                )
            filled.append(right)
        result = np.asarray(filled, dtype=float)
        return np.unique(result[(result >= 0.0) & (result <= duration)])

    @staticmethod
    def _beat_density_curve(beats: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
        if beats.size == 0 or time_axis.size == 0:
            return np.zeros_like(time_axis)
        half_width = 3.0
        left = np.searchsorted(beats, time_axis - half_width, side="left")
        right = np.searchsorted(beats, time_axis + half_width, side="right")
        density = (right - left).astype(float) / (2.0 * half_width)
        return _robust_scale(density)

    @staticmethod
    def _compute_sequence_repeatability(
        chroma: np.ndarray, ds_fps: float
    ) -> np.ndarray:
        n_frames = chroma.shape[1] if chroma.ndim == 2 else 0
        if n_frames < 2:
            return np.zeros(n_frames, dtype=float)

        stride = max(1, int(round(ds_fps / 2.0)))
        sampled = chroma[:, ::stride].T
        norms = np.linalg.norm(sampled, axis=1, keepdims=True)
        sampled = sampled / (norms + 1e-8)
        n_sampled = sampled.shape[0]
        exclude = max(1, int(round(15.0 * ds_fps / stride)))
        smooth = max(2, int(round(4.0 * ds_fps / stride)))
        recurrence = np.zeros(n_sampled, dtype=float)
        kernel = np.ones(smooth, dtype=float) / smooth

        for lag in range(exclude, n_sampled):
            similarity = np.sum(sampled[:-lag] * sampled[lag:], axis=1)
            if similarity.size == 0:
                continue
            active_kernel = kernel
            if similarity.size < smooth:
                active_kernel = np.ones(similarity.size, dtype=float) / similarity.size
            sequence_similarity = np.convolve(similarity, active_kernel, mode="same")
            recurrence[:-lag] = np.maximum(recurrence[:-lag], sequence_similarity)
            recurrence[lag:] = np.maximum(recurrence[lag:], sequence_similarity)

        sampled_times = np.arange(n_sampled, dtype=float) * stride / ds_fps
        full_times = np.arange(n_frames, dtype=float) / ds_fps
        expanded = np.interp(full_times, sampled_times, recurrence)
        return _robust_scale(expanded)

    @staticmethod
    def _compute_pulse_clarity(features: dict[str, Any]) -> np.ndarray:
        onset = np.asarray(features["v3_onset"], dtype=float)
        ds_fps = float(features["ds_fps"])
        try:
            pulse = librosa.beat.plp(
                onset_envelope=onset,
                sr=ds_fps,
                hop_length=1,
                tempo_min=45.0,
                tempo_max=300.0,
            )
            if len(pulse) == len(onset) and np.any(np.isfinite(pulse)):
                return _robust_scale(np.asarray(pulse, dtype=float))
        except (ValueError, FloatingPointError):
            pass

        pulse = np.zeros_like(onset)
        beat_indices = np.rint(np.asarray(features.get("beats", [])) * ds_fps).astype(
            int
        )
        beat_indices = beat_indices[(beat_indices >= 0) & (beat_indices < len(pulse))]
        pulse[beat_indices] = 1.0
        if pulse.size:
            kernel = np.asarray([0.15, 0.45, 1.0, 0.45, 0.15])
            pulse = np.convolve(pulse, kernel, mode="same")
        return np.asarray(_clip01(pulse), dtype=float)

    @staticmethod
    def _compute_boundary_strength(features: dict[str, Any]) -> np.ndarray:
        rms = np.asarray(features["v3_rms"], dtype=float)
        chroma = np.asarray(features["chroma"], dtype=float)
        novelty = np.asarray(features["v3_novelty"], dtype=float)
        ds_fps = float(features["ds_fps"])
        n_frames = len(rms)
        radius = max(1, int(round(2.0 * ds_fps)))
        energy_change = np.zeros(n_frames, dtype=float)
        chroma_change = np.zeros(n_frames, dtype=float)

        for idx in range(1, n_frames - 1):
            left_start = max(0, idx - radius)
            right_end = min(n_frames, idx + radius)
            left_rms = rms[left_start:idx]
            right_rms = rms[idx:right_end]
            if left_rms.size and right_rms.size:
                energy_change[idx] = abs(_safe_mean(right_rms) - _safe_mean(left_rms))

            left_chroma = chroma[:, left_start:idx]
            right_chroma = chroma[:, idx:right_end]
            if left_chroma.size and right_chroma.size:
                left_mean = np.mean(left_chroma, axis=1)
                right_mean = np.mean(right_chroma, axis=1)
                denominator = np.linalg.norm(left_mean) * np.linalg.norm(right_mean)
                if denominator > 1e-8:
                    cosine = float(np.dot(left_mean, right_mean) / denominator)
                    chroma_change[idx] = max(0.0, 1.0 - cosine)

        boundary = (
            0.50 * novelty
            + 0.30 * _robust_scale(energy_change)
            + 0.20 * _robust_scale(chroma_change)
        )
        return np.asarray(_clip01(boundary), dtype=float)

    @staticmethod
    def _infer_metric_grid(features: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        beats = np.asarray(features.get("beats", []), dtype=float)
        if beats.size < 4:
            return beats.copy(), beats.copy()

        ds_fps = float(features["ds_fps"])
        boundary = np.asarray(features["boundary_strength"], dtype=float)
        onset = np.asarray(features["v3_onset"], dtype=float)

        def strength_at(times: np.ndarray) -> float:
            indices = np.rint(times * ds_fps).astype(int)
            indices = indices[(indices >= 0) & (indices < len(boundary))]
            if indices.size == 0:
                return 0.0
            return _safe_mean(0.7 * boundary[indices] + 0.3 * onset[indices])

        beat_phase = max(
            range(4), key=lambda phase: (strength_at(beats[phase::4]), -phase)
        )
        downbeats = beats[beat_phase::4]
        if downbeats.size < 4:
            return downbeats, downbeats.copy()
        phrase_phase = max(
            range(4),
            key=lambda phase: (strength_at(downbeats[phase::4]), -phase),
        )
        return downbeats, downbeats[phrase_phase::4]

    @staticmethod
    def _nearest_alignment(time: float, points: np.ndarray, tolerance: float) -> float:
        if points.size == 0:
            return 0.0
        position = int(np.searchsorted(points, time))
        distances = []
        if position < points.size:
            distances.append(abs(float(points[position]) - time))
        if position > 0:
            distances.append(abs(float(points[position - 1]) - time))
        distance = min(distances) if distances else tolerance
        return float(max(0.0, 1.0 - distance / max(tolerance, 1e-6)))

    def _metric_alignment(self, idx: int, features: dict[str, Any]) -> float:
        time_axis = np.asarray(features["time_axis"], dtype=float)
        safe_idx = max(0, min(idx, len(time_axis) - 1))
        time = float(time_axis[safe_idx])
        beats = np.asarray(features.get("beats", []), dtype=float)
        downbeats = np.asarray(features.get("downbeats", []), dtype=float)
        phrases = np.asarray(features.get("phrase_boundaries", []), dtype=float)
        beat_score = self._nearest_alignment(time, beats, 0.30)
        if downbeats.size == 0:
            return beat_score
        downbeat_score = self._nearest_alignment(time, downbeats, 0.45)
        phrase_score = (
            self._nearest_alignment(time, phrases, 0.65) if phrases.size else 0.0
        )
        return float(0.20 * beat_score + 0.50 * downbeat_score + 0.30 * phrase_score)

    def _start_boundary_score(self, idx: int, features: dict[str, Any]) -> float:
        rms = np.asarray(features.get("v3_rms", features["rms"]), dtype=float)
        onset = np.asarray(features.get("v3_onset", features["onset"]), dtype=float)
        boundary = np.asarray(
            features.get("boundary_strength", features["novelty"]), dtype=float
        )
        ds_fps = float(features["ds_fps"])
        safe_idx = max(0, min(idx, len(rms) - 1))
        radius = max(1, int(round(2.0 * ds_fps)))
        before = rms[max(0, safe_idx - radius) : safe_idx]
        after = rms[safe_idx : min(len(rms), safe_idx + radius)]
        before_mean = _safe_mean(before, _safe_mean(after))
        after_mean = _safe_mean(after, before_mean)
        energy_rise = float(_clip01(0.5 + 1.25 * (after_mean - before_mean)))
        return float(
            _clip01(
                0.30 * self._metric_alignment(safe_idx, features)
                + 0.25 * boundary[safe_idx]
                + 0.25 * energy_rise
                + 0.20 * onset[safe_idx]
            )
        )

    def _end_boundary_score(self, idx: int, features: dict[str, Any]) -> float:
        rms = np.asarray(features.get("v3_rms", features["rms"]), dtype=float)
        onset = np.asarray(features.get("v3_onset", features["onset"]), dtype=float)
        boundary = np.asarray(
            features.get("boundary_strength", features["novelty"]), dtype=float
        )
        ds_fps = float(features["ds_fps"])
        safe_idx = max(0, min(idx, len(rms) - 1))
        radius = max(1, int(round(2.0 * ds_fps)))
        before = rms[max(0, safe_idx - radius) : safe_idx]
        after = rms[safe_idx : min(len(rms), safe_idx + radius)]
        before_mean = _safe_mean(before, _safe_mean(after))
        after_mean = _safe_mean(after, before_mean)
        release = float(_clip01(0.5 + 1.25 * (before_mean - after_mean)))
        final_hit = float(onset[safe_idx]) * release
        return float(
            _clip01(
                0.35 * self._metric_alignment(safe_idx, features)
                + 0.30 * boundary[safe_idx]
                + 0.25 * release
                + 0.10 * final_hit
            )
        )

    def refine_clip_boundaries(
        self,
        start_idx: int,
        end_idx: int,
        features: dict[str, Any],
        min_length: float,
        max_length: float,
        search_radius: float = 4.0,
    ) -> tuple[int, int]:
        ds_fps = float(features["ds_fps"])
        n_frames = len(features["time_axis"])
        if n_frames == 0:
            return 0, 0
        radius = max(1, int(round(search_radius * ds_fps)))
        nominal_start = max(0, min(int(start_idx), n_frames - 2))
        nominal_end = max(nominal_start + 1, min(int(end_idx), n_frames - 1))
        start_range = range(
            max(0, nominal_start - radius),
            min(n_frames - 1, nominal_start + radius) + 1,
        )
        end_range = range(
            max(1, nominal_end - radius), min(n_frames - 1, nominal_end + radius) + 1
        )

        start_options = sorted(
            start_range,
            key=lambda idx: (
                self._start_boundary_score(idx, features),
                -abs(idx - nominal_start),
            ),
            reverse=True,
        )[:16]
        end_options = sorted(
            end_range,
            key=lambda idx: (
                self._end_boundary_score(idx, features),
                -abs(idx - nominal_end),
            ),
            reverse=True,
        )[:16]

        target_duration = (nominal_end - nominal_start) / ds_fps
        best: tuple[float, float, int, int] | None = None
        for candidate_start in start_options:
            start_score = self._start_boundary_score(candidate_start, features)
            for candidate_end in end_options:
                duration = (candidate_end - candidate_start) / ds_fps
                if duration < min_length - 1e-6 or duration > max_length + 1e-6:
                    continue
                end_score = self._end_boundary_score(candidate_end, features)
                closeness = max(
                    0.0,
                    1.0
                    - abs(duration - target_duration) / max(search_radius * 2.0, 1e-6),
                )
                joint = (
                    0.40 * start_score
                    + 0.40 * end_score
                    + 0.14 * min(start_score, end_score)
                    + 0.06 * closeness
                )
                key = (
                    joint,
                    -float(
                        abs(candidate_start - nominal_start)
                        + abs(candidate_end - nominal_end)
                    ),
                    -candidate_start,
                    -candidate_end,
                )
                if best is None or key > best:
                    best = key

        if best is None:
            return nominal_start, nominal_end
        return int(-best[2]), int(-best[3])

    @staticmethod
    def _duration_fit(
        duration: float,
        min_length: float,
        target_min_length: float,
        target_max_length: float,
        max_length: float,
    ) -> float:
        if target_min_length <= duration <= target_max_length:
            return 1.0
        if duration < target_min_length:
            return float(
                _clip01(
                    (duration - min_length) / max(target_min_length - min_length, 1e-6)
                )
            )
        return float(
            _clip01((max_length - duration) / max(max_length - target_max_length, 1e-6))
        )

    def detect_highlight_nuclei(
        self,
        features: dict[str, Any],
        nucleus_durations: Sequence[float] = (8.0, 12.0, 16.0, 20.0),
        step_duration: float = 0.5,
        max_nuclei: int = 36,
        min_center_separation: float = 4.0,
    ) -> list[dict[str, Any]]:
        ds_fps = float(features["ds_fps"])
        time_axis = np.asarray(features["time_axis"], dtype=float)
        n_frames = len(time_axis)
        if n_frames < 2 or max_nuclei <= 0:
            return []

        hook = np.asarray(features["hook_curve"], dtype=float)
        rms = np.asarray(features["v3_rms"], dtype=float)
        step_frames = max(1, int(round(step_duration * ds_fps)))
        proposals: list[dict[str, Any]] = []

        for requested_duration in nucleus_durations:
            length = max(2, int(round(float(requested_duration) * ds_fps)))
            if length >= n_frames:
                continue
            duration_prior = float(_clip01(1.0 - abs(requested_duration - 12.0) / 12.0))
            for start_idx in range(0, n_frames - length, step_frames):
                end_idx = start_idx + length
                w_hook = hook[start_idx:end_idx]
                w_rms = rms[start_idx:end_idx]
                if w_hook.size == 0:
                    continue
                pre_start = max(0, start_idx - int(round(6.0 * ds_fps)))
                pre_energy = _safe_mean(rms[pre_start:start_idx], _safe_mean(w_rms))
                early_frames = max(1, min(len(w_rms), int(round(4.0 * ds_fps))))
                drop_lift = float(
                    _clip01(
                        0.5 + 1.25 * (_safe_mean(w_rms[:early_frames]) - pre_energy)
                    )
                )
                thirds = np.array_split(w_rms, 3)
                early = _safe_mean(thirds[0])
                middle = _safe_mean(thirds[1])
                late = _safe_mean(thirds[2])
                payoff = float(
                    _clip01(
                        0.55 * (0.5 + 1.25 * (max(middle, late) - early))
                        + 0.45 * (0.5 + 1.25 * (max(early, middle, late) - pre_energy))
                    )
                )
                low_energy = float(np.mean(w_rms < 0.15))
                instability = _safe_mean(np.abs(np.diff(w_hook))) * 2.0
                start_boundary = self._start_boundary_score(start_idx, features)
                mean_hook = _safe_mean(w_hook)
                upper_hook = float(np.percentile(w_hook, 75))
                score = float(
                    _clip01(
                        0.42 * mean_hook
                        + 0.18 * upper_hook
                        + 0.14 * start_boundary
                        + 0.12 * drop_lift
                        + 0.10 * payoff
                        + 0.04 * duration_prior
                        - 0.08 * low_energy
                        - 0.04 * instability
                    )
                )
                proposals.append(
                    {
                        "nucleus_start_idx": start_idx,
                        "nucleus_end_idx": end_idx,
                        "nucleus_start": round(float(time_axis[start_idx]), 3),
                        "nucleus_end": round(float(time_axis[end_idx]), 3),
                        "nucleus_duration": round(
                            float(time_axis[end_idx] - time_axis[start_idx]), 3
                        ),
                        "nucleus_score": score,
                        "breakdown": {
                            "mean_hook": mean_hook,
                            "upper_hook": upper_hook,
                            "start_boundary": start_boundary,
                            "drop_lift": drop_lift,
                            "payoff": payoff,
                            "low_energy": low_energy,
                            "instability": float(instability),
                        },
                    }
                )

        accepted: list[dict[str, Any]] = []
        proposals.sort(
            key=lambda item: (
                -float(item["nucleus_score"]),
                float(item["nucleus_start"]),
                float(item["nucleus_end"]),
            )
        )
        for proposal in proposals:
            center = (
                float(proposal["nucleus_start"]) + float(proposal["nucleus_end"])
            ) / 2.0
            duplicate = False
            for existing in accepted:
                existing_center = (
                    float(existing["nucleus_start"]) + float(existing["nucleus_end"])
                ) / 2.0
                overlap = max(
                    0.0,
                    min(float(proposal["nucleus_end"]), float(existing["nucleus_end"]))
                    - max(
                        float(proposal["nucleus_start"]),
                        float(existing["nucleus_start"]),
                    ),
                )
                shorter = max(
                    1e-6,
                    min(
                        float(proposal["nucleus_duration"]),
                        float(existing["nucleus_duration"]),
                    ),
                )
                if (
                    abs(center - existing_center) < min_center_separation
                    or overlap / shorter >= 0.70
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            proposal = dict(proposal)
            proposal["nucleus_id"] = f"nucleus_{len(accepted) + 1:03d}"
            accepted.append(proposal)
            if len(accepted) >= max_nuclei:
                break
        return accepted

    @staticmethod
    def _padding_risk(rms: np.ndarray) -> float:
        if rms.size == 0:
            return 1.0
        low = rms < 0.15
        fraction = float(np.mean(low))
        longest = 0
        current = 0
        for is_low in low:
            current = current + 1 if is_low else 0
            longest = max(longest, current)
        longest_fraction = longest / len(low)
        return float(_clip01(0.65 * fraction + 0.35 * longest_fraction))

    def _score_candidate(
        self,
        start_idx: int,
        end_idx: int,
        nucleus: dict[str, Any],
        features: dict[str, Any],
        min_length: float,
        target_min_length: float,
        target_max_length: float,
        max_length: float,
    ) -> dict[str, Any]:
        ds_fps = float(features["ds_fps"])
        time_axis = np.asarray(features["time_axis"], dtype=float)
        hook = np.asarray(features["hook_curve"], dtype=float)
        rms = np.asarray(features["v3_rms"], dtype=float)
        onset = np.asarray(features["v3_onset"], dtype=float)
        nucleus_start_idx = int(nucleus["nucleus_start_idx"])
        nucleus_end_idx = int(nucleus["nucleus_end_idx"])
        duration = float(time_axis[end_idx] - time_axis[start_idx])

        w_hook = hook[start_idx:end_idx]
        w_rms = rms[start_idx:end_idx]
        opening_end = min(end_idx, start_idx + max(1, int(round(3.0 * ds_fps))))
        opening_onset_end = min(end_idx, start_idx + max(1, int(round(2.0 * ds_fps))))
        opening_hook = float(
            _clip01(
                0.55 * _safe_mean(hook[start_idx:opening_end])
                + 0.25 * float(np.percentile(onset[start_idx:opening_onset_end], 90))
                + 0.20 * self._start_boundary_score(start_idx, features)
            )
        )
        window_hook = float(_clip01(0.55 * opening_hook + 0.45 * _safe_mean(w_hook)))

        nucleus_energy = _safe_mean(rms[nucleus_start_idx:nucleus_end_idx])
        pre_energy = _safe_mean(rms[start_idx:nucleus_start_idx], nucleus_energy)
        post_energy = _safe_mean(rms[nucleus_end_idx:end_idx], nucleus_energy)
        lift = float(_clip01(0.5 + 1.25 * (nucleus_energy - pre_energy)))
        resolution = float(_clip01(0.5 + 1.25 * (nucleus_energy - post_energy)))
        measured_arc = float(_clip01(0.65 * lift + 0.35 * resolution))

        start_boundary = self._start_boundary_score(start_idx, features)
        end_boundary = self._end_boundary_score(end_idx, features)
        boundary_quality = math.sqrt(max(0.0, start_boundary * end_boundary))
        duration_fit = self._duration_fit(
            duration,
            min_length,
            target_min_length,
            target_max_length,
            max_length,
        )
        nucleus_center = (nucleus_start_idx + nucleus_end_idx) / 2.0
        relative_position = (nucleus_center - start_idx) / max(1.0, end_idx - start_idx)
        position_fit = float(_clip01(1.0 - abs(relative_position - 0.35) / 0.45))
        song_position = float(
            _clip01(
                1.0 - abs(nucleus_center / max(1.0, len(time_axis) - 1) - 0.50) / 0.50
            )
        )
        padding_risk = self._padding_risk(w_rms)
        cta_frames = max(1, int(round(5.0 * ds_fps)))
        cta_start = max(start_idx, end_idx - cta_frames)
        cta_rise = max(
            0.0,
            _safe_mean(rms[cta_start:end_idx])
            - _safe_mean(rms[max(start_idx, cta_start - cta_frames) : cta_start]),
        )
        nucleus_in_cta = max(0, nucleus_end_idx - cta_start) / max(
            1, nucleus_end_idx - nucleus_start_idx
        )
        cta_conflict = float(_clip01(0.6 * nucleus_in_cta + 0.8 * cta_rise))

        event_score = float(
            _clip01(
                0.60 * float(nucleus["nucleus_score"])
                + 0.10 * measured_arc
                + 0.15 * boundary_quality
                + 0.15 * song_position
            )
        )
        edit_score = float(
            _clip01(
                0.30 * window_hook
                + 0.25 * boundary_quality
                + 0.15 * duration_fit
                + 0.15 * position_fit
                + 0.10 * (1.0 - padding_risk)
                + 0.05 * (1.0 - cta_conflict)
            )
        )
        structure_score = float(_clip01(0.85 * event_score + 0.15 * edit_score))

        legacy = super().score_window(
            start_idx,
            end_idx,
            features,
            min_length=min_length,
            target_min_length=target_min_length,
            target_max_length=target_max_length,
            max_length=max_length,
        )
        legacy_breakdown = dict(legacy.get("breakdown", {}))
        legacy_breakdown.update(
            {
                "v3_nucleus": round(float(nucleus["nucleus_score"]), 4),
                "v3_window_hook": round(window_hook, 4),
                "v3_opening_hook": round(opening_hook, 4),
                "v3_measured_arc": round(measured_arc, 4),
                "v3_start_boundary": round(start_boundary, 4),
                "v3_end_boundary": round(end_boundary, 4),
                "v3_boundary_quality": round(boundary_quality, 4),
                "v3_duration_fit": round(duration_fit, 4),
                "v3_nucleus_position": round(position_fit, 4),
                "v3_song_position": round(song_position, 4),
                "v3_padding_risk": round(padding_risk, 4),
                "v3_cta_conflict": round(cta_conflict, 4),
                "v3_event_score": round(event_score, 4),
                "v3_edit_score": round(edit_score, 4),
                "v3_structure_score": round(structure_score, 4),
            }
        )
        return {
            "candidate_id": "",
            "start_time": round(float(time_axis[start_idx]), 3),
            "end_time": round(float(time_axis[end_idx]), 3),
            "duration": round(duration, 3),
            "nucleus_id": nucleus["nucleus_id"],
            "nucleus_start": float(nucleus["nucleus_start"]),
            "nucleus_end": float(nucleus["nucleus_end"]),
            "nucleus_score": round(float(nucleus["nucleus_score"]), 4),
            "event_score": round(event_score, 4),
            "edit_score": round(edit_score, 4),
            "structure_score": round(structure_score, 4),
            "preference_score": 0.5,
            "quality_score": round(structure_score, 4),
            "combined_score": round(structure_score, 4),
            "legacy_combined_score": legacy.get("combined_score", 0.0),
            "core_score": legacy.get("core_score", 0.0),
            "advanced_score": legacy.get("advanced_score", 0.0),
            "breakdown": legacy_breakdown,
        }

    def generate_candidates(
        self,
        features: dict[str, Any],
        min_length: float = 45.0,
        target_min_length: float = 55.0,
        target_max_length: float = 65.0,
        max_length: float = 75.0,
        boundary_search_radius: float = 4.0,
        max_nuclei: int = 36,
        preference_profile: Any | None = None,
        preference_weight: float = 0.15,
    ) -> list[dict[str, Any]]:
        n_frames = len(features["time_axis"])
        ds_fps = float(features["ds_fps"])
        if n_frames < 2:
            return []
        available = (n_frames - 1) / ds_fps
        effective_max = min(float(max_length), available)
        effective_min = min(float(min_length), effective_max)
        effective_target_min = min(
            max(float(target_min_length), effective_min), effective_max
        )
        effective_target_max = min(
            max(float(target_max_length), effective_target_min),
            effective_max,
        )
        if effective_max <= 0.0:
            return []

        nucleus_durations = tuple(
            duration
            for duration in (8.0, 12.0, 16.0, 20.0)
            if duration <= effective_max + 1e-6
        )
        if not nucleus_durations:
            nucleus_durations = (effective_max,)
        nuclei = self.detect_highlight_nuclei(
            features,
            nucleus_durations=nucleus_durations,
            max_nuclei=max_nuclei,
        )
        if not nuclei:
            fallback_end = min(n_frames - 1, max(1, int(round(effective_min * ds_fps))))
            nuclei = [
                {
                    "nucleus_id": "nucleus_001",
                    "nucleus_start_idx": 0,
                    "nucleus_end_idx": fallback_end,
                    "nucleus_start": 0.0,
                    "nucleus_end": round(fallback_end / ds_fps, 3),
                    "nucleus_duration": round(fallback_end / ds_fps, 3),
                    "nucleus_score": 0.0,
                    "breakdown": {},
                }
            ]

        durations = sorted(
            {
                round(value, 3)
                for value in (
                    effective_min,
                    effective_target_min,
                    (effective_target_min + effective_target_max) / 2.0,
                    effective_target_max,
                    effective_max,
                )
                if effective_min - 1e-6 <= value <= effective_max + 1e-6
            }
        )
        placements = (0.25, 0.40, 0.55)
        seen: set[tuple[int, int, str]] = set()
        candidates: list[dict[str, Any]] = []

        for nucleus in nuclei:
            nucleus_start_idx = int(nucleus["nucleus_start_idx"])
            nucleus_end_idx = int(nucleus["nucleus_end_idx"])
            nucleus_center = (nucleus_start_idx + nucleus_end_idx) / 2.0
            for duration in durations:
                length = max(1, int(round(duration * ds_fps)))
                if length < nucleus_end_idx - nucleus_start_idx:
                    continue
                for placement in placements:
                    nominal_start = int(round(nucleus_center - placement * length))
                    nominal_start = min(nominal_start, nucleus_start_idx)
                    nominal_start = max(nominal_start, nucleus_end_idx - length)
                    nominal_start = max(0, min(nominal_start, n_frames - length - 1))
                    nominal_end = min(n_frames - 1, nominal_start + length)
                    refined_start, refined_end = self.refine_clip_boundaries(
                        nominal_start,
                        nominal_end,
                        features,
                        min_length=effective_min,
                        max_length=effective_max,
                        search_radius=boundary_search_radius,
                    )
                    if (
                        refined_start > nucleus_start_idx
                        or refined_end < nucleus_end_idx
                    ):
                        refined_start, refined_end = nominal_start, nominal_end
                    if (
                        refined_start > nucleus_start_idx
                        or refined_end < nucleus_end_idx
                    ):
                        continue
                    actual_duration = (refined_end - refined_start) / ds_fps
                    if (
                        actual_duration < effective_min - 1e-6
                        or actual_duration > effective_max + 1e-6
                    ):
                        continue
                    key = (refined_start, refined_end, str(nucleus["nucleus_id"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        self._score_candidate(
                            refined_start,
                            refined_end,
                            nucleus,
                            features,
                            min_length=effective_min,
                            target_min_length=effective_target_min,
                            target_max_length=effective_target_max,
                            max_length=effective_max,
                        )
                    )

        candidates.sort(
            key=lambda item: (
                -float(item["structure_score"]),
                float(item["start_time"]),
                float(item["end_time"]),
            )
        )
        for index, candidate in enumerate(candidates, 1):
            candidate["candidate_id"] = f"candidate_{index:04d}"

        profile_enabled = bool(getattr(preference_profile, "enabled", False))
        if candidates and profile_enabled and preference_weight > 0.0:
            preference_scores = np.asarray(
                preference_profile.score_candidates(candidates),
                dtype=float,
            )
            weight = float(_clip01(preference_weight))
            for candidate, preference_score in zip(
                candidates, preference_scores, strict=True
            ):
                preference_score = float(
                    _clip01(_finite_float(preference_score, default=0.5))
                )
                quality = (1.0 - weight) * float(
                    candidate["structure_score"]
                ) + weight * preference_score
                candidate["preference_score"] = round(preference_score, 4)
                candidate["quality_score"] = round(quality, 4)
                candidate["combined_score"] = round(quality, 4)

        qualities = np.asarray(
            [float(item["quality_score"]) for item in candidates], dtype=float
        )
        if qualities.size:
            order = np.argsort(np.argsort(qualities, kind="stable"), kind="stable")
            percentiles = order / max(1, len(qualities) - 1)
            for candidate, percentile in zip(candidates, percentiles, strict=True):
                candidate["quality_percentile"] = round(float(percentile), 4)
        return candidates

    def score_windows(
        self, features: dict[str, Any], **kwargs: Any
    ) -> list[dict[str, Any]]:
        aliases = {
            "end_search_radius": "boundary_search_radius",
            "max_dynamic_anchors": "max_nuclei",
        }
        supported = {
            "min_length",
            "target_min_length",
            "target_max_length",
            "max_length",
            "boundary_search_radius",
            "max_nuclei",
            "preference_profile",
            "preference_weight",
        }
        normalized: dict[str, Any] = {}
        for key, value in kwargs.items():
            normalized_key = aliases.get(key, key)
            if normalized_key in supported:
                normalized[normalized_key] = value
        return self.generate_candidates(features, **normalized)

    @staticmethod
    def _candidate_quality(candidate: dict[str, Any]) -> float:
        return _finite_float(
            candidate.get("quality_score", candidate.get("combined_score", math.nan)),
            default=math.nan,
        )

    def cluster_events(
        self,
        candidates: Iterable[dict[str, Any]],
        event_overlap_threshold: float = 0.5,
        nucleus_tolerance: float = 8.0,
    ) -> list[list[dict[str, Any]]]:
        valid_candidates = [
            item for item in candidates if math.isfinite(self._candidate_quality(item))
        ]
        ordered = sorted(
            valid_candidates,
            key=lambda item: (
                -self._candidate_quality(item),
                float(item["start_time"]),
                float(item["end_time"]),
            ),
        )
        clusters: list[list[dict[str, Any]]] = []
        for candidate in ordered:
            matched: list[dict[str, Any]] | None = None
            for cluster in clusters:
                champion = cluster[0]
                has_nuclei = all(
                    key in candidate and key in champion
                    for key in ("nucleus_start", "nucleus_end")
                )
                if has_nuclei:
                    candidate_center = (
                        float(candidate["nucleus_start"])
                        + float(candidate["nucleus_end"])
                    ) / 2.0
                    champion_center = (
                        float(champion["nucleus_start"])
                        + float(champion["nucleus_end"])
                    ) / 2.0
                    overlap = max(
                        0.0,
                        min(
                            float(candidate["nucleus_end"]),
                            float(champion["nucleus_end"]),
                        )
                        - max(
                            float(candidate["nucleus_start"]),
                            float(champion["nucleus_start"]),
                        ),
                    )
                    shorter = max(
                        1e-6,
                        min(
                            float(candidate["nucleus_end"])
                            - float(candidate["nucleus_start"]),
                            float(champion["nucleus_end"])
                            - float(champion["nucleus_start"]),
                        ),
                    )
                    same_nucleus = (
                        abs(candidate_center - champion_center) <= nucleus_tolerance
                        or overlap / shorter >= event_overlap_threshold
                    )
                    same_edit = _interval_overlap_ratio(candidate, champion) >= 0.85
                    same_event = same_nucleus or same_edit
                else:
                    same_event = (
                        _interval_overlap_ratio(candidate, champion)
                        >= event_overlap_threshold
                    )
                if same_event:
                    matched = cluster
                    break
            if matched is None:
                clusters.append([candidate])
            else:
                matched.append(candidate)
        return clusters

    def select_top_clips(
        self,
        candidates: list[dict[str, Any]],
        num_clips: int = 5,
        quality_floor_ratio: float = 0.94,
        quality_floor_absolute: float | None = None,
        event_overlap_threshold: float = 0.5,
        nucleus_tolerance: float = 8.0,
    ) -> list[dict[str, Any]]:
        if not candidates or num_clips <= 0:
            return []

        ranked = []
        for candidate in candidates:
            quality = self._candidate_quality(candidate)
            if not math.isfinite(quality):
                continue
            copy = dict(candidate)
            copy["quality_score"] = quality
            copy["combined_score"] = quality
            copy["breakdown"] = dict(candidate.get("breakdown", {}))
            ranked.append(copy)
        ranked.sort(
            key=lambda item: (
                -float(item["quality_score"]),
                float(item["start_time"]),
                float(item["end_time"]),
            )
        )
        if not ranked:
            return []
        for quality_rank, candidate in enumerate(ranked, 1):
            candidate["quality_rank"] = quality_rank

        clusters = self.cluster_events(
            ranked,
            event_overlap_threshold=event_overlap_threshold,
            nucleus_tolerance=nucleus_tolerance,
        )
        champions = [cluster[0] for cluster in clusters]
        champions.sort(
            key=lambda item: (
                -float(item["quality_score"]),
                int(item["quality_rank"]),
            )
        )
        best_quality = float(champions[0]["quality_score"])
        floor = best_quality * max(0.0, float(quality_floor_ratio))
        if quality_floor_absolute is not None:
            floor = max(floor, float(quality_floor_absolute))
        eligible = [item for item in champions if float(item["quality_score"]) >= floor]
        if not eligible:
            eligible = champions[:1]
        chosen = eligible[:num_clips]

        timeline_positions = {
            id(candidate): timeline_order
            for timeline_order, candidate in enumerate(
                sorted(
                    chosen,
                    key=lambda item: (
                        float(item["start_time"]),
                        float(item["end_time"]),
                    ),
                ),
                1,
            )
        }
        cluster_sizes = {id(cluster[0]): len(cluster) for cluster in clusters}
        result: list[dict[str, Any]] = []
        for selection_rank, champion in enumerate(chosen, 1):
            selected = dict(champion)
            selected["breakdown"] = dict(champion.get("breakdown", {}))
            selected.setdefault("nucleus_start", float(selected["start_time"]))
            selected.setdefault("nucleus_end", float(selected["end_time"]))
            selected.setdefault("nucleus_score", float(selected["quality_score"]))
            selected["event_id"] = f"event_{champions.index(champion) + 1:03d}"
            selected["selection_rank"] = selection_rank
            selected["recommendation_rank"] = selection_rank
            selected["timeline_order"] = timeline_positions[id(champion)]
            selected["confidence"] = round(
                float(
                    _clip01(float(selected["quality_score"]) / max(best_quality, 1e-8))
                ),
                4,
            )
            selected["is_champion"] = True
            selected["is_song_champion"] = selection_rank == 1
            selected["event_alternatives"] = max(
                0, cluster_sizes.get(id(champion), 1) - 1
            )
            result.append(selected)
        return result
