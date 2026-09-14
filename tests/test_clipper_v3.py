from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from antigravityPicker.analysis_v3 import MusicVideoAnalyzerV3


def candidate(
    start_time: float,
    end_time: float,
    quality_score: float,
    **overrides: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "quality_score": quality_score,
        "combined_score": quality_score,
        "core_score": quality_score,
        "advanced_score": quality_score,
        "breakdown": {},
    }
    result.update(overrides)
    return result


class ClipperV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = MusicVideoAnalyzerV3.__new__(MusicVideoAnalyzerV3)

    def test_refine_clip_boundaries_moves_both_cuts_to_phrase_boundaries(self) -> None:
        ds_fps = 10.0
        time_axis = np.arange(1000) / ds_fps
        rms = np.full(1000, 0.15)
        rms[190:620] = 0.9
        rms[620:] = 0.1
        onset = np.zeros(1000)
        novelty = np.zeros(1000)
        onset[[190, 620]] = 1.0
        novelty[[190, 620]] = 1.0
        features = {
            "ds_fps": ds_fps,
            "time_axis": time_axis,
            "rms": rms,
            "onset": onset,
            "novelty": novelty,
            "beats": np.array([19.0, 62.0]),
        }

        refined_start, refined_end = self.analyzer.refine_clip_boundaries(
            start_idx=200,
            end_idx=600,
            features=features,
            min_length=38.0,
            max_length=46.0,
            search_radius=4.0,
        )

        self.assertEqual(refined_start, 190)
        self.assertEqual(refined_end, 620)
        self.assertGreaterEqual((refined_end - refined_start) / ds_fps, 38.0)
        self.assertLessEqual((refined_end - refined_start) / ds_fps, 46.0)

    def test_cluster_events_groups_alternate_edits_of_the_same_moment(self) -> None:
        candidates = [
            candidate(10.0, 40.0, 0.99),
            candidate(12.0, 42.0, 0.97),
            candidate(80.0, 110.0, 0.94),
            candidate(83.0, 113.0, 0.91),
            candidate(150.0, 180.0, 0.88),
        ]

        clusters = self.analyzer.cluster_events(
            candidates,
            event_overlap_threshold=0.5,
            nucleus_tolerance=8.0,
        )

        clustered_starts = sorted(
            sorted(float(item["start_time"]) for item in cluster)
            for cluster in clusters
        )
        self.assertEqual(
            clustered_starts,
            [[10.0, 12.0], [80.0, 83.0], [150.0]],
        )

    def test_cluster_events_suppresses_same_edit_with_different_nuclei(self) -> None:
        candidates = [
            candidate(
                100.0,
                160.0,
                0.95,
                nucleus_start=108.0,
                nucleus_end=120.0,
            ),
            candidate(
                101.0,
                161.0,
                0.94,
                nucleus_start=140.0,
                nucleus_end=152.0,
            ),
            candidate(
                200.0,
                260.0,
                0.90,
                nucleus_start=220.0,
                nucleus_end=232.0,
            ),
        ]

        clusters = self.analyzer.cluster_events(candidates)

        self.assertEqual(len(clusters), 2)
        self.assertEqual([item["start_time"] for item in clusters[0]], [100.0, 101.0])

    def test_selection_keeps_one_event_champion_in_strict_quality_order(self) -> None:
        candidates = [
            candidate(150.0, 180.0, 0.99, combined_score=0.10),
            candidate(151.0, 181.0, 0.98, combined_score=1.00),
            candidate(20.0, 50.0, 0.96),
            candidate(90.0, 120.0, 0.90),
        ]

        selected = self.analyzer.select_top_clips(
            candidates,
            num_clips=3,
            quality_floor_ratio=0.0,
        )

        self.assertEqual(
            [clip["start_time"] for clip in selected],
            [150.0, 20.0, 90.0],
        )
        self.assertEqual(
            [clip["quality_score"] for clip in selected],
            sorted((clip["quality_score"] for clip in selected), reverse=True),
        )
        self.assertNotIn(151.0, [clip["start_time"] for clip in selected])
        self.assertEqual([clip["quality_rank"] for clip in selected], [1, 3, 4])

    def test_quality_floor_returns_fewer_than_the_requested_quota(self) -> None:
        candidates = [
            candidate(10.0, 40.0, 1.00),
            candidate(70.0, 100.0, 0.86),
            candidate(130.0, 160.0, 0.79),
            candidate(190.0, 220.0, 0.55),
        ]

        selected = self.analyzer.select_top_clips(
            candidates,
            num_clips=5,
            quality_floor_ratio=0.85,
            quality_floor_absolute=0.80,
        )

        self.assertEqual(
            [clip["quality_score"] for clip in selected],
            [1.00, 0.86],
        )

    def test_selected_clips_include_rank_confidence_and_champion_metadata(self) -> None:
        candidates = [
            candidate(
                80.0,
                110.0,
                0.94,
                nucleus_start=91.0,
                nucleus_end=103.0,
                nucleus_score=0.97,
            ),
            candidate(
                20.0,
                50.0,
                0.90,
                nucleus_start=29.0,
                nucleus_end=41.0,
                nucleus_score=0.93,
            ),
        ]

        selected = self.analyzer.select_top_clips(
            candidates,
            num_clips=5,
            quality_floor_ratio=0.0,
        )

        self.assertEqual([clip["selection_rank"] for clip in selected], [1, 2])
        self.assertEqual([clip["quality_rank"] for clip in selected], [1, 2])
        self.assertEqual(len({clip["event_id"] for clip in selected}), 2)
        for clip in selected:
            self.assertTrue(clip["is_champion"])
            self.assertIsInstance(clip["confidence"], float)
            self.assertGreaterEqual(clip["confidence"], 0.0)
            self.assertLessEqual(clip["confidence"], 1.0)
            self.assertIn("nucleus_start", clip)
            self.assertIn("nucleus_end", clip)
            self.assertIn("nucleus_score", clip)

    def test_empty_zero_quota_and_below_floor_fallbacks_are_deterministic(self) -> None:
        self.assertEqual(self.analyzer.select_top_clips([], num_clips=5), [])
        self.assertEqual(
            self.analyzer.select_top_clips(
                [candidate(10.0, 40.0, 0.9)],
                num_clips=0,
            ),
            [],
        )

        candidates = [
            candidate(80.0, 110.0, 0.50),
            candidate(20.0, 50.0, 0.45),
        ]
        selected = self.analyzer.select_top_clips(
            candidates,
            num_clips=5,
            quality_floor_ratio=0.9,
            quality_floor_absolute=0.8,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["start_time"], 80.0)
        self.assertEqual(selected[0]["quality_rank"], 1)
        self.assertEqual(selected[0]["selection_rank"], 1)

    def test_combined_score_is_supported_as_a_legacy_fallback(self) -> None:
        candidates = [
            {
                "start_time": 80.0,
                "end_time": 110.0,
                "duration": 30.0,
                "combined_score": 0.92,
                "breakdown": {},
            },
            {
                "start_time": 20.0,
                "end_time": 50.0,
                "duration": 30.0,
                "combined_score": 0.88,
                "breakdown": {},
            },
        ]

        selected = self.analyzer.select_top_clips(
            candidates,
            num_clips=2,
            quality_floor_ratio=0.0,
        )

        self.assertEqual([clip["start_time"] for clip in selected], [80.0, 20.0])
        self.assertEqual([clip["quality_score"] for clip in selected], [0.92, 0.88])

    def test_incomplete_essentia_beats_are_replaced_for_the_full_track(self) -> None:
        self.analyzer.audio = np.zeros(1000)
        self.analyzer.target_sr = 100
        features = {
            "duration": 100.0,
            "time_axis": np.arange(1000) / 10.0,
            "beats": np.arange(0.5, 40.0, 0.5),
            "bpm": 120.0,
            "beat_density": np.zeros(1000),
        }
        repaired_beats = np.arange(0.5, 99.0, 0.5)

        with (
            patch(
                "antigravityPicker.analysis_v3.librosa.onset.onset_strength",
                return_value=np.ones(200),
            ),
            patch(
                "antigravityPicker.analysis_v3.librosa.beat.beat_track",
                return_value=(np.array([120.0]), np.arange(len(repaired_beats))),
            ),
            patch(
                "antigravityPicker.analysis_v3.librosa.frames_to_time",
                return_value=repaired_beats,
            ),
        ):
            self.analyzer._repair_incomplete_beats(features)

        self.assertEqual(features["beat_source"], "librosa_coverage_fallback")
        self.assertIsNone(features["beats_confidence"])
        self.assertEqual(features["beats_confidence_source"], "unavailable_for_librosa")
        self.assertFalse(features["beat_grid_extended"])
        self.assertGreater(features["beat_coverage"], 0.95)
        np.testing.assert_array_equal(features["beats"], repaired_beats)
        self.assertEqual(len(features["beat_density"]), len(features["time_axis"]))

        extended = self.analyzer._complete_beat_grid(
            np.arange(20.0, 60.5, 0.5),
            duration=100.0,
        )
        self.assertLess(extended[0], 0.5)
        self.assertGreater(extended[-1], 99.5)
        sparse_local = np.array([40.0, 40.5, 41.0, 41.5])
        np.testing.assert_array_equal(
            self.analyzer._complete_beat_grid(sparse_local, duration=100.0),
            sparse_local,
        )

    def test_worse_beat_fallback_is_rejected_before_grid_extension(self) -> None:
        self.analyzer.audio = np.zeros(1000)
        self.analyzer.target_sr = 100
        original_beats = np.arange(10.0, 70.5, 0.5)
        features = {
            "duration": 100.0,
            "time_axis": np.arange(1000) / 10.0,
            "beats": original_beats,
            "bpm": 120.0,
            "beats_confidence": 0.75,
            "beat_density": np.zeros(1000),
        }

        with (
            patch(
                "antigravityPicker.analysis_v3.librosa.onset.onset_strength",
                return_value=np.ones(20),
            ),
            patch(
                "antigravityPicker.analysis_v3.librosa.beat.beat_track",
                return_value=(np.array([100.0]), np.arange(4)),
            ),
            patch(
                "antigravityPicker.analysis_v3.librosa.frames_to_time",
                return_value=np.array([40.0, 40.6, 41.2, 41.8]),
            ),
        ):
            self.analyzer._repair_incomplete_beats(features)

        self.assertEqual(features["beat_source"], "essentia_grid_extended")
        self.assertEqual(features["beats_confidence"], 0.75)
        self.assertEqual(features["beats_confidence_source"], "essentia_detected_beats")
        self.assertLess(features["beat_detected_coverage"], features["beat_coverage"])
        self.assertTrue(np.any(np.isclose(features["beats"], original_beats[0])))

    def test_nonfinite_candidate_scores_are_not_selected(self) -> None:
        candidates = [
            candidate(10.0, 40.0, math.nan),
            candidate(50.0, 80.0, math.inf),
        ]

        self.assertEqual(self.analyzer.select_top_clips(candidates), [])

    def test_generates_phrase_aligned_candidates_around_compact_nuclei(self) -> None:
        ds_fps = 10.0
        n_frames = 900
        time_axis = np.arange(n_frames) / ds_fps
        rms = np.full(n_frames, 0.20)
        rms[180:360] = 0.78
        rms[580:760] = 0.92
        onset = np.full(n_frames, 0.10)
        onset[::5] = 0.85
        novelty = np.zeros(n_frames)
        novelty[[180, 360, 580, 760]] = 1.0
        hook = np.full(n_frames, 0.20)
        hook[190:350] = 0.78
        hook[590:750] = 0.95
        beats = np.arange(0.5, time_axis[-1], 0.5)
        chroma = np.tile(np.eye(12), (1, (n_frames + 11) // 12))[:, :n_frames]
        boundary = novelty.copy()

        features = {
            "duration": float(time_axis[-1]),
            "ds_fps": ds_fps,
            "time_axis": time_axis,
            "beats": beats,
            "bpm": 120.0,
            "beats_confidence": 1.0,
            "rms": rms,
            "bass": rms * 0.8,
            "onset": onset,
            "beat_density": np.ones(n_frames),
            "flux": onset,
            "centroid": np.full(n_frames, 0.5),
            "flatness": np.full(n_frames, 0.2),
            "synth": rms * 0.7,
            "vocal": rms * 0.6,
            "melody_salience": hook * 0.8,
            "repeatability": hook * 0.9,
            "novelty": novelty,
            "chroma": chroma,
            "mfcc": np.zeros((13, n_frames)),
            "v3_rms": rms,
            "v3_onset": onset,
            "v3_melody_salience": hook * 0.8,
            "hook_curve": hook,
            "boundary_strength": boundary,
            "downbeats": beats[::4],
            "phrase_boundaries": beats[::16],
        }
        self.analyzer.duration = float(time_axis[-1])

        candidates = self.analyzer.generate_candidates(
            features,
            min_length=20.0,
            target_min_length=24.0,
            target_max_length=30.0,
            max_length=35.0,
            max_nuclei=6,
        )
        selected = self.analyzer.select_top_clips(
            candidates,
            num_clips=3,
            quality_floor_ratio=0.0,
        )

        self.assertGreater(len(candidates), 0)
        self.assertGreater(len(selected), 0)
        for item in candidates:
            self.assertGreaterEqual(item["duration"], 20.0)
            self.assertLessEqual(item["duration"], 35.0)
            self.assertLessEqual(item["start_time"], item["nucleus_start"])
            self.assertGreaterEqual(item["end_time"], item["nucleus_end"])
            self.assertIn("event_score", item)
            self.assertIn("edit_score", item)
        self.assertEqual(
            [item["quality_score"] for item in selected],
            sorted((item["quality_score"] for item in selected), reverse=True),
        )

        short_candidates = self.analyzer.generate_candidates(
            features,
            min_length=3.0,
            target_min_length=4.0,
            target_max_length=5.0,
            max_length=6.0,
            max_nuclei=3,
        )
        self.assertGreater(len(short_candidates), 0)
        for item in short_candidates:
            self.assertLessEqual(item["duration"], 6.0)
            self.assertLessEqual(item["start_time"], item["nucleus_start"])
            self.assertGreaterEqual(item["end_time"], item["nucleus_end"])


if __name__ == "__main__":
    unittest.main()
