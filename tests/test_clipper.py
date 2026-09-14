from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from antigravityPicker.analysis import MusicVideoAnalyzer

class ClipperTests(unittest.TestCase):
    def test_analyzer_end_to_end(self) -> None:
        """
        Generates a synthetic MP4 file with audio, runs the full analysis,
        scores sliding windows (both dynamic and fixed fallback),
        and checks clip selection logic.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dummy_video = root / "dummy_video.mp4"

            subprocess.run(
                [
                    "ffmpeg",
                    "-v", "error",
                    "-y",
                    "-f", "lavfi",
                    "-i", "testsrc2=size=64x64:rate=24",
                    "-f", "lavfi",
                    "-i", "sine=frequency=1000:duration=60",
                    "-t", "60",
                    "-pix_fmt", "yuv420p",
                    str(dummy_video),
                ],
                check=True,
            )

            analyzer = MusicVideoAnalyzer(str(dummy_video), target_sr=11025)
            self.assertGreater(analyzer.duration, 59.0)

            features = analyzer.analyze()
            self.assertIn("bpm", features)
            self.assertIn("rms", features)

            # Test dynamic scoring
            candidates_dyn = analyzer.score_windows(
                features,
                dynamic=True,
                min_length=15.0,
                target_min_length=20.0,
                target_max_length=30.0,
                max_length=40.0
            )
            self.assertGreater(len(candidates_dyn), 0)

            # Verify new breakdown terms exist
            bd = candidates_dyn[0]["breakdown"]
            self.assertIn("start_impact", bd)
            self.assertIn("ending_cleanliness", bd)
            self.assertIn("energy_density", bd)
            self.assertIn("arc_payoff", bd)
            self.assertIn("phrase_boundary", bd)
            self.assertIn("duration_bonus", bd)
            self.assertIn("repetition_penalty", bd)
            self.assertIn("padding_penalty", bd)
            self.assertIn("short_form_suitability", bd)

            # Test fixed fallback
            candidates_fixed = analyzer.score_windows(
                features,
                dynamic=False,
                window_duration=20.0,
                step_duration=1.0
            )
            self.assertGreater(len(candidates_fixed), 0)
            for cand in candidates_fixed:
                self.assertEqual(cand["duration"], 20.0)

    def test_soft_overlap_penalty_allows_high_value_overlap(self) -> None:
        """
        Overlap should be a penalty, not a hard rejection. A very strong
        overlapping second clip can still be selected if it beats alternatives
        after the diversity penalty.
        """
        analyzer = MusicVideoAnalyzer.__new__(MusicVideoAnalyzer)

        candidates = [
            {"start_time": 0.0, "end_time": 60.0, "duration": 60.0, "combined_score": 1.00, "core_score": 0.5, "advanced_score": 0.5, "breakdown": {}},
            {"start_time": 35.0, "end_time": 95.0, "duration": 60.0, "combined_score": 0.98, "core_score": 0.5, "advanced_score": 0.5, "breakdown": {}},
            {"start_time": 120.0, "end_time": 180.0, "duration": 60.0, "combined_score": 0.55, "core_score": 0.5, "advanced_score": 0.5, "breakdown": {}},
            {"start_time": 185.0, "end_time": 245.0, "duration": 60.0, "combined_score": 0.50, "core_score": 0.5, "advanced_score": 0.5, "breakdown": {}},
            {"start_time": 250.0, "end_time": 310.0, "duration": 60.0, "combined_score": 0.45, "core_score": 0.5, "advanced_score": 0.5, "breakdown": {}},
        ]

        selected = analyzer.select_top_clips(candidates, num_clips=5, min_gap=15.0)

        self.assertEqual(len(selected), 5)
        self.assertIn(35.0, [clip["start_time"] for clip in selected])
        overlapping = next(clip for clip in selected if clip["start_time"] == 35.0)
        self.assertGreater(overlapping["selection_penalty"], 0.0)
        self.assertGreater(overlapping["selection_overlap_ratio"], 0.0)

    def test_soft_overlap_suppresses_near_duplicates(self) -> None:
        """Near-identical windows should not fill multiple export slots."""
        analyzer = MusicVideoAnalyzer.__new__(MusicVideoAnalyzer)

        candidates = [
            {"start_time": 0.0, "end_time": 60.0, "duration": 60.0, "combined_score": 1.00, "core_score": 0.5, "advanced_score": 0.5, "breakdown": {}},
            {"start_time": 2.0, "end_time": 62.0, "duration": 60.0, "combined_score": 0.99, "core_score": 0.5, "advanced_score": 0.5, "breakdown": {}},
            {"start_time": 80.0, "end_time": 140.0, "duration": 60.0, "combined_score": 0.50, "core_score": 0.5, "advanced_score": 0.5, "breakdown": {}},
        ]

        selected = analyzer.select_top_clips(candidates, num_clips=3)

        self.assertEqual([clip["start_time"] for clip in selected], [0.0, 80.0])

    def test_champion_picker_prefers_longer_clips_via_duration_bonus(self) -> None:
        """
        Given two candidates with similar base scores covering the same
        musical moment, the longer one should win because the duration bonus
        adds a small positive weight.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dummy_video = root / "dummy_video.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v", "error",
                    "-y",
                    "-f", "lavfi",
                    "-i", "testsrc2=size=64x64:rate=24",
                    "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=60",
                    "-t", "60",
                    "-pix_fmt", "yuv420p",
                    str(dummy_video),
                ],
                check=True,
            )
            analyzer = MusicVideoAnalyzer(str(dummy_video), target_sr=11025)
            features = analyzer.analyze()
            ds_fps = features["ds_fps"]

            # Score a short and a long window starting at the same point
            start_idx = int(10.0 * ds_fps)
            short_end = start_idx + int(15.0 * ds_fps)
            long_end = start_idx + int(30.0 * ds_fps)

            short_cand = analyzer.score_window(
                start_idx, short_end, features,
                min_length=15.0, target_min_length=20.0,
                target_max_length=30.0, max_length=40.0
            )
            long_cand = analyzer.score_window(
                start_idx, long_end, features,
                min_length=15.0, target_min_length=20.0,
                target_max_length=30.0, max_length=40.0
            )

            # The longer candidate should have a positive duration bonus
            self.assertGreater(long_cand["breakdown"]["duration_bonus"],
                               short_cand["breakdown"]["duration_bonus"])

    def test_champion_picker_handles_empty_candidates(self) -> None:
        """select_top_clips should return an empty list for empty input."""
        analyzer = MusicVideoAnalyzer.__new__(MusicVideoAnalyzer)
        self.assertEqual(analyzer.select_top_clips([], num_clips=5, min_gap=15.0), [])

    def test_refine_end_index_moves_to_clean_audio_boundary(self) -> None:
        analyzer = MusicVideoAnalyzer.__new__(MusicVideoAnalyzer)
        ds_fps = 10.0
        time_axis = np.arange(1000) / ds_fps
        rms = np.full(1000, 0.5)
        rms[620:] = 0.1
        novelty = np.zeros(1000)
        novelty[620] = 1.0
        features = {
            "ds_fps": ds_fps,
            "time_axis": time_axis,
            "rms": rms,
            "onset": np.zeros(1000),
            "novelty": novelty,
            "beats": np.array([62.0]),
        }

        refined = analyzer.refine_end_index(
            start_idx=0,
            nominal_end_idx=600,
            features=features,
            min_length=45.0,
            max_length=75.0,
            search_radius=4.0,
        )

        self.assertEqual(time_axis[refined], 62.0)

    def test_overlap_chains_do_not_merge_an_entire_video(self) -> None:
        """
        Dense sliding windows form a transitive overlap chain across a video.
        Selection must suppress candidates against each chosen champion rather
        than collapsing the complete chain into one group.
        """
        analyzer = MusicVideoAnalyzer.__new__(MusicVideoAnalyzer)

        peak_scores = {20.0: 0.99, 90.0: 0.95, 160.0: 0.90, 230.0: 0.85, 300.0: 0.80}
        candidates = []
        for start in range(0, 341):
            start_time = float(start)
            candidates.append({
                "start_time": start_time,
                "end_time": start_time + 60.0,
                "duration": 60.0,
                "combined_score": peak_scores.get(start_time, 0.10),
                "core_score": 0.5,
                "advanced_score": 0.5,
                "breakdown": {},
            })

        selected = analyzer.select_top_clips(candidates, num_clips=5, min_gap=15.0)

        self.assertEqual(len(selected), 5)
        self.assertEqual(
            [clip["start_time"] for clip in selected],
            [20.0, 90.0, 160.0, 230.0, 300.0],
        )


if __name__ == "__main__":
    unittest.main()
