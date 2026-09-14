from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from antigravityPicker.preferences_v3 import FEATURE_NAMES, PreferenceProfile


def _candidate(signal: float) -> dict[str, object]:
    breakdown = {name: 0.0 for name in FEATURE_NAMES}
    breakdown["loudness"] = signal
    return {"breakdown": breakdown}


def _write_history_run(
    root: Path,
    run_name: str,
    track_path: str,
    *,
    chosen_signal: float,
    rejected_signals: tuple[float, ...] = (0.0, 0.2),
) -> None:
    run_dir = root / run_name
    chosen_dir = run_dir / "shorts" / "1This"
    chosen_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"music": {"path": track_path}}),
        encoding="utf-8",
    )
    selections = [_candidate(value) for value in rejected_signals]
    selections.insert(1, _candidate(chosen_signal))
    (run_dir / "shorts" / "selections.json").write_text(
        json.dumps({"selections": selections}),
        encoding="utf-8",
    )
    (chosen_dir / "short_2_ss10_to_20.mp4").write_bytes(b"")


class PreferenceProfileTests(unittest.TestCase):
    def test_fits_validated_pairwise_profile_and_scores_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(6):
                _write_history_run(
                    root,
                    f"run-{index}",
                    f"/music/track-{index}.flac",
                    chosen_signal=1.0,
                )

            profile = PreferenceProfile.from_history(root, min_tracks=4)
            scores = profile.score_candidates([_candidate(0.0), _candidate(1.0)])

            self.assertTrue(profile.enabled)
            self.assertIsNone(profile.reason)
            self.assertEqual(profile.track_count, 6)
            self.assertGreater(profile.validation_accuracy or 0.0, 0.55)
            self.assertEqual(scores.shape, (2,))
            self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))
            self.assertGreater(scores[1], scores[0])
            self.assertTrue(profile.to_dict()["history_fingerprint"])

    def test_deduplicates_music_paths_and_excludes_current_track(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(5):
                _write_history_run(
                    root,
                    f"run-{index}",
                    f"/music/track-{index}.flac",
                    chosen_signal=1.0,
                )
            _write_history_run(
                root,
                "duplicate",
                "/music/track-1.flac",
                chosen_signal=1.0,
            )

            profile = PreferenceProfile.from_history(
                root,
                current_track="/music/track-0.flac",
                min_tracks=4,
            )
            metadata = profile.to_dict()

            self.assertTrue(profile.enabled)
            self.assertEqual(profile.track_count, 4)
            self.assertEqual(
                metadata["diagnostics"]["reports_excluded_current_track"], 1
            )
            self.assertGreater(
                metadata["diagnostics"]["duplicate_pairwise_examples"], 0
            )

    def test_disables_when_history_is_too_small_and_scores_neutrally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_history_run(
                root,
                "run-0",
                "/music/track-0.flac",
                chosen_signal=1.0,
            )

            profile = PreferenceProfile.from_history(root, min_tracks=4)
            scores = profile.score_candidates([_candidate(0.0), _candidate(1.0)])

            self.assertFalse(profile.enabled)
            self.assertIn("requires at least 4", profile.reason or "")
            np.testing.assert_allclose(scores, [0.5, 0.5])

            validation_profile = PreferenceProfile.from_history(root, min_tracks=1)
            self.assertFalse(validation_profile.enabled)
            self.assertIn(
                "validation requires at least 2", validation_profile.reason or ""
            )

    def test_disables_when_leave_one_track_out_accuracy_is_weak(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(6):
                if index % 2 == 0:
                    chosen, rejected = 1.0, (0.0, 0.2)
                else:
                    chosen, rejected = 0.0, (0.8, 1.0)
                _write_history_run(
                    root,
                    f"run-{index}",
                    f"/music/track-{index}.flac",
                    chosen_signal=chosen,
                    rejected_signals=rejected,
                )

            profile = PreferenceProfile.from_history(root, min_tracks=4)

            self.assertFalse(profile.enabled)
            self.assertIsNotNone(profile.validation_accuracy)
            self.assertLessEqual(float(profile.validation_accuracy), 0.55)
            self.assertIn("pairwise accuracy", profile.reason or "")


if __name__ == "__main__":
    unittest.main()
