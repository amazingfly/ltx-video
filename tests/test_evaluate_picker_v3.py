from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_picker_v3 import evaluate_run, evaluate_run_detailed, summarize


class EvaluatePickerV3Tests(unittest.TestCase):
    def test_evaluates_event_rank_from_historical_choice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            chosen_dir = run_dir / "shorts" / "1This"
            v3_dir = run_dir / "shorts" / "v3"
            chosen_dir.mkdir(parents=True)
            v3_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps({"music": {"path": "/music/track.flac"}}),
                encoding="utf-8",
            )
            (run_dir / "shorts" / "selections.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "algorithm_version": "3.0.0",
                        "scoring_model_version": "test-model",
                        "generated_at": "2026-07-13T00:00:00Z",
                        "generation_id": "test-generation",
                        "video": {"path": str((run_dir / "music_video.mp4").resolve())},
                        "selections": [
                            {"start_time": 10.0, "end_time": 30.0},
                            {"start_time": 50.0, "end_time": 80.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (chosen_dir / "short_2_ss50_to_80.mp4").write_bytes(b"")
            (v3_dir / "selections.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "algorithm_version": "3.0.0",
                        "scoring_model_version": "test-model",
                        "generated_at": "2026-07-13T00:00:00Z",
                        "generation_id": "test-generation",
                        "video": {"path": str((run_dir / "music_video.mp4").resolve())},
                        "selections": [
                            {
                                "start_time": 0.0,
                                "end_time": 60.0,
                                "nucleus_start": 10.0,
                                "nucleus_end": 20.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (v3_dir / "candidates.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "algorithm_version": "3.0.0",
                        "scoring_model_version": "test-model",
                        "generated_at": "2026-07-13T00:00:00Z",
                        "generation_id": "test-generation",
                        "video_path": str((run_dir / "music_video.mp4").resolve()),
                        "candidates": [
                            {
                                "start_time": 0.0,
                                "end_time": 60.0,
                                "nucleus_start": 10.0,
                                "nucleus_end": 20.0,
                                "quality_score": 0.9,
                                "quality_rank": 1,
                                "is_event_champion": True,
                            },
                            {
                                "start_time": 40.0,
                                "end_time": 100.0,
                                "nucleus_start": 58.0,
                                "nucleus_end": 70.0,
                                "quality_score": 0.8,
                                "quality_rank": 2,
                                "is_event_champion": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            row = evaluate_run(run_dir, tolerance=0.0)

            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row.matched_event_rank, 2)
            self.assertEqual(row.closest_distance, 0.0)
            summary = summarize([row])
            self.assertEqual(summary["event_hit_at_1"], 0.0)
            self.assertEqual(summary["event_hit_at_3"], 1.0)

            candidate_path = v3_dir / "candidates.json"
            candidate_report = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate_report["generated_at"] = "2026-07-13T00:00:01Z"
            candidate_path.write_text(json.dumps(candidate_report), encoding="utf-8")

            mismatched, reason = evaluate_run_detailed(run_dir, tolerance=0.0)
            self.assertIsNone(mismatched)
            self.assertEqual(reason, "paired-report metadata mismatch: generated_at")


if __name__ == "__main__":
    unittest.main()
