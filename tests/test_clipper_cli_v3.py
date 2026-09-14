from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from antigravityPicker.cliV3 import main, parse_args, process_video, relative_subdir


def _fake_analyzer() -> MagicMock:
    candidate = {
        "candidate_id": "candidate_0001",
        "start_time": 20.0,
        "end_time": 80.0,
        "duration": 60.0,
        "nucleus_id": "nucleus_001",
        "nucleus_start": 24.0,
        "nucleus_end": 36.0,
        "nucleus_score": 0.9,
        "structure_score": 0.88,
        "preference_score": 0.5,
        "quality_score": 0.88,
        "combined_score": 0.88,
        "core_score": 0.0,
        "advanced_score": 0.0,
        "breakdown": {},
    }
    selected = {
        **candidate,
        "quality_rank": 1,
        "selection_rank": 1,
        "recommendation_rank": 1,
        "timeline_order": 1,
        "confidence": 1.0,
        "event_id": "event_001",
        "is_champion": True,
        "is_song_champion": True,
    }
    analyzer = MagicMock()
    analyzer.duration = 100.0
    analyzer.analyze.return_value = {
        "bpm": 120.0,
        "beats": np.arange(0.5, 99.0, 0.5),
        "beat_source": "librosa_coverage_fallback",
        "beat_detected_coverage": 0.98,
        "beat_coverage": 0.98,
        "beat_grid_extended": False,
        "beats_confidence": None,
        "beats_confidence_source": "unavailable_for_librosa",
        "essentia_beats_confidence": 0.8,
    }
    analyzer.generate_candidates.return_value = [candidate]
    analyzer.select_top_clips.return_value = [selected]
    analyzer.cluster_events.return_value = [[candidate]]
    return analyzer


class ClipperV3CliTests(unittest.TestCase):
    def test_defaults_use_isolated_v3_output_and_quality_floor(self) -> None:
        args = parse_args([])

        self.assertEqual(args.output_subdir, Path("shorts/v3"))
        self.assertEqual(args.quality_floor, 0.94)
        self.assertEqual(args.max_nuclei, 36)

    def test_output_subdir_rejects_paths_outside_v3_namespace(self) -> None:
        for value in (
            "/tmp/v3",
            "../v3",
            "shorts/../legacy",
            "shorts",
            "shorts/v2",
            "shorts/v3-other",
            "",
        ):
            with self.subTest(value=value), self.assertRaises(Exception):
                relative_subdir(value)

        self.assertEqual(relative_subdir("shorts/v3"), Path("shorts/v3"))
        self.assertEqual(
            relative_subdir("shorts/v3/experiment"),
            Path("shorts/v3/experiment"),
        )

    def test_all_float_arguments_reject_non_finite_values(self) -> None:
        flags = (
            "--min-length",
            "--target-min-length",
            "--target-max-length",
            "--max-length",
            "--boundary-search-radius",
            "--quality-floor",
            "--quality-floor-absolute",
            "--preference-weight",
            "--preference-ridge",
            "--cta-duration",
            "--cta-scale",
        )
        for flag in flags:
            for value in ("nan", "inf", "-inf"):
                with (
                    self.subTest(flag=flag, value=value),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    parse_args([flag, value])

    def test_output_symlink_cannot_escape_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            outside = root / "outside"
            (run_dir / "shorts").mkdir(parents=True)
            outside.mkdir()
            (run_dir / "shorts" / "v3").symlink_to(outside, target_is_directory=True)
            video_path = run_dir / "music_video.mp4"
            video_path.write_bytes(b"video")
            args = parse_args(["--outputs-dir", str(root), "--no-export", "--force"])

            with (
                patch("antigravityPicker.cliV3.MusicVideoAnalyzerV3") as analyzer,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                report = process_video(video_path, args)

            self.assertIsNone(report)
            analyzer.assert_not_called()
            self.assertEqual(list(outside.iterdir()), [])

    def test_no_export_writes_only_v3_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            legacy_dir = run_dir / "shorts"
            legacy_dir.mkdir(parents=True)
            video_path = run_dir / "music_video.mp4"
            video_path.write_bytes(b"video")
            legacy_report = legacy_dir / "selections.json"
            legacy_report.write_text('{"legacy": true}\n', encoding="utf-8")
            analyzer = _fake_analyzer()
            args = parse_args(
                [
                    "--outputs-dir",
                    str(root),
                    "--no-export",
                    "--force",
                ]
            )

            with (
                patch(
                    "antigravityPicker.cliV3.MusicVideoAnalyzerV3",
                    return_value=analyzer,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = process_video(video_path, args)

            self.assertIsNotNone(report)
            self.assertEqual(
                legacy_report.read_text(encoding="utf-8"), '{"legacy": true}\n'
            )
            v3_report = run_dir / "shorts" / "v3" / "selections.json"
            candidate_report = run_dir / "shorts" / "v3" / "candidates.json"
            self.assertTrue(v3_report.is_file())
            self.assertTrue(candidate_report.is_file())
            saved = json.loads(v3_report.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], 3)
            self.assertEqual(saved["champion_candidate_id"], "candidate_0001")
            self.assertEqual(saved["selections"][0]["selection_rank"], 1)
            self.assertIsNone(saved["audio"]["beats_confidence"])
            self.assertEqual(
                saved["audio"]["beats_confidence_source"],
                "unavailable_for_librosa",
            )
            self.assertEqual(saved["audio"]["essentia_beats_confidence"], 0.8)

    def test_export_failure_does_not_publish_reports_and_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            output_dir = run_dir / "shorts" / "v3"
            output_dir.mkdir(parents=True)
            video_path = run_dir / "music_video.mp4"
            video_path.write_bytes(b"video")
            selections_path = output_dir / "selections.json"
            candidates_path = output_dir / "candidates.json"
            selections_path.write_text('{"old": "selection"}\n', encoding="utf-8")
            candidates_path.write_text('{"old": "candidate"}\n', encoding="utf-8")
            args = parse_args(["--outputs-dir", str(root), "--force"])

            with (
                patch(
                    "antigravityPicker.cliV3.MusicVideoAnalyzerV3",
                    return_value=_fake_analyzer(),
                ),
                patch("antigravityPicker.cliV3.export_clip", return_value=False),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                report = process_video(video_path, args)

            self.assertIsNone(report)
            self.assertEqual(
                selections_path.read_text(encoding="utf-8"),
                '{"old": "selection"}\n',
            )
            self.assertEqual(
                candidates_path.read_text(encoding="utf-8"),
                '{"old": "candidate"}\n',
            )
            failures = list((output_dir / "exports").glob("*/export_failure.json"))
            self.assertEqual(len(failures), 1)

    def test_no_candidate_report_archives_previous_canonical_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            output_dir = run_dir / "shorts" / "v3"
            output_dir.mkdir(parents=True)
            video_path = run_dir / "music_video.mp4"
            video_path.write_bytes(b"video")
            candidates_path = output_dir / "candidates.json"
            candidates_path.write_text('{"old": true}\n', encoding="utf-8")
            args = parse_args(
                [
                    "--outputs-dir",
                    str(root),
                    "--no-export",
                    "--no-candidate-report",
                    "--force",
                ]
            )

            with (
                patch(
                    "antigravityPicker.cliV3.MusicVideoAnalyzerV3",
                    return_value=_fake_analyzer(),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = process_video(video_path, args)

            self.assertIsNotNone(report)
            self.assertFalse(candidates_path.exists())
            archived = list((output_dir / "history").glob("*/candidates.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_text(encoding="utf-8"), '{"old": true}\n')

    def test_main_returns_nonzero_when_processing_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "music_video.mp4").write_bytes(b"video")

            with (
                patch("antigravityPicker.cliV3.process_video", return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = main(["--outputs-dir", str(root)])

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
