from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from antigravityPicker.cli import build_cta_filter, export_clip, parse_args


class CtaExportTests(unittest.TestCase):
    def _cta_region(self, video_path: Path, timestamp: float) -> bytes:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v", "error",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", "crop=iw*0.9:ih*0.25:iw*0.05:ih*0.58,format=gray",
                "-f", "rawvideo",
                "-",
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout

    def _mean_pixel_delta(self, left: bytes, right: bytes) -> float:
        self.assertEqual(len(left), len(right))
        return sum(abs(a - b) for a, b in zip(left, right)) / len(left)

    def test_cli_cta_options(self) -> None:
        with patch.object(sys, "argv", ["clipper"]):
            defaults = parse_args()
        self.assertEqual(defaults.cta_duration, 5.0)
        self.assertEqual(defaults.cta_scale, 2.0)
        self.assertFalse(defaults.no_cta)

        with patch.object(
            sys,
            "argv",
            ["clipper", "--cta-duration", "3.5", "--cta-scale", "1.25", "--no-cta"],
        ):
            configured = parse_args()
        self.assertEqual(configured.cta_duration, 3.5)
        self.assertEqual(configured.cta_scale, 1.25)
        self.assertTrue(configured.no_cta)

    def test_filter_keeps_text_stable_and_pulses_only_arrow(self) -> None:
        filter_graph = build_cta_filter(1080, 1920, start_time=20.0, duration=5.0)

        self.assertIn("text='FULL TRACK'", filter_graph)
        self.assertIn("text='BELOW'", filter_graph)
        self.assertIn("text='▼'", filter_graph)
        self.assertIn("fontsize=119", filter_graph)
        self.assertEqual(filter_graph.count("alpha="), 1)
        self.assertEqual(filter_graph.count("enable="), 3)
        self.assertIn("y=998", filter_graph)

    def test_vertical_export_shows_cta_only_at_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            with_cta = root / "with_cta.mp4"
            without_cta = root / "without_cta.mp4"

            subprocess.run(
                [
                    "ffmpeg",
                    "-v", "error",
                    "-y",
                    "-f", "lavfi",
                    "-i", "color=c=0x345678:s=360x640:r=30:d=8",
                    "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=8",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-shortest",
                    str(source),
                ],
                check=True,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                success = export_clip(
                    source,
                    with_cta,
                    0.0,
                    8.0,
                    cta_enabled=True,
                    cta_duration=2.0,
                )
            self.assertTrue(success)
            self.assertIn(
                "CTA applied: yes (start: 6.00s, duration: 2.00s, scale: 2.00x)",
                output.getvalue(),
            )

            self.assertGreater(
                self._mean_pixel_delta(
                    self._cta_region(with_cta, 4.0),
                    self._cta_region(with_cta, 7.0),
                ),
                1.0,
            )
            self.assertGreater(
                self._mean_pixel_delta(
                    self._cta_region(with_cta, 6.1),
                    self._cta_region(with_cta, 6.45),
                ),
                0.01,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                success = export_clip(
                    source,
                    without_cta,
                    0.0,
                    8.0,
                    cta_enabled=False,
                )
            self.assertTrue(success)
            self.assertIn("CTA applied: no (--no-cta)", output.getvalue())
            self.assertLess(
                self._mean_pixel_delta(
                    self._cta_region(without_cta, 4.0),
                    self._cta_region(without_cta, 7.0),
                ),
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
