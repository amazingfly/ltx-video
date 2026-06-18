from __future__ import annotations

import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ltx_music_video.cli import (
    apply_motion_generation_contract,
    choose_repeated,
    resolve_image_directory,
)
from ltx_music_video.gemma import (
    build_ltx_prompt,
    clean_prompt,
    prepare_image,
    validate_motion_prompt,
)
from ltx_music_video.manifest import load_manifest, save_manifest
from ltx_music_video.media import (
    assemble_video_with_transitions,
    conditioned_video_is_valid,
    crossfade_timeline_duration,
    evenly_spaced_indices,
    required_clip_count,
    required_crossfade_clip_count,
    resolve_transition_output_fps,
    video_has_motion,
)


class PipelineTests(unittest.TestCase):
    def test_clip_count_rounds_up(self) -> None:
        self.assertEqual(required_clip_count(380.0, 2.0), 190)
        self.assertEqual(required_clip_count(5.1, 2.0), 3)

    def test_crossfade_clip_count_accounts_for_overlap(self) -> None:
        self.assertEqual(required_crossfade_clip_count(380.0, 4.0, 0.5), 109)
        self.assertEqual(crossfade_timeline_duration(109, 4.0, 0.5), 382.0)
        self.assertLess(crossfade_timeline_duration(108, 4.0, 0.5), 380.0)

    def test_transition_output_fps_defaults_to_source_fps(self) -> None:
        self.assertEqual(resolve_transition_output_fps(24, 12, None), 24)
        self.assertEqual(resolve_transition_output_fps(24, 12, 24), 24)
        with self.assertRaisesRegex(ValueError, "lower than playback"):
            resolve_transition_output_fps(24, 12, 6)

    def test_transition_assembly_interpolates_back_to_output_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = [root / f"clip_{index}.mp4" for index in range(2)]
            for clip in clips:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "testsrc2=size=64x64:rate=24",
                        "-t",
                        "0.5",
                        "-pix_fmt",
                        "yuv420p",
                        str(clip),
                    ],
                    check=True,
                )
            music = root / "music.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-t",
                    "1.0",
                    str(music),
                ],
                check=True,
            )
            output = root / "transition.mp4"
            assemble_video_with_transitions(
                clips,
                music,
                output,
                audio_duration=1.0,
                source_clip_seconds=0.5,
                source_fps=24,
                playback_fps=12,
                output_fps=24,
                transition_seconds=0.2,
                transitions=["fade"],
            )

            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.stdout.strip(), "24/1")

    def test_even_selection_covers_the_full_manifest(self) -> None:
        indices = evenly_spaced_indices(190, 109)
        self.assertEqual(len(indices), 109)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 189)
        self.assertEqual(len(set(indices)), 109)

    def test_image_selection_repeats_only_after_a_full_batch(self) -> None:
        images = [Path(f"/tmp/{index}.png") for index in range(3)]
        selected = choose_repeated(images, 5, random.Random(10))
        self.assertEqual(len(set(selected[:3])), 3)
        self.assertEqual(len(selected), 5)

    def test_clean_prompt_removes_small_model_wrapping(self) -> None:
        raw = '  Image-to-video prompt: "The subject blinks; camera pushes in."  '
        self.assertEqual(clean_prompt(raw), "The subject blinks; camera pushes in.")

    def test_dynamic_effect_prompt_is_accepted(self) -> None:
        validate_motion_prompt(
            "The woman's hair blows in the wind as neon lights flare behind her "
            "and thin fog curls around her boots."
        )

    def test_camera_motion_prompt_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "camera motion"):
            validate_motion_prompt(
                "The woman blinks while her hair sways and the camera slowly pans "
                "right across the neon background."
            )

    def test_large_body_motion_prompt_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "large body motion"):
            validate_motion_prompt(
                "The armored woman shifts her weight while neon lights pulse "
                "and thin fog curls around her boots."
            )

    def test_ltx_prompt_locks_identity_and_framing(self) -> None:
        prompt = build_ltx_prompt(
            "The woman's hair moves gently as neon light pulses and thin fog "
            "curls around her boots."
        )
        self.assertIn("camera remains locked", prompt)
        self.assertIn("identity", prompt)
        self.assertIn("clearly perceptible", prompt)
        self.assertIn("no camera movement", prompt)

    def test_motion_generation_contract_uses_start_only_conditioning(self) -> None:
        manifest = {
            "settings": {"ltx": {"prompt_contract_version": 2}},
            "clips": [
                {
                    "motion_prompt": (
                        "Her hair blows in the wind as neon light pulses behind her."
                    ),
                    "prompt": "old",
                    "status": "generated",
                }
            ],
        }
        apply_motion_generation_contract(manifest)
        settings = manifest["settings"]["ltx"]
        self.assertEqual(settings["conditioning_anchor_frames"], "start")
        self.assertEqual(settings["image_cond_noise_scale"], 0.15)
        self.assertEqual(settings["generation_attempts_per_clip"], 4)
        self.assertEqual(settings["minimum_motion_mad"], 0.5)
        self.assertEqual(manifest["clips"][0]["status"], "prompted")
        self.assertIn("clearly perceptible", manifest["clips"][0]["prompt"])

    def test_manifest_write_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            save_manifest(path, {"version": 1, "clips": []})
            loaded = load_manifest(path)
            self.assertEqual(loaded["version"], 1)
            self.assertIn("updated_at", loaded)

    def test_image_is_converted_to_bounded_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from io import BytesIO

            from PIL import Image

            source = Path(directory) / "large.png"
            Image.new("RGB", (1000, 500), "red").save(source)
            mime_type, image_bytes = prepare_image(source, max_edge=200)
            self.assertEqual(mime_type, "image/jpeg")
            converted = Image.open(BytesIO(image_bytes))
            self.assertEqual(converted.size, (200, 100))

    def test_missing_default_image_dir_uses_existing_fallback(self) -> None:
        resolved = resolve_image_directory(
            Path("/mnt/storage/projects/agentic/images/scripts/outputs")
        )
        self.assertEqual(resolved.name, "output")

    def test_noise_video_fails_conditioning_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            noise = root / "noise.png"
            video = root / "noise.mp4"
            gradient = np.tile(np.arange(96, dtype=np.uint8), (64, 1))
            source_array = np.stack((gradient, np.flipud(gradient), gradient), axis=2)
            noise_array = np.random.default_rng(1).integers(
                0, 256, size=(64, 96, 3), dtype=np.uint8
            )
            Image.fromarray(source_array).save(source)
            Image.fromarray(noise_array).save(noise)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    str(noise),
                    "-t",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )
            self.assertFalse(conditioned_video_is_valid(video, source))
            self.assertFalse(
                conditioned_video_is_valid(
                    video,
                    source,
                    sample_positions=(0.0,),
                )
            )

    def test_matching_static_video_passes_anchor_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            video = root / "source.mp4"
            gradient = np.tile(np.arange(96, dtype=np.uint8), (64, 1))
            source_array = np.stack((gradient, np.flipud(gradient), gradient), axis=2)
            Image.fromarray(source_array).save(source)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    str(source),
                    "-t",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )
            self.assertTrue(conditioned_video_is_valid(video, source))
            self.assertFalse(video_has_motion(video))

    def test_changing_video_passes_motion_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "moving.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=96x64:rate=24",
                    "-t",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )
            self.assertTrue(video_has_motion(video))


if __name__ == "__main__":
    unittest.main()
