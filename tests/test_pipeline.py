from __future__ import annotations

import random
import subprocess
import tempfile
import unittest
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ltx_music_video.cli import (
    apply_motion_generation_contract,
    choose_repeated,
    resolve_image_directory,
)
from ltx_music_video.gemma import (
    LITTLE_QUEEN_MODE_FALLBACKS,
    build_diversity_instruction,
    build_ltx_prompt,
    clean_prompt,
    fallback_prompt_for_style,
    little_queen_allows_secondary_reaction,
    little_queen_mode_for_clip,
    parse_little_queen_response,
    prepare_image,
    recent_phrase_bans,
    validate_little_queen_mode_prompt,
    validate_style_prompt,
    validate_motion_prompt,
)
from ltx_music_video.manifest import load_manifest, save_manifest
from ltx_music_video.media import (
    assemble_video_with_transitions,
    conditioned_video_is_valid,
    crossfade_timeline_duration,
    evenly_spaced_indices,
    list_images,
    required_clip_count,
    required_crossfade_clip_count,
    resolve_transition_output_fps,
    video_has_motion,
)

TOKEN_CYCLE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_token_cycle.py"
TOKEN_CYCLE_SPEC = importlib.util.spec_from_file_location(
    "colab_token_cycle",
    TOKEN_CYCLE_PATH,
)
assert TOKEN_CYCLE_SPEC and TOKEN_CYCLE_SPEC.loader
TOKEN_CYCLE = importlib.util.module_from_spec(TOKEN_CYCLE_SPEC)
TOKEN_CYCLE_SPEC.loader.exec_module(TOKEN_CYCLE)


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

    def test_image_selection_can_preserve_curated_order(self) -> None:
        images = [Path(f"/tmp/{index}.png") for index in range(3)]
        selected = choose_repeated(
            images,
            5,
            random.Random(10),
            shuffle=False,
        )
        self.assertEqual(selected, images + images[:2])

    def test_list_images_sorts_symlink_names_before_resolving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            selected = root / "selected"
            sources.mkdir()
            selected.mkdir()
            last = sources / "a.png"
            first = sources / "z.png"
            Image.new("RGB", (4, 4), "red").save(last)
            Image.new("RGB", (4, 4), "blue").save(first)
            (selected / "lq_0001.png").symlink_to(first)
            (selected / "lq_0002.png").symlink_to(last)

            self.assertEqual(list_images(selected), [first.resolve(), last.resolve()])

    def test_clean_prompt_removes_small_model_wrapping(self) -> None:
        raw = '  Image-to-video prompt: "The subject blinks; camera pushes in."  '
        self.assertEqual(clean_prompt(raw), "The subject blinks; camera pushes in.")

    def test_dynamic_effect_prompt_is_accepted(self) -> None:
        validate_motion_prompt(
            "The woman's hair blows in the wind as neon lights flare behind her "
            "and thin fog curls around her boots."
        )

    def test_contained_dance_prompt_is_accepted(self) -> None:
        validate_motion_prompt(
            "The dancer sways her hips in place as hair and skirt fabric whip "
            "while neon lights flare behind her."
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

    def test_little_queen_ltx_prompt_allows_magical_overlays(self) -> None:
        prompt = build_ltx_prompt(
            (
                "Rainbow energy compresses around the little queen's current "
                "pose, then erupts as a roaring aura column while one circular "
                "shockwave blasts across the background."
            ),
            "little-queen",
        )
        self.assertIn("anchored in its source position", prompt)
        self.assertIn("exact starting pose", prompt)
        self.assertIn("Only the described luminous transformation", prompt)
        self.assertNotIn("energy wings", prompt)
        self.assertNotIn("morphing", prompt)

    def test_little_queen_ltx_wrapper_does_not_invent_physical_details(self) -> None:
        prompt = build_ltx_prompt(
            (
                "Pink-gold light gathers around the close-framed queen's "
                "silhouette, then erupts as one circular aura wave while her "
                "fierce expression and exact pose remain unchanged."
            ),
            "little-queen",
        )

        self.assertNotIn("crown", prompt.lower())
        self.assertNotIn("dress", prompt.lower())
        self.assertNotIn("wings", prompt.lower())
        self.assertNotIn("armor", prompt.lower())
        self.assertNotIn("beam", prompt.lower())

    def test_little_queen_style_rejects_unsafe_plural_anatomy(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsafe plural anatomy"):
            validate_style_prompt(
                (
                    "Rainbow energy gathers around the queen's arms while she "
                    "braces in place, then erupts into one roaring aura column "
                    "and a circular shockwave."
                ),
                "little-queen",
            )

    def test_both_hands_require_positive_visual_evidence(self) -> None:
        prompt = (
            "The little queen charges a star core between both hands, then "
            "fires one focused rainbow beam as recoil light snaps around her "
            "silhouette."
        )
        with self.assertRaisesRegex(RuntimeError, "without verified two-hand"):
            validate_little_queen_mode_prompt(prompt, "attack")
        validate_little_queen_mode_prompt(
            prompt,
            "attack",
            two_hands_clear=True,
        )

    def test_little_queen_response_parses_hidden_hand_evidence(self) -> None:
        prompt, evidence = parse_little_queen_response(
            "TWO_HANDS_CLEAR: yes\nPROMPT: The little queen charges a star core "
            "between both hands, then fires one focused rainbow beam as recoil "
            "light snaps around her silhouette."
        )
        self.assertTrue(evidence)
        self.assertTrue(prompt.startswith("The little queen"))

    def test_little_queen_style_rejects_visual_analysis_language(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "visual-analysis language"):
            validate_style_prompt(
                (
                    "Rainbow energy gathers around the queen's shown hand and "
                    "silhouette, then erupts into one roaring aura column as a "
                    "circular shockwave blasts outward."
                ),
                "little-queen",
            )

    def test_little_queen_style_rejects_template_collapse(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "too generic"):
            validate_style_prompt(
                (
                    "The little queen performs a cute pose pulse as her dress "
                    "fabric ripples, while sparkling starlight bursts emanate "
                    "from her crown."
                ),
                "little-queen",
            )

    def test_little_queen_style_rejects_old_repetitive_phrases(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "too generic"):
            validate_style_prompt(
                (
                    "The little queen performs a contained spin as her skirt "
                    "flares and wand energy spirals around the castle tower."
                ),
                "little-queen",
            )

    def test_little_queen_diversity_instruction_rotates_and_bans_recent_phrases(self) -> None:
        recent = [
            (
                "The little queen plants her feet in a power stance as "
                "prismatic lightning crackles around her crown."
            ),
            (
                "The little queen thrusts both hands upward as effervescent "
                "rainbow energy surges around her sleeves."
            ),
        ]

        first = build_diversity_instruction("little-queen", 0, recent)
        second = build_diversity_instruction("little-queen", 1, recent)

        self.assertIn("Required creative assignment", first)
        self.assertIn("Mode: transformation", first)
        self.assertIn("battle-armor ignition", first)
        self.assertIn("Mode: power-up", second)
        self.assertIn("dragon-prism aura roar", second)
        self.assertIn("mode and high-energy payoff are mandatory", first)
        self.assertIn("energy buildup, the word 'then'", first)
        self.assertIn("Do not mention hair or fabric", first)
        self.assertIn("existing background light", first)
        self.assertIn('"power stance"', first)
        self.assertIn('"prismatic lightning"', first)
        self.assertIn('"effervescent rainbow energy"', first)
        self.assertNotEqual(first, second)

    def test_little_queen_schedule_has_exact_transformation_heavy_quota(self) -> None:
        self.assertEqual(
            Counter(little_queen_mode_for_clip(index) for index in range(190)),
            Counter(
                {
                    "transformation": 67,
                    "power-up": 57,
                    "attack": 47,
                    "environmental-spell": 19,
                }
            ),
        )

    def test_little_queen_mode_requires_buildup_connector_and_payoff(self) -> None:
        missing_buildup = (
            "The little queen remains fierce before her silhouette, then fires "
            "one focused rainbow beam as recoil light snaps backward around her."
        )
        missing_connector = (
            "The little queen charges a star core before her silhouette and "
            "fires one focused rainbow beam as recoil light snaps backward."
        )
        with self.assertRaisesRegex(RuntimeError, "lacks an energy buildup"):
            validate_little_queen_mode_prompt(missing_buildup, "attack")
        with self.assertRaisesRegex(RuntimeError, "lacks an explicit"):
            validate_little_queen_mode_prompt(missing_connector, "attack")

    def test_little_queen_wrong_mode_is_rejected(self) -> None:
        attack = (
            "The little queen charges a star core before her silhouette, then "
            "fires one focused rainbow beam as recoil light snaps backward "
            "through the surrounding aura."
        )
        validate_little_queen_mode_prompt(attack, "attack")
        with self.assertRaisesRegex(RuntimeError, "required transformation"):
            validate_little_queen_mode_prompt(attack, "transformation")

    def test_moon_prism_shield_release_is_a_valid_counterattack(self) -> None:
        prompt = (
            "Dragon-shaped aura coils around her crown and both hands, then "
            "slams outward as a forceful moon-prism shield burst with violent "
            "lightning arcs."
        )
        validate_little_queen_mode_prompt(
            prompt,
            "attack",
            two_hands_clear=True,
        )

    def test_little_queen_hair_motion_is_limited_by_clip_index(self) -> None:
        prompt = (
            "Rainbow energy compresses around the little queen as her hair "
            "streams upward, then the roaring aura erupts and blasts one "
            "circular shockwave outward."
        )
        self.assertFalse(little_queen_allows_secondary_reaction(1))
        self.assertTrue(little_queen_allows_secondary_reaction(4))
        with self.assertRaisesRegex(RuntimeError, "reserved for primary action"):
            validate_little_queen_mode_prompt(
                prompt,
                "power-up",
                clip_index=1,
            )
        validate_little_queen_mode_prompt(prompt, "power-up", clip_index=4)

    def test_every_little_queen_mode_fallback_passes_its_validator(self) -> None:
        for index in range(20):
            mode = little_queen_mode_for_clip(index)
            prompt = fallback_prompt_for_style("little-queen", index)
            self.assertEqual(prompt, LITTLE_QUEEN_MODE_FALLBACKS[mode])
            validate_motion_prompt(prompt)
            validate_style_prompt(prompt, "little-queen")
            validate_little_queen_mode_prompt(
                prompt,
                mode,
                clip_index=index,
            )

    def test_recent_phrase_bans_extracts_limited_overused_terms(self) -> None:
        bans = recent_phrase_bans(
            [
                (
                    "The little queen plants her feet in a power stance as "
                    "prismatic lightning crackles around her crown."
                )
            ]
        )

        self.assertIn("power stance", bans)
        self.assertIn("prismatic lightning", bans)
        self.assertIn("around her crown", bans)

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
        from unittest.mock import patch
        import ltx_music_video.cli as cli
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "outputs"
            fallback = Path(directory) / "output"
            fallback.mkdir()
            with patch.object(cli, "DEFAULT_IMAGE_DIR", missing), patch.object(
                cli, "IMAGE_DIR_FALLBACK", fallback
            ):
                self.assertEqual(resolve_image_directory(missing), fallback)

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

    def test_colab_token_cycles_to_next_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "token.json").write_text("A", encoding="utf-8")
            (root / "t.json").write_text("B", encoding="utf-8")
            (root / "token.third.json").write_text("C", encoding="utf-8")

            next_token = TOKEN_CYCLE.cycle_token(root)
            self.assertEqual(next_token.name, "t.json")
            self.assertEqual((root / "token.json").read_text(encoding="utf-8"), "B")

            next_token = TOKEN_CYCLE.cycle_token(root)
            self.assertEqual(next_token.name, "token.third.json")
            self.assertEqual((root / "token.json").read_text(encoding="utf-8"), "C")

            next_token = TOKEN_CYCLE.cycle_token(root)
            self.assertEqual(next_token.name, "t.json")
            self.assertEqual((root / "token.json").read_text(encoding="utf-8"), "B")


if __name__ == "__main__":
    unittest.main()
