# Gemma 4 to LTX Music Video

This pipeline:

1. Picks a random music track.
2. Uses its exact duration to select one image per two seconds.
3. Sends each image to the local Gemma 4 E4B vision server for a short motion
   prompt.
4. Records the track, image paths, prompts, seeds, and generation state in an
   atomic JSON manifest.
5. Queues pending images through the official LTX-Video 2B distilled model in
   a Colab GPU session using `colab exec`.
6. Normalizes every clip to exactly two seconds and joins the clips with the
   selected audio as an MP4.

The supplied image path used `outputs`, but the existing directory is
`/mnt/storage/projects/agentic/images/scripts/output`. The CLI automatically
uses that singular-path fallback.

## Requirements

- Local `ffmpeg`, `ffprobe`, Python 3.10+, and Pillow-compatible images.
- Gemma 4 E4B model, projector, and llama.cpp build under
  `/home/derek/projects/hug/gemma4_llamacpp`.
- Google Colab CLI 0.5.9 or compatible, authenticated with GPU access.
- Internet access inside Colab for the official LTX and text-encoder weights.

## Run

Install the local command:

```bash
python3 -m pip install -e .
```

Run everything:

```bash
ltx-music-video all
```

All tracks currently in the configured music directory are 380 seconds, so a
full run selects and generates 190 clips. The manifest is printed when
preparation begins and lives under `outputs/<timestamp>/manifest.json`.

Use a specific track:

```bash
ltx-music-video all --music /absolute/path/to/track.ogg
```

Use a faster/lower-resolution render:

```bash
ltx-music-video all --width 384 --height 576
```

LTX dimensions must be divisible by 32. The default 49 frames at 24 FPS satisfy
LTX's `8n+1` frame rule and provide just over two seconds, which FFmpeg trims to
exactly two seconds.

## Staged And Resumable

For a long run, stages are easier to inspect:

```bash
ltx-music-video prepare
ltx-music-video generate --manifest outputs/<run>/manifest.json --batch-size 0
ltx-music-video assemble --manifest outputs/<run>/manifest.json
```

Rerunning `prepare` with an existing manifest skips completed prompts. Rerunning
`generate` validates local MP4s and sends only missing clips to Colab. Colab
downloads each completed clip with a SHA-256 report, so a lost runtime does not
discard earlier work.

Use `--batch-size 0` for normal production runs. That sends every pending clip
to one remote worker, so LTX model weights are loaded once and reused across the
whole pending set while the Colab runtime remains alive. A nonzero batch size is
only for deliberately small debugging runs; it makes the worker exit after that
batch, which forces a fresh model load for the next batch.

Generation starts a detached worker inside Colab, then polls its log, status,
and completed clips. The runner also sends Colab's tunnel keep-alive ping from
the polling loop and starts an authenticated frontend keep-alive when possible.
This follows the working pattern used by the related SA3 Colab scripts and lets
a later attempt resume from clips that were already downloaded. The pipeline
does not depend on Colab CLI's older keep-alive RPC because that RPC can return
`USER_PROJECT_DENIED` for an otherwise valid account.

Each LTX clip is conditioned from the source image at frame zero only. The
worker measures decoded frame changes before accepting a result and retries
with a new seed and progressively higher image-conditioning noise when motion
is too weak. A clip must preserve the conditioned first frame and exceed both
the motion-difference and changed-pixel thresholds. Assembly repeats these
checks locally, so an effectively static clip cannot enter a final video.

The final file is `outputs/<run>/music_video.mp4`. Intermediate files include:

- `manifest.json`: source paths, prompts, seeds, and status.
- `jobs/`: one upload bundle per Colab attempt.
- `clips/`: raw LTX outputs and integrity reports.
- `normalized/`: exact two-second H.264 clips used for concatenation.

To test slower playback with soft randomized transitions without changing the
main manifest or final video:

```bash
ltx-music-video assemble-transitions \
  --manifest outputs/<run>/manifest.json \
  --playback-fps 12 \
  --output-fps 24 \
  --transition-seconds 0.5
```

This doubles each 24 FPS clip's playback duration, uses motion-compensated
frame interpolation to synthesize the missing frames back to 24 FPS, calculates
the number of clips needed after accounting for crossfade overlaps, samples
those clips evenly across the manifest, and trims the result exactly to the
music duration. The output MP4 and a reproducible JSON transition plan are
written beside the manifest.

To measure whether generated clips contain real frame-to-frame changes:

```bash
uv run python scripts/analyze_video_motion.py \
  --manifest outputs/<run>/manifest.json
```

The analyzer compares every raw Colab clip, its normalized copy, and the final
video segments. It writes `motion_analysis.json` beside the manifest and
separates visible changes from compression-level pixel drift.

## Useful Options

```text
--selection-seed N       Reproduce track-image selection
--strict-prompts         Fail instead of using the conservative fallback prompt
--regenerate-prompts     Replace all prompts and regenerate their clips
--keep-gemma-running     Leave a Gemma server started by the pipeline running
--gpu T4                 Colab accelerator request
--colab-authuser 1       Browser Google-account index for frontend keep-alive
--max-attempts 8         Maximum fresh Colab sessions for missing clips
--batch-size 0           Send all remaining clips; avoids artificial model reloads
--no-open-frontend       Disable the authenticated frontend keep-alive
```

Keep-alive controls are available through environment variables:
`COLAB_TUNNEL_KEEPALIVE=0` disables the official tunnel ping,
`COLAB_TUNNEL_KEEPALIVE_INTERVAL=45` controls its cadence, and
`COLAB_FRONTEND_KEEPALIVE=0` falls back to opening a normal browser tab instead
of the headless Playwright frontend. These reduce idle pruning, but they do not
override Colab's hard runtime, quota, or compute-unit limits.

The remote setup is pinned to official LTX-Video commit
`4b2d053057623ddd4d0a1d3e9cd28890e9ef487f` and checkpoint
`ltxv-2b-0.9.8-distilled.safetensors`.

The official configuration points at a 19 GB float32 T5-XXL text encoder. The
Colab worker loads the equivalent encoder-only T5 v1.1 XXL weights from
`city96/t5-v1_1-xxl-encoder-bf16`, converts them to FP16 for T4 compatibility,
precomputes both positive and negative prompt embeddings, then releases the
text encoder before loading the LTX transformer and VAE. The transformer and
VAE run in FP16. The latent upsampler intentionally remains FP32 because the
first-pass scheduler produces FP32 latents; forcing the upsampler to FP16
causes the convolution input/bias dtype failure seen in the earlier variants.

The pipeline uses its own Gemma launcher because the general Hug launcher has a
256-token microbatch that is too small for this multimodal projector. The local
launcher uses `--ubatch-size 2048` and records logs in
`/tmp/ltx_gemma4_vision.log`.
