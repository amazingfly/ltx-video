# Stable Audio 3 on Google Colab CLI

This directory runs selectable Stable Audio 3 music models on a Google Colab
GPU using normal Python scripts and `colab exec`. It does not use a notebook.
Generated audio is FLAC-only.

## Prerequisites

- Google Colab CLI installed and authenticated.
- A Colab account with GPU availability.
- About 16 GB of temporary VM storage for Medium dependencies, model cache, and
  output.
- Optional: a Colab secret named `HF_TOKEN` if changing from the default public
  mirror to an official gated Stability AI repository.

The active configuration is [generation_config.json](generation_config.json).
It currently selects the public `cocktailpeanut/stable-audio-3-medium` mirror
and generates one 360-second FLAC.

On the first machine run, authenticate before starting the wrapper:

```bash
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
OAUTHLIB_RELAX_TOKEN_SCOPE=1 "$COLAB_PYTHON" colab/auth_colab.py
```

This helper uses the OAuth client bundled with the installed Colab CLI, disables
Google's granular-consent mode, and refuses to save the token unless the
required `colaboratory` scope was actually granted. `run_colab.sh` invokes the
helper automatically when no saved Colab token exists.

## One-command workflow

From the project root:

```bash
chmod +x colab/run_colab.sh
./colab/run_colab.sh
```

The wrapper creates a named T4 session, opens that exact CLI runtime in the
normal Colab frontend for keep-alive, uploads the Colab files, installs pinned
dependencies, restarts the kernel, probes the GPU and model, generates audio,
downloads each completed track as an integrity-checked checkpoint, and stops
the runtime even when a step fails.
Setup removes Colab's preinstalled `torchvision` because its compiled operators
can become ABI-incompatible when the SA3-compatible PyTorch version is pinned;
SA3 does not use vision features.

Keep the automatically opened Colab tab open until the command completes.
`google-colab-cli 0.5.9` can return `USER_PROJECT_DENIED` from its detached
keep-alive RPC even with a valid bundled OAuth token. The frontend tab attaches
to the same CLI-created runtime and supplies Colab's normal web keep-alive; all
setup and generation still run through `colab exec`.

Override the session name or accelerator when needed:

```bash
COLAB_SESSION=sa3-medium COLAB_GPU=A100 ./colab/run_colab.sh
```

Use a different config or local download directory without changing the
checked-in defaults:

```bash
SA3_COLAB_CONFIG=/absolute/path/to/config.json \
SA3_COLAB_LOCAL_OUTPUT_DIR=/absolute/path/to/output \
./colab/run_colab.sh
```

The web server uses these overrides through
`scripts/run_sa3_medium_colab.py`. That adapter accepts both the legacy web
schema (`tracks[0].prompt`, `model.hf_repo`, and
`generation.seed_start`) and the native Colab schema
(`generation.prompt`, `model.repo`, and `generation.seed`). It deliberately
rejects non-Medium models and writes only FLAC output.

For web queues, the local adapter receives:

```json
{
  "queue_version": 1,
  "items": [
    {"model": {"name": "medium"}, "generation": {}, "tracks": []},
    {"model": {"name": "medium"}, "generation": {}, "tracks": []}
  ]
}
```

It normalizes that file before upload into one shared model plus a generation
list:

```json
{
  "queue_version": 1,
  "model": {"name": "medium", "repo": "..."},
  "generations": [
    {"id": "track-one", "prompt": "...", "output_name": "...flac"},
    {"id": "track-two", "prompt": "...", "output_name": "...flac"}
  ]
}
```

`launch_generate.py` starts `run_generate.py` as a detached remote worker using
a short `colab exec` call. The wrapper polls an atomic exit report and downloads
new console output every 15 seconds, so generation does not depend on one
long-lived CLI WebSocket. `run_generate.py` loads the model before entering the
generation loop. It then generates and validates each list item sequentially,
releases each audio tensor, and records `model_load_count: 1` in
`generation.json`. Dependency setup, kernel restart, probing, and runtime
shutdown each happen once for an uninterrupted queue attempt. The local adapter
retries only missing or invalid tracks if Colab prunes a runtime; completed
tracks are not regenerated.

The Colab runtime generates FLAC only. While generation is running, the wrapper
downloads each completed FLAC and its JSON report to a temporary local path,
then verifies the exact byte count and SHA-256 before accepting the checkpoint.
After the queue is complete, the local web adapter publishes each FLAC under
`musicLibrary/flac/` and uses local FFmpeg (`libvorbis`, quality 6) to create
the matching file under `musicLibrary/ogg/`. Both paths are recorded in the web
manifest.

A two-item local-adapter example is available in
`colab/queue_config.example.json`.

Resume an interrupted adapter run while preserving verified checkpoints:

```bash
python scripts/run_sa3_medium_colab.py \
  --config web_runs/configs/<job-id>.json \
  --job-id <job-id> \
  --resume-run-dir outputs/<existing-run-directory>
```

Medium uses Flash Attention 2 on Ampere, Ada, or Hopper GPUs. The checked-in
configuration enables SA3's built-in PyTorch SDPA and chunked-halo fallback for
T4 because this Colab account does not have L4/A100 entitlement. The runner
skips PyTorch flex-attention compilation on T4 because its Triton kernels do not
compile for compute capability 7.5.

## Model selection

Edit `model.name`, `model.repo`, and `model.conditioning_repo` in
`generation_config.json`.

Supported configurations:

```json
{
  "name": "small-music",
  "repo": "cocktailpeanut/stable-audio-3-small-music",
  "conditioning_repo": "cocktailpeanut/stable-audio-3-small-music",
  "max_duration_seconds": 120
}
```

```json
{
  "name": "medium",
  "repo": "cocktailpeanut/stable-audio-3-medium",
  "conditioning_repo": "cocktailpeanut/stable-audio-3-medium",
  "model_half": true,
  "low_memory_load": true,
  "max_duration_seconds": 380
}
```

Set `generation.output_name` to a `.flac` filename. Other extensions are
rejected.

## Exact manual commands

Run these commands from the project root. Keep the final `colab stop` command;
an idle GPU runtime consumes quota.

```bash
SESSION=sa3-medium

colab new -s "$SESSION" --gpu T4
colab status -s "$SESSION"
colab url -s "$SESSION" --open

printf '%s\n' \
  "from pathlib import Path" \
  "Path('/content/sa3/colab').mkdir(parents=True, exist_ok=True)" \
  "Path('/content/outputs').mkdir(parents=True, exist_ok=True)" \
  | colab exec -s "$SESSION" --timeout 60

colab upload -s "$SESSION" colab/requirements.txt /content/sa3/colab/requirements.txt
colab upload -s "$SESSION" colab/generation_config.json /content/sa3/colab/generation_config.json
colab upload -s "$SESSION" colab/setup_colab.py /content/sa3/colab/setup_colab.py
colab upload -s "$SESSION" colab/probe_colab.py /content/sa3/colab/probe_colab.py
colab upload -s "$SESSION" colab/run_generate.py /content/sa3/colab/run_generate.py
colab upload -s "$SESSION" colab/launch_generate.py /content/sa3/colab/launch_generate.py

colab exec -s "$SESSION" -f colab/setup_colab.py --timeout 1800
colab restart-kernel -s "$SESSION"
colab exec -s "$SESSION" -f colab/probe_colab.py --timeout 300
colab exec -s "$SESSION" -f colab/launch_generate.py --timeout 60

mkdir -p outputs/colab
until colab download -s "$SESSION" \
  /content/outputs/generation_exit.json \
  outputs/colab/generation_exit.json 2>/dev/null; do
  colab download -s "$SESSION" \
    /content/outputs/generation_console.log \
    outputs/colab/generation_console.log 2>/dev/null || true
  sleep 15
done

colab download -s "$SESSION" /content/outputs/sa3_medium_360s.flac \
  outputs/colab/sa3_medium_360s.flac
colab download -s "$SESSION" /content/outputs/sa3_medium_360s.json \
  outputs/colab/sa3_medium_360s.json
colab download -s "$SESSION" /content/outputs/generation.json \
  outputs/colab/generation.json
colab download -s "$SESSION" /content/outputs/probe.json \
  outputs/colab/probe.json
colab download -s "$SESSION" /content/outputs/setup.json \
  outputs/colab/setup.json
colab download -s "$SESSION" /content/outputs/generation_worker.json \
  outputs/colab/generation_worker.json

colab stop -s "$SESSION"
```

## Outputs

Remote files are created under `/content/outputs`:

- `sa3_medium_360s.flac`: generated 44.1 kHz stereo 24-bit FLAC.
- `sa3_medium_360s.json`: per-track size, SHA-256, duration, and settings.
- `generation.json`: settings, GPU information, SHA-256, duration, peak, and RMS.
- `generation_console.log`: streamed stdout/stderr from the detached worker.
- `generation_exit.json`: detached worker exit code and timestamps.
- `generation_worker.json`: detached worker PID and remote paths.
- `probe.json`: Python, Torch, CUDA, GPU, imports, model, and path checks.
- `setup.json`: setup summary.
- `generation_error.log`: traceback when generation fails.

The downloaded copies are placed under local `outputs/colab/`.
When a standalone run reuses an existing output filename, the previous FLAC
and its reports are moved under `outputs/colab/history/<timestamp>-<id>/`
instead of being overwritten.

## Generation settings

Generation settings live in `generation_config.json`. Medium supports up to 380
seconds. The current six-minute run uses FP16 allocation from model
construction onward, streaming safetensors loading, eight diffusion steps, and
chunked decoding to reduce host RAM and peak VRAM.

## Failure inspection

```bash
colab log -s sa3-gpu -n 20
colab ls -s sa3-gpu /content/outputs
colab download -s sa3-gpu \
  /content/outputs/generation_error.log \
  outputs/colab/generation_error.log
colab stop -s sa3-gpu
```
