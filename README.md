# LTX music video

Generate resumable music videos from still images and audio using a local vision
model, LTX-Video on Colab, and FFmpeg. Includes motion/identity checks, transition
assembly, and quality-ranked short selection.

## Setup

Use Python 3.10+ (3.11 recommended), FFmpeg/ffprobe, and [uv](https://docs.astral.sh/uv/).
Run from this checkout; the Colab runner and model launchers are repository assets.

```bash
uv sync --locked --group dev
uv run ltx-music-video --help
uv run antigravity-clipper-v3 --help
uv run pytest
```

Rendering also needs an authenticated Colab CLI and a local Gemma vision server.
See [usage and model setup](docs/usage.md) for requirements, retry behavior,
resuming jobs, generation settings, and short selection.

```bash
uv run ltx-music-video all --image-dir /path/to/images --music /path/to/song.ogg
# Stage retry wrapper (all arguments are forwarded):
uv run python scripts/run_full_pipeline.py --help
```

`LTX_IMAGE_DIR` and `LTX_MUSIC_DIR` override local default input directories.
`GEMMA_BASE_DIR`, `GEMMA_SERVER`, `GEMMA_MODEL`, and `GEMMA_MMPROJ` configure the
vision launcher. Existing workstation defaults remain supported.

## Layout

| Path | Responsibility |
| --- | --- |
| `ltx_music_video/` | CLI, manifests, vision prompts, generation and assembly |
| `antigravityPicker/` | Audio analysis and short selection, including V3 |
| `colab/` | LTX remote setup and workers |
| `scripts/` | LTX retries, diagnostics, model and Colab session helpers |
| `tests/` | CPU tests; FFmpeg is required for media checks |
| `docs/` | Operating instructions and repository boundaries |
| `outputs/`, `auth/` | Ignored local run artifacts and credentials |

Image training/validation belongs to `images`; music generation belongs to `sa3`.
Cross-repository orchestration and publishing belong to `media-pipeline`.
See [repository boundaries](docs/repositories.md). No generated media or model
weights are committed. This project is an application checkout, not a standalone
wheel containing the remote worker assets.

See [workflow support status](docs/workflows.md). Use `python scripts/workspace.py doctor`
to check centralized checkout/interpreter configuration, and
`python scripts/workspace.py run --component ltx -- {python} SCRIPT [ARGS]`
to launch with shared paths. Workspace setup is documented in
[media-pipeline](https://github.com/amazingfly/media-pipeline/blob/main/docs/workspace.md).
