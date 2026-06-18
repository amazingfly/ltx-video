#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step() {
  local name="$1"
  shift
  local attempt=1
  local max_attempts="${PIPELINE_RETRIES:-5}"

  while true; do
    log "Starting ${name} (attempt ${attempt}/${max_attempts})"
    if "$@"; then
      log "Finished ${name}"
      return 0
    fi

    log "Failed ${name}"
    if (( attempt >= max_attempts )); then
      log "Giving up on ${name} after ${attempt} attempt(s)"
      return 1
    fi

    attempt=$((attempt + 1))
    sleep "${PIPELINE_RETRY_SLEEP_SECONDS:-20}"
  done
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="${MANIFEST:-$ROOT_DIR/outputs/$(date +%Y%m%d-%H%M%S)/manifest.json}"
MUSIC_TRACK="${MUSIC_TRACK:-}"
PLAYBACK_FPS="${PLAYBACK_FPS:-12}"
OUTPUT_FPS="${OUTPUT_FPS:-24}"
TRANSITION_SECONDS="${TRANSITION_SECONDS:-0.5}"

mkdir -p "$(dirname "$MANIFEST")"

log "Using manifest: $MANIFEST"
if [[ -n "$MUSIC_TRACK" ]]; then
  log "Using music track: $MUSIC_TRACK"
  PREPARE_MUSIC_ARGS=(--music "$MUSIC_TRACK")
else
  log "Using a random music track from the configured directory"
  PREPARE_MUSIC_ARGS=()
fi
log "Logging to stdout only; set PIPELINE_RETRIES / PIPELINE_RETRY_SLEEP_SECONDS to tune retries"

run_step "prepare" ltx-music-video prepare --manifest "$MANIFEST" "${PREPARE_MUSIC_ARGS[@]}"
run_step "generate" ltx-music-video generate --manifest "$MANIFEST" --batch-size 0
run_step "assemble" ltx-music-video assemble --manifest "$MANIFEST"
run_step \
  "assemble-transitions" \
  ltx-music-video assemble-transitions \
    --manifest "$MANIFEST" \
    --playback-fps "$PLAYBACK_FPS" \
    --output-fps "$OUTPUT_FPS" \
    --transition-seconds "$TRANSITION_SECONDS"

log "Pipeline complete"
