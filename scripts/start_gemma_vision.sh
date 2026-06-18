#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/derek/projects/hug/gemma4_llamacpp"
SERVER="${BASE_DIR}/llama.cpp/build/bin/llama-server"
MODEL="${BASE_DIR}/models/gemma4/google_gemma-4-E4B-it-Q4_K_M.gguf"
MMPROJ="${BASE_DIR}/models/gemma4/mmproj-google_gemma-4-E4B-it-f16.gguf"
PID_FILE="/tmp/ltx_gemma4_vision.pid"
LOG_FILE="/tmp/ltx_gemma4_vision.log"
HOST="127.0.0.1"
PORT="8080"

[[ -x "${SERVER}" ]] || { echo "Missing llama-server: ${SERVER}" >&2; exit 1; }
[[ -f "${MODEL}" ]] || { echo "Missing Gemma model: ${MODEL}" >&2; exit 1; }
[[ -f "${MMPROJ}" ]] || { echo "Missing Gemma projector: ${MMPROJ}" >&2; exit 1; }

if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "Gemma vision server is already running."
  exit 0
fi

echo "Starting Gemma 4 E4B vision server..."
setsid "${SERVER}" \
  -m "${MODEL}" \
  --mmproj "${MMPROJ}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --threads 8 \
  --ctx-size 8192 \
  --parallel 1 \
  --n-gpu-layers 0 \
  --reasoning off \
  --reasoning-budget 0 \
  --timeout 86400 \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --cont-batching \
  --cache-ram 512 \
  >"${LOG_FILE}" 2>&1 &

pid=$!
disown "${pid}"
printf '%s\n' "${pid}" >"${PID_FILE}"

for _ in {1..120}; do
  if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "Gemma vision server is ready."
    exit 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Gemma vision server exited during startup. See ${LOG_FILE}" >&2
    exit 1
  fi
  sleep 1
done

echo "Gemma vision server startup timed out. See ${LOG_FILE}" >&2
exit 1

