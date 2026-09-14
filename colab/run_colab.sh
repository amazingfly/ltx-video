#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-ltx-music-video}"
GPU="${COLAB_GPU:-T4}"
BUNDLE="${LTX_BUNDLE:?LTX_BUNDLE must point to the generated tar.gz job}"
LOCAL_OUTPUT_DIR="${LTX_LOCAL_OUTPUT_DIR:-${ROOT}/outputs/colab}"
LOCAL_PROMPT_CACHE="${LTX_PROMPT_CACHE:?LTX_PROMPT_CACHE must be set}"
GENERATE_TIMEOUT="${LTX_GENERATE_TIMEOUT:-43200}"
GUARDIAN_START_TIMEOUT="${LTX_GUARDIAN_START_TIMEOUT:-60}"
GUARDIAN_STALE_AFTER="${LTX_GUARDIAN_STALE_AFTER:-75}"
PROGRESS_STALE_AFTER="${LTX_PROGRESS_STALE_AFTER:-1200}"
TUNNEL_KEEPALIVE_INTERVAL="${COLAB_TUNNEL_KEEPALIVE_INTERVAL:-45}"
TUNNEL_KEEPALIVE_TIMEOUT="${COLAB_TUNNEL_KEEPALIVE_TIMEOUT:-10}"
FRONTEND_KEEPALIVE_START_TIMEOUT="${COLAB_FRONTEND_KEEPALIVE_START_TIMEOUT:-60}"
UPLOAD_CHUNK_BYTES="${COLAB_UPLOAD_CHUNK_BYTES:-33554432}"
REMOTE_ROOT="/content/ltx_music_video"
REMOTE_COLAB="${REMOTE_ROOT}/colab"
REMOTE_JOB="${REMOTE_ROOT}/job"
REMOTE_CHUNKS="${REMOTE_ROOT}/chunks"
REMOTE_OUTPUT="/content/outputs"
STREAM_BATCH_SIZE="${LTX_STREAM_BATCH_SIZE:-0}"

mkdir -p "${LOCAL_OUTPUT_DIR}"
mapfile -t OUTPUT_NAMES < <(
  python3 - "${BUNDLE}" <<'PY'
import json, sys, tarfile
with tarfile.open(sys.argv[1], "r:gz") as archive:
    job = json.load(archive.extractfile("job.json"))
for clip in job["clips"]:
    print(clip["output_name"])
PY
)
STREAMING="$(
  python3 - "${BUNDLE}" "${STREAM_BATCH_SIZE}" <<'PY'
import json, sys, tarfile
with tarfile.open(sys.argv[1], "r:gz") as archive:
    job = json.load(archive.extractfile("job.json"))
batch_size = int(sys.argv[2])
print("1" if job.get("streaming") and batch_size > 0 else "0")
PY
)"
STREAM_CHUNK_COUNT="$(
  python3 - "${BUNDLE}" "${STREAM_BATCH_SIZE}" <<'PY'
import json, sys, tarfile
with tarfile.open(sys.argv[1], "r:gz") as archive:
    job = json.load(archive.extractfile("job.json"))
batch_size = int(sys.argv[2])
if not job.get("streaming") or batch_size <= 0:
    print(0)
else:
    print((len(job["clips"]) + batch_size - 1) // batch_size)
PY
)"

session_created=0
guardian_pid=""
guardian_log="${LOCAL_OUTPUT_DIR}/colab_guardian.log"
frontend_pid=""
frontend_ready="${LOCAL_OUTPUT_DIR}/colab_frontend_keepalive.ready.json"
frontend_log="${LOCAL_OUTPUT_DIR}/colab_frontend_keepalive.log"
tunnel_keepalive_log="${LOCAL_OUTPUT_DIR}/colab_tunnel_keepalive.log"
last_tunnel_keepalive_at=0
cleanup() {
  if [[ -n "${guardian_pid}" ]]; then
    kill "${guardian_pid}" >/dev/null 2>&1 || true
    wait "${guardian_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${frontend_pid}" ]]; then
    kill "${frontend_pid}" >/dev/null 2>&1 || true
    wait "${frontend_pid}" >/dev/null 2>&1 || true
  fi
  if [[ "${session_created}" -eq 1 ]]; then
    echo "Stopping Colab session ${SESSION}"
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

resolve_colab_python() {
  local colab_path shebang
  colab_path="$(command -v colab)"
  shebang="$(head -n 1 "${colab_path}" 2>/dev/null || true)"
  if [[ "${shebang}" == "#!"* ]] && [[ "${shebang}" != *"/usr/bin/env "* ]]; then
    printf '%s\n' "${shebang#\#!}"
  else
    printf '%s\n' "python3"
  fi
}

COLAB_PYTHON="${COLAB_PYTHON:-$(resolve_colab_python)}"

run_exec_with_retries() {
  local wall_timeout="$1"
  shift
  local attempt
  for attempt in 1 2 3; do
    if timeout --foreground "${wall_timeout}" \
      colab exec -s "${SESSION}" "$@" </dev/null; then
      return 0
    fi
    echo "Colab exec attempt ${attempt}/3 failed; retrying in 5s." >&2
    sleep 5
  done
  return 1
}

run_stdin_exec_with_retries() {
  local wall_timeout="$1"
  shift
  local input_file attempt
  input_file="$(mktemp)"
  cat >"${input_file}"
  for attempt in 1 2 3; do
    if timeout --foreground "${wall_timeout}" \
      colab exec -s "${SESSION}" "$@" <"${input_file}"; then
      rm -f "${input_file}"
      return 0
    fi
    echo "Colab exec attempt ${attempt}/3 failed; retrying in 5s." >&2
    sleep 5
  done
  rm -f "${input_file}"
  return 1
}

upload_with_retries() {
  local source="$1"
  local destination="$2"
  local attempt
  for attempt in 1 2 3; do
    if timeout --foreground 600s \
      colab upload -s "${SESSION}" "${source}" "${destination}"; then
      return 0
    fi
    echo "Colab upload attempt ${attempt}/3 failed; retrying in 5s." >&2
    sleep 5
  done
  return 1
}

upload_bundle_with_retries() {
  local source="$1"
  local destination="$2"
  local source_size chunk_dir part remote_part part_number expected_parts
  source_size="$(stat -c '%s' "${source}")"
  if (( UPLOAD_CHUNK_BYTES <= 0 || source_size <= UPLOAD_CHUNK_BYTES )); then
    upload_with_retries "${source}" "${destination}"
    return
  fi

  echo "Uploading ${source_size} byte job bundle in ${UPLOAD_CHUNK_BYTES} byte chunks"
  printf '%s\n' \
    "from pathlib import Path" \
    "destination = Path('${destination}')" \
    "destination.unlink(missing_ok=True)" \
    "for part in destination.parent.glob(destination.name + '.part.*'):" \
    "    part.unlink(missing_ok=True)" \
    | run_stdin_exec_with_retries 90s --timeout 60

  chunk_dir="$(mktemp -d "${LOCAL_OUTPUT_DIR}/bundle-chunks.XXXXXX")"
  split -b "${UPLOAD_CHUNK_BYTES}" -d -a 4 "${source}" "${chunk_dir}/job.tar.gz.part."
  expected_parts="$(find "${chunk_dir}" -maxdepth 1 -type f | wc -l)"
  part_number=0
  for part in "${chunk_dir}"/job.tar.gz.part.*; do
    remote_part="${destination}.part.$(printf '%04d' "${part_number}")"
    upload_with_retries "${part}" "${remote_part}" || {
      rm -rf "${chunk_dir}"
      return 1
    }
    part_number=$((part_number + 1))
  done
  rm -rf "${chunk_dir}"

  printf '%s\n' \
    "import os, shutil" \
    "from pathlib import Path" \
    "destination = Path('${destination}')" \
    "parts = sorted(destination.parent.glob(destination.name + '.part.*'))" \
    "expected_parts = ${expected_parts}" \
    "expected_size = ${source_size}" \
    "if len(parts) != expected_parts:" \
    "    raise RuntimeError(f'Expected {expected_parts} bundle parts, found {len(parts)}')" \
    "temporary = destination.with_name(destination.name + '.tmp')" \
    "with temporary.open('wb') as output:" \
    "    for part in parts:" \
    "        with part.open('rb') as input_file:" \
    "            shutil.copyfileobj(input_file, output)" \
    "if temporary.stat().st_size != expected_size:" \
    "    raise RuntimeError(f'Reassembled bundle has {temporary.stat().st_size} bytes, expected {expected_size}')" \
    "os.replace(temporary, destination)" \
    "for part in parts:" \
    "    part.unlink(missing_ok=True)" \
    "print(f'Reassembled bundle: {destination} ({expected_size} bytes)', flush=True)" \
    | run_stdin_exec_with_retries 300s --timeout 240
}

create_stream_chunk_bundle() {
  local chunk_index="$1"
  local destination="$2"
  python3 - "${BUNDLE}" "${STREAM_BATCH_SIZE}" "${chunk_index}" "${destination}" <<'PY'
import json
import sys
import tarfile
from pathlib import Path

bundle = Path(sys.argv[1])
batch_size = int(sys.argv[2])
chunk_index = int(sys.argv[3])
destination = Path(sys.argv[4])
destination.parent.mkdir(parents=True, exist_ok=True)

with tarfile.open(bundle, "r:gz") as archive:
    job = json.load(archive.extractfile("job.json"))

start = chunk_index * batch_size
clips = job["clips"][start : start + batch_size]
if not clips:
    raise SystemExit(f"empty stream chunk {chunk_index}")

chunk = {
    "chunk_index": chunk_index,
    "clip_ids": [clip["id"] for clip in clips],
}
chunk_json = destination.with_name(f"{destination.name}.json")
chunk_json.write_text(json.dumps(chunk, indent=2) + "\n", encoding="utf-8")
with tarfile.open(destination, "w:gz") as archive:
    archive.add(chunk_json, arcname="chunk.json", recursive=False)
    for clip in clips:
        source = Path(clip["local_image_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        archive.add(source, arcname=clip["image_path"], recursive=False)
chunk_json.unlink(missing_ok=True)
print(f"Created stream chunk {chunk_index:04d} with {len(clips)} clip(s)")
PY
}

upload_stream_chunk() {
  local chunk_index="$1"
  local chunk_bundle ready_file remote_final remote_ready
  chunk_bundle="$(mktemp "${LOCAL_OUTPUT_DIR}/stream-chunk-${chunk_index}.XXXXXX")"
  ready_file="$(mktemp "${LOCAL_OUTPUT_DIR}/stream-chunk-${chunk_index}.ready.XXXXXX")"
  create_stream_chunk_bundle "${chunk_index}" "${chunk_bundle}"
  remote_final="${REMOTE_CHUNKS}/chunk-$(printf '%04d' "${chunk_index}").tar.gz"
  remote_ready="${REMOTE_CHUNKS}/chunk-$(printf '%04d' "${chunk_index}").ready.json"
  upload_with_retries "${chunk_bundle}" "${remote_final}" || {
    rm -f "${chunk_bundle}" "${ready_file}"
    return 1
  }
  rm -f "${chunk_bundle}"
  printf '{"chunk_index": %s}\n' "${chunk_index}" >"${ready_file}"
  upload_with_retries "${ready_file}" "${remote_ready}" || {
    rm -f "${ready_file}"
    return 1
  }
  rm -f "${ready_file}"
  echo "Stream chunk ${chunk_index} is ready"
}

send_tunnel_keepalive() {
  [[ "${COLAB_TUNNEL_KEEPALIVE:-1}" == "1" ]] || return 0
  local now config_args=()
  now="$(date +%s)"
  if (( now - last_tunnel_keepalive_at < TUNNEL_KEEPALIVE_INTERVAL )); then
    return 0
  fi
  last_tunnel_keepalive_at="${now}"
  if [[ -n "${COLAB_CONFIG:-}" ]]; then
    config_args=(--config "${COLAB_CONFIG}")
  fi
  if ! "${COLAB_PYTHON}" "${ROOT}/scripts/colab_tunnel_keepalive.py" \
    --session "${SESSION}" \
    --authuser "${COLAB_AUTHUSER:-0}" \
    --auth-provider "${COLAB_AUTH_PROVIDER:-oauth2}" \
    --request-timeout "${TUNNEL_KEEPALIVE_TIMEOUT}" \
    "${config_args[@]}" \
    >>"${tunnel_keepalive_log}" 2>&1; then
    echo "Colab tunnel keep-alive ping failed; continuing. See ${tunnel_keepalive_log}" >&2
    return 1
  fi
}

open_frontend_url() {
  local frontend_url="$1"
  echo "Opening the exact CLI runtime in the Colab frontend for keep-alive"
  python3 - "${frontend_url}" "${COLAB_AUTHUSER:-0}" <<'PY'
import sys
import urllib.parse
import webbrowser

url, authuser = sys.argv[1:]
parts = urllib.parse.urlsplit(url)
query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
query = [(key, value) for key, value in query if key != "authuser"]
query.append(("authuser", authuser))
webbrowser.open(
    urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), parts.fragment)
    )
)
PY
}

open_frontend() {
  local frontend_url
  frontend_url="$(colab url -s "${SESSION}")"
  open_frontend_url "${frontend_url}"
}

start_frontend_keepalive() {
  [[ "${COLAB_OPEN_FRONTEND:-1}" == "1" ]] || return 0
  local frontend_url started_at
  frontend_url="$(colab url -s "${SESSION}")"
  if [[ "${COLAB_FRONTEND_KEEPALIVE:-1}" != "1" ]]; then
    open_frontend_url "${frontend_url}"
    return 0
  fi

  rm -f "${frontend_ready}" "${frontend_log}"
  echo "Starting authenticated Colab frontend keep-alive"
  python3 "${ROOT}/scripts/colab_frontend_keepalive.py" \
    --url "${frontend_url}" \
    --authuser "${COLAB_AUTHUSER:-0}" \
    --ready-file "${frontend_ready}" \
    >"${frontend_log}" 2>&1 &
  frontend_pid=$!
  started_at="$(date +%s)"
  while [[ ! -s "${frontend_ready}" ]]; do
    if ! kill -0 "${frontend_pid}" 2>/dev/null; then
      wait "${frontend_pid}" >/dev/null 2>&1 || true
      frontend_pid=""
      echo "Colab frontend keep-alive failed to start; falling back to browser open. See ${frontend_log}" >&2
      open_frontend_url "${frontend_url}"
      return 0
    fi
    if (( $(date +%s) - started_at > FRONTEND_KEEPALIVE_START_TIMEOUT )); then
      kill "${frontend_pid}" >/dev/null 2>&1 || true
      wait "${frontend_pid}" >/dev/null 2>&1 || true
      frontend_pid=""
      echo "Timed out starting Colab frontend keep-alive; falling back to browser open. See ${frontend_log}" >&2
      open_frontend_url "${frontend_url}"
      return 0
    fi
    sleep 2
  done
  echo "Colab frontend keep-alive is active"
}

download_completed() {
  local output report temporary report_temporary
  for output in "${OUTPUT_NAMES[@]}"; do
    [[ -s "${LOCAL_OUTPUT_DIR}/${output}" ]] && continue
    report="${output%.mp4}.json"
    temporary="${LOCAL_OUTPUT_DIR}/.${output}.partial"
    report_temporary="${LOCAL_OUTPUT_DIR}/.${report}.partial"
    rm -f "${temporary}" "${report_temporary}"
    if ! timeout 30s colab download -s "${SESSION}" \
      "${REMOTE_OUTPUT}/${report}" "${report_temporary}" >/dev/null 2>&1; then
      rm -f "${report_temporary}"
      continue
    fi
    if timeout 180s colab download -s "${SESSION}" \
      "${REMOTE_OUTPUT}/${output}" "${temporary}" >/dev/null 2>&1 \
      && python3 - "${temporary}" "${report_temporary}" <<'PY'
import hashlib, json, os, sys
video, report = sys.argv[1:]
expected = json.load(open(report))
with open(video, "rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
assert expected["bytes"] == os.path.getsize(video)
assert expected["sha256"] == digest
PY
    then
      mv "${temporary}" "${LOCAL_OUTPUT_DIR}/${output}"
      mv "${report_temporary}" "${LOCAL_OUTPUT_DIR}/${report}"
      echo "Downloaded completed clip: ${output}"
      downloaded_any=1
    else
      rm -f "${temporary}" "${report_temporary}"
    fi
  done
}

stream_next_upload=1
stream_completed_until=-1
sync_stream_chunks() {
  [[ "${STREAMING}" == "1" ]] || return 0
  while (( stream_next_upload < STREAM_CHUNK_COUNT )); do
    local done_index marker temporary
    done_index=$((stream_next_upload - 1))
    if (( done_index > stream_completed_until )); then
      marker="chunk-$(printf '%04d' "${done_index}").done.json"
      temporary="${LOCAL_OUTPUT_DIR}/.${marker}.partial"
      rm -f "${temporary}"
      if ! timeout 60s colab download -s "${SESSION}" \
        "${REMOTE_OUTPUT}/${marker}" "${temporary}" >/dev/null 2>&1; then
        rm -f "${temporary}"
        return 0
      fi
      mv "${temporary}" "${LOCAL_OUTPUT_DIR}/${marker}"
      stream_completed_until="${done_index}"
    fi
    echo "Remote chunk ${done_index} completed; uploading stream chunk ${stream_next_upload}/${STREAM_CHUNK_COUNT}"
    upload_stream_chunk "${stream_next_upload}"
    stream_next_upload=$((stream_next_upload + 1))
  done
}

download_prompt_cache() {
  local temporary report_temporary expected_bytes current_bytes
  temporary="${LOCAL_PROMPT_CACHE}.partial"
  report_temporary="${LOCAL_PROMPT_CACHE}.json.partial"
  mkdir -p "$(dirname "${LOCAL_PROMPT_CACHE}")"
  rm -f "${temporary}" "${report_temporary}"
  if ! timeout 30s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT}/prompt_embeddings.json" \
    "${report_temporary}" >/dev/null 2>&1; then
    rm -f "${report_temporary}"
    return
  fi
  expected_bytes="$(
    python3 -c \
      'import json,sys; print(int(json.load(open(sys.argv[1]))["bytes"]))' \
      "${report_temporary}"
  )"
  current_bytes="$(stat -c '%s' "${LOCAL_PROMPT_CACHE}" 2>/dev/null || echo 0)"
  if [[ "${current_bytes}" == "${expected_bytes}" ]]; then
    rm -f "${report_temporary}"
    return
  fi
  if timeout 300s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT}/prompt_embeddings.pt" "${temporary}" >/dev/null 2>&1 \
    && [[ "$(stat -c '%s' "${temporary}")" == "${expected_bytes}" ]]; then
    mv "${temporary}" "${LOCAL_PROMPT_CACHE}"
    mv "${report_temporary}" "${LOCAL_PROMPT_CACHE}.json"
    echo "Downloaded prompt embedding cache"
  else
    rm -f "${temporary}" "${report_temporary}"
  fi
}

echo "Creating Colab ${GPU} session: ${SESSION}"
existing_status="$(colab status -s "${SESSION}" 2>&1 || true)"
if [[ "${existing_status}" == *"Status:"* ]] \
  && [[ "${existing_status}" != *"not found"* ]]; then
  echo "Reusing existing Colab session: ${SESSION}"
else
  session_ready=0
  for attempt in 1 2 3 4 5; do
    if colab new -s "${SESSION}" --gpu "${GPU}"; then
      session_ready=1
      break
    fi
    echo "Colab allocation attempt ${attempt}/5 failed; retrying in 30s." >&2
    sleep 30
  done
  if [[ "${session_ready}" -ne 1 ]]; then
    echo "Could not allocate a Colab ${GPU} runtime." >&2
    exit 75
  fi
fi
session_created=1
colab status -s "${SESSION}"
send_tunnel_keepalive || true

runtime_ready=0
for attempt in 1 2 3; do
  if printf '%s\n' \
    "from pathlib import Path" \
    "Path('${REMOTE_COLAB}').mkdir(parents=True, exist_ok=True)" \
    "Path('${REMOTE_JOB}').mkdir(parents=True, exist_ok=True)" \
    "Path('${REMOTE_CHUNKS}').mkdir(parents=True, exist_ok=True)" \
    "Path('${REMOTE_OUTPUT}').mkdir(parents=True, exist_ok=True)" \
    | run_stdin_exec_with_retries 75s --timeout 60; then
    runtime_ready=1
    break
  fi
  echo "Colab kernel connection attempt ${attempt}/3 failed; retrying." >&2
  sleep 5
done
if [[ "${runtime_ready}" -ne 1 ]]; then
  echo "Could not establish a stable Colab kernel connection." >&2
  exit 1
fi

printf '%s\n' \
  "import os, signal" \
  "from pathlib import Path" \
  "patterns = (b'/content/ltx_music_video/colab/run_generate.py', b'/content/ltx_music_video/colab/launch_generate.py')" \
  "for proc in Path('/proc').glob('[0-9]*'):" \
  "    try:" \
  "        command = (proc / 'cmdline').read_bytes()" \
  "        if any(pattern in command for pattern in patterns):" \
  "            os.kill(int(proc.name), signal.SIGTERM)" \
  "    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):" \
  "        pass" \
  | run_stdin_exec_with_retries 90s --timeout 60
sleep 3

for name in requirements.txt setup_colab.py run_generate.py launch_generate.py; do
  upload_with_retries \
    "${ROOT}/colab/${name}" "${REMOTE_COLAB}/${name}"
done
upload_bundle_with_retries "${BUNDLE}" "${REMOTE_ROOT}/job.tar.gz"
if [[ -s "${HOME}/.cache/huggingface/token" ]]; then
  printf '%s\n' \
    "from pathlib import Path" \
    "Path('/root/.cache/huggingface').mkdir(parents=True, exist_ok=True)" \
    | run_stdin_exec_with_retries 90s --timeout 60
  upload_with_retries \
    "${HOME}/.cache/huggingface/token" "/root/.cache/huggingface/token"
fi

printf '%s\n' \
  "import tarfile" \
  "tarfile.open('${REMOTE_ROOT}/job.tar.gz', 'r:gz').extractall('${REMOTE_JOB}', filter='data')" \
  | run_stdin_exec_with_retries 300s --timeout 240

printf '%s\n' \
  "import json" \
  "from pathlib import Path" \
  "job = json.loads(Path('${REMOTE_JOB}/job.json').read_text())" \
  "outputs = Path('${REMOTE_OUTPUT}')" \
  "for clip in job['clips']:" \
  "    for name in (clip['output_name'], Path(clip['output_name']).with_suffix('.json').name):" \
  "        (outputs / name).unlink(missing_ok=True)" \
  | run_stdin_exec_with_retries 90s --timeout 60

if [[ "${STREAMING}" == "1" ]]; then
  printf '%s\n' \
    "import shutil" \
    "from pathlib import Path" \
    "chunks = Path('${REMOTE_CHUNKS}')" \
    "shutil.rmtree(chunks, ignore_errors=True)" \
    "chunks.mkdir(parents=True, exist_ok=True)" \
    | run_stdin_exec_with_retries 90s --timeout 60
  echo "Streaming ${STREAM_CHUNK_COUNT} image chunk(s) into one live Colab runtime"
  upload_stream_chunk 0
fi

run_exec_with_retries 300s \
  -f "${ROOT}/colab/setup_colab.py" --timeout 270

console_offset=0
console_progress=0
sync_generation_console() {
  local temporary size
  console_progress=0
  temporary="${LOCAL_OUTPUT_DIR}/.generation_console.log.partial"
  rm -f "${temporary}"
  if ! timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT}/generation_console.log" \
    "${temporary}" >/dev/null 2>&1; then
    rm -f "${temporary}"
    return
  fi
  size="$(stat -c '%s' "${temporary}")"
  if (( size > console_offset )); then
    tail -c "+$((console_offset + 1))" "${temporary}"
    console_offset="${size}"
    console_progress=1
  fi
  mv "${temporary}" "${LOCAL_OUTPUT_DIR}/generation_console.log"
}

echo "Launching detached LTX generation worker"
run_exec_with_retries 90s \
  -f "${ROOT}/colab/launch_generate.py" --timeout 60

start_guardian() {
  local guardian_started_at
  rm -f "${guardian_log}"
  {
    printf '%s\n' \
      "import time" \
      "from pathlib import Path" \
      "exit_report = Path('${REMOTE_OUTPUT}/generation_exit.json')" \
      "while not exit_report.exists():" \
      "    print('LTX allocation guardian active', flush=True)" \
      "    time.sleep(20)"
  } | timeout --foreground "${GENERATE_TIMEOUT}s" \
    colab exec -s "${SESSION}" --timeout "${GENERATE_TIMEOUT}" \
    >"${guardian_log}" 2>&1 &
  guardian_pid=$!
  guardian_started_at="$(date +%s)"
  while ! grep -q "LTX allocation guardian active" "${guardian_log}" 2>/dev/null; do
    if ! kill -0 "${guardian_pid}" 2>/dev/null; then
      wait "${guardian_pid}" || true
      cat "${guardian_log}" >&2 || true
      echo "Colab allocation guardian failed to start." >&2
      return 1
    fi
    if (( $(date +%s) - guardian_started_at > GUARDIAN_START_TIMEOUT )); then
      kill "${guardian_pid}" >/dev/null 2>&1 || true
      wait "${guardian_pid}" >/dev/null 2>&1 || true
      cat "${guardian_log}" >&2 || true
      echo "Colab allocation guardian did not produce a heartbeat." >&2
      return 1
    fi
    sleep 2
  done
  echo "Colab allocation guardian is active"
}

start_guardian

start_frontend_keepalive

generation_return_code=""
started_at="$(date +%s)"
last_progress_at="${started_at}"
while [[ -z "${generation_return_code}" ]]; do
  sleep 15
  send_tunnel_keepalive || true
  sync_generation_console
  if [[ "${console_progress}" -eq 1 ]]; then
    last_progress_at="$(date +%s)"
  fi
  download_prompt_cache
  downloaded_any=0
  download_completed
  if [[ "${downloaded_any}" -eq 1 ]]; then
    last_progress_at="$(date +%s)"
  fi
  sync_stream_chunks

  exit_temporary="${LOCAL_OUTPUT_DIR}/.generation_exit.json.partial"
  rm -f "${exit_temporary}"
  if timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT}/generation_exit.json" \
    "${exit_temporary}" >/dev/null 2>&1; then
    generation_return_code="$(
      python3 -c \
        'import json,sys; print(int(json.load(open(sys.argv[1]))["return_code"]))' \
        "${exit_temporary}"
    )"
    mv "${exit_temporary}" "${LOCAL_OUTPUT_DIR}/generation_exit.json"
  fi

  if [[ -z "${generation_return_code}" ]] \
    && { ! kill -0 "${guardian_pid}" 2>/dev/null \
      || (( $(date +%s) - $(stat -c '%Y' "${guardian_log}" 2>/dev/null || echo 0) \
        > GUARDIAN_STALE_AFTER )); }; then
    kill "${guardian_pid}" >/dev/null 2>&1 || true
    wait "${guardian_pid}" >/dev/null 2>&1 || true
    echo "Colab allocation guardian is missing or stale; reconnecting." >&2
    if ! start_guardian; then
      echo "The Colab assignment expired before generation completed." >&2
      exit 75
    fi
  fi

  if (( $(date +%s) - started_at > GENERATE_TIMEOUT )); then
    echo "LTX generation exceeded ${GENERATE_TIMEOUT} seconds." >&2
    exit 124
  fi
  if (( $(date +%s) - last_progress_at > PROGRESS_STALE_AFTER )); then
    echo \
      "LTX generation made no console or clip progress for " \
      "${PROGRESS_STALE_AFTER} seconds; restarting this Colab attempt." >&2
    exit 76
  fi
  echo "Colab detached-worker heartbeat: guardian and artifact sync active"
done

return_code="${generation_return_code}"
sync_generation_console
download_prompt_cache
download_completed
for report in \
  generation.json generation_error.log generation_worker.json setup.json; do
  timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT}/${report}" "${LOCAL_OUTPUT_DIR}/${report}" >/dev/null 2>&1 || true
done

if [[ "${return_code}" -ne 0 ]]; then
  echo "Remote LTX generation exited with code ${return_code}." >&2
  exit "${return_code}"
fi
