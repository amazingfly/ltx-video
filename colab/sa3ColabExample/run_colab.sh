#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-sa3-medium}"
GPU="${COLAB_GPU:-T4}"
GENERATE_TIMEOUT="${SA3_COLAB_GENERATE_TIMEOUT:-21600}"
CONFIG_PATH="${SA3_COLAB_CONFIG:-${ROOT}/colab/generation_config.json}"
LOCAL_OUTPUT_DIR="${SA3_COLAB_LOCAL_OUTPUT_DIR:-${ROOT}/outputs/colab}"
REMOTE_ROOT="/content/sa3"
REMOTE_COLAB_DIR="${REMOTE_ROOT}/colab"
REMOTE_OUTPUT_DIR="/content/outputs"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Colab generation config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "${LOCAL_OUTPUT_DIR}"

mapfile -t OUTPUT_NAMES < <(
  python3 -c \
    'import json,sys; c=json.load(open(sys.argv[1])); gs=c.get("generations") or [c["generation"]]; print(*[g["output_name"] for g in gs], sep="\n")' \
    "${CONFIG_PATH}"
)

sync_completed_outputs() {
  local output_name report_name temporary report_temporary
  for output_name in "${OUTPUT_NAMES[@]}"; do
    if [[ -s "${LOCAL_OUTPUT_DIR}/${output_name}" ]]; then
      continue
    fi
    temporary="${LOCAL_OUTPUT_DIR}/.${output_name}.partial"
    report_name="${output_name%.flac}.json"
    report_temporary="${LOCAL_OUTPUT_DIR}/.${report_name}.partial"
    rm -f "${temporary}"
    rm -f "${report_temporary}"
    # The report is written only after FLAC validation. Fetch it first so we
    # do not waste a short-lived runtime probing large files that do not exist.
    if ! timeout 30s colab download -s "${SESSION}" \
      "${REMOTE_OUTPUT_DIR}/${report_name}" \
      "${report_temporary}" >/dev/null 2>&1; then
      rm -f "${report_temporary}"
      break
    fi
    if timeout 180s colab download -s "${SESSION}" \
      "${REMOTE_OUTPUT_DIR}/${output_name}" \
      "${temporary}" >/dev/null 2>&1 \
      && python3 -c \
        'import hashlib,json,os,sys; audio,report=sys.argv[1:]; expected=json.load(open(report))["audio"]; handle=open(audio,"rb"); actual=hashlib.file_digest(handle,"sha256").hexdigest(); handle.close(); assert expected["bytes"] == os.path.getsize(audio); assert expected["sha256"] == actual' \
        "${temporary}" "${report_temporary}"; then
      mv "${temporary}" "${LOCAL_OUTPUT_DIR}/${output_name}"
      mv "${report_temporary}" "${LOCAL_OUTPUT_DIR}/${report_name}"
      echo "Checkpointed Colab audio: ${LOCAL_OUTPUT_DIR}/${output_name}"
      continue
    fi
    echo "Discarded incomplete Colab checkpoint: ${output_name}" >&2
    rm -f "${temporary}" "${report_temporary}"
    break
  done
}

run_colab_exec_watched() {
  local label="$1"
  local wall_timeout="$2"
  shift 2

  local command_log command_pid return_code
  command_log="$(mktemp "${LOCAL_OUTPUT_DIR}/.colab-exec.XXXXXX.log")"
  timeout --foreground "${wall_timeout}s" \
    colab exec -s "${SESSION}" "$@" > >(tee "${command_log}") 2>&1 &
  command_pid=$!

  while kill -0 "${command_pid}" 2>/dev/null; do
    if grep -Fq "Connection was lost." "${command_log}"; then
      echo "${label}: Colab CLI connection was lost; aborting this attempt." >&2
      kill "${command_pid}" 2>/dev/null || true
      sleep 1
      kill -KILL "${command_pid}" 2>/dev/null || true
      wait "${command_pid}" 2>/dev/null || true
      rm -f "${command_log}"
      return 1
    fi
    sleep 2
  done

  if wait "${command_pid}"; then
    return_code=0
  else
    return_code=$?
  fi
  rm -f "${command_log}"
  return "${return_code}"
}

run_colab_file_with_retries() {
  local label="$1"
  local file="$2"
  local remote_timeout="$3"
  local wall_timeout="$4"
  local attempt

  for attempt in 1 2 3; do
    if run_colab_exec_watched \
      "${label}" \
      "${wall_timeout}" \
      -f "${file}" \
      --timeout "${remote_timeout}"; then
      return 0
    fi
    echo "${label} attempt ${attempt}/3 failed; reconnecting." >&2
    sleep 5
  done
  return 1
}

if [[ ! -s "${HOME}/.config/colab-cli/token.json" ]]; then
  COLAB_BIN="$(command -v colab)"
  COLAB_PYTHON="$(sed -n '1s/^#!//p' "${COLAB_BIN}")"
  OAUTHLIB_RELAX_TOKEN_SCOPE=1 "${COLAB_PYTHON}" "${ROOT}/colab/auth_colab.py"
fi

session_created=0
cleanup() {
  if [[ "${session_created}" -eq 1 ]]; then
    echo "Stopping Colab session ${SESSION}"
    colab stop -s "${SESSION}" 2>/dev/null || true
  fi
  if [[ -f "${HOME}/.config/colab-cli/token.json" ]]; then
    chmod 600 "${HOME}/.config/colab-cli/token.json"
  fi
}
trap cleanup EXIT

echo "Creating Colab ${GPU} session: ${SESSION}"
colab new -s "${SESSION}" --gpu "${GPU}"
session_created=1
colab status -s "${SESSION}"

# Colab CLI 0.5.9's detached keep-alive can receive USER_PROJECT_DENIED
# with its bundled OAuth client. The official frontend attaches to this exact
# CLI-created runtime and supplies the normal Colab web keep-alive.
if [[ "${COLAB_OPEN_FRONTEND:-1}" == "1" ]]; then
  echo "Opening the CLI runtime in the Colab frontend for keep-alive"
  colab url -s "${SESSION}" --open
  sleep 5
fi

runtime_ready=0
for attempt in 1 2 3; do
  if printf '%s\n' \
    "from pathlib import Path" \
    "Path('${REMOTE_COLAB_DIR}').mkdir(parents=True, exist_ok=True)" \
    "Path('${REMOTE_OUTPUT_DIR}').mkdir(parents=True, exist_ok=True)" \
    | timeout 75s colab exec -s "${SESSION}" --timeout 60; then
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

for name in requirements.txt setup_colab.py probe_colab.py run_generate.py launch_generate.py; do
  colab upload -s "${SESSION}" \
    "${ROOT}/colab/${name}" \
    "${REMOTE_COLAB_DIR}/${name}"
done
colab upload -s "${SESSION}" \
  "${CONFIG_PATH}" \
  "${REMOTE_COLAB_DIR}/generation_config.json"
LOCAL_HF_TOKEN="${HOME}/.cache/huggingface/token"
if [[ -s "${LOCAL_HF_TOKEN}" ]]; then
  colab upload -s "${SESSION}" \
    "${LOCAL_HF_TOKEN}" \
    "${REMOTE_ROOT}/hf_token" >/dev/null
  echo "Uploaded Hugging Face credentials for authenticated model download"
fi

run_colab_file_with_retries \
  "Colab setup" \
  "${ROOT}/colab/setup_colab.py" \
  1800 \
  1860

kernel_restarted=0
for attempt in 1 2 3; do
  if timeout --foreground 120s colab restart-kernel -s "${SESSION}"; then
    kernel_restarted=1
    break
  fi
  echo "Colab kernel restart attempt ${attempt}/3 failed; retrying." >&2
  sleep 5
done
if [[ "${kernel_restarted}" -ne 1 ]]; then
  echo "Could not restart the Colab kernel after setup." >&2
  exit 1
fi

run_colab_file_with_retries \
  "Colab probe" \
  "${ROOT}/colab/probe_colab.py" \
  300 \
  360

launch_report="${LOCAL_OUTPUT_DIR}/.generation_worker.json"
rm -f "${launch_report}"
worker_launched=0
for attempt in 1 2 3; do
  timeout 90s colab exec -s "${SESSION}" \
    -f "${ROOT}/colab/launch_generate.py" --timeout 60 || true
  rm -f "${launch_report}"
  if timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT_DIR}/generation_worker.json" \
    "${launch_report}" >/dev/null 2>&1; then
    echo "Verified detached Colab generation worker launch."
    worker_launched=1
    break
  fi
  echo "Colab generation launch attempt ${attempt}/3 failed; retrying." >&2
  sleep 5
done
rm -f "${launch_report}"
if [[ "${worker_launched}" -ne 1 ]]; then
  echo "Could not launch the detached Colab generation worker." >&2
  exit 1
fi

console_offset=0
sync_generation_console() {
  local temporary size
  temporary="${LOCAL_OUTPUT_DIR}/.generation_console.log.partial"
  rm -f "${temporary}"
  if ! timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT_DIR}/generation_console.log" \
    "${temporary}" >/dev/null 2>&1; then
    rm -f "${temporary}"
    return
  fi
  size="$(stat -c '%s' "${temporary}")"
  if (( size > console_offset )); then
    tail -c "+$((console_offset + 1))" "${temporary}"
    console_offset="${size}"
  fi
  mv "${temporary}" "${LOCAL_OUTPUT_DIR}/generation_console.log"
}

generation_return_code=""
generation_started_at="$(date +%s)"
while [[ -z "${generation_return_code}" ]]; do
  sleep 5
  sync_generation_console
  sync_completed_outputs
  status_output="$(colab status -s "${SESSION}" 2>&1 || true)"
  if [[ "${status_output}" != *"Status:"* ]] \
    || [[ "${status_output}" == *"not found"* ]] \
    || [[ "${status_output}" == *"No active sessions"* ]]; then
    failure_log="${LOCAL_OUTPUT_DIR}/colab_runtime_error.log"
    {
      echo "Colab runtime disappeared during queue generation."
      echo "Detected at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "Last Colab status: ${status_output}"
      history_path="${HOME}/.config/colab-cli/history/${SESSION}.jsonl"
      if [[ -f "${history_path}" ]]; then
        python3 -c \
          'import json,sys; events=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]; relevant=[event for event in events if event.get("event_type") in {"keep_alive_error","keep_alive_stopped","session_terminated"}]; print("Last lifecycle event:", json.dumps(relevant[-1] if relevant else events[-1], sort_keys=True))' \
          "${history_path}"
      fi
    } | tee "${failure_log}" >&2
    exit 1
  fi
  exit_temporary="${LOCAL_OUTPUT_DIR}/.generation_exit.json.partial"
  rm -f "${exit_temporary}"
  if timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT_DIR}/generation_exit.json" \
    "${exit_temporary}" >/dev/null 2>&1; then
    generation_return_code="$(
      python3 -c \
        'import json,sys; print(int(json.load(open(sys.argv[1]))["return_code"]))' \
        "${exit_temporary}"
    )"
    mv "${exit_temporary}" "${LOCAL_OUTPUT_DIR}/generation_exit.json"
  fi
  if (( $(date +%s) - generation_started_at > GENERATE_TIMEOUT )); then
    echo "Detached Colab generation exceeded ${GENERATE_TIMEOUT} seconds." >&2
    exit 124
  fi
  echo "Colab detached-worker heartbeat: ${status_output}"
done
sync_generation_console
sync_completed_outputs
if [[ "${generation_return_code}" -ne 0 ]]; then
  echo "Detached Colab generation worker exited with code ${generation_return_code}." >&2
  exit "${generation_return_code}"
fi

for name in generation.json generation_error.log probe.json setup.json generation_worker.json; do
  temporary="${LOCAL_OUTPUT_DIR}/.${name}.partial"
  rm -f "${temporary}"
  if timeout 60s colab download -s "${SESSION}" \
    "${REMOTE_OUTPUT_DIR}/${name}" \
    "${temporary}" >/dev/null 2>&1; then
    mv "${temporary}" "${LOCAL_OUTPUT_DIR}/${name}"
  else
    rm -f "${temporary}"
  fi
done

for output_name in "${OUTPUT_NAMES[@]}"; do
  if [[ -s "${LOCAL_OUTPUT_DIR}/${output_name}" ]]; then
    echo "Downloaded audio: ${LOCAL_OUTPUT_DIR}/${output_name}"
  else
    echo "Queue output missing or failed: ${output_name}" >&2
  fi
done
