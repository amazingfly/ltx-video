#!/usr/bin/env python3
"""Run the full Gemma/LTX music-video workflow with robust stage retries."""

from __future__ import annotations

# Load centralized workstation defaults; explicit environment/CLI values win.
import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
for _workspace_root in _WorkspacePath(__file__).resolve().parents:
    if (_workspace_root / "media_workspace").is_dir():
        _workspace_sys.path.insert(0, str(_workspace_root))
        break
from media_workspace.config import apply_environment as _apply_workspace
_apply_workspace()


import argparse
import datetime as dt
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PLAYBACK_FPS = 12
DEFAULT_OUTPUT_FPS = 24
DEFAULT_TRANSITION_SECONDS = 0.5
DEFAULT_STAGE_RETRIES = 5
DEFAULT_RETRY_SLEEP_SECONDS = 20
DEFAULT_QUOTA_RETRY_SLEEP_SECONDS = 300
DEFAULT_LONG_QUOTA_RETRY_SECONDS = 1800
DEFAULT_QUOTA_REFUSALS_BEFORE_LONG_RETRY = 1
DEFAULT_IDLE_TIMEOUT_SECONDS = 1800
DEFAULT_STAGE_TIMEOUT_SECONDS = 86400
DEFAULT_COLAB_TIMEOUT_SECONDS = 43200

FATAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"No such file or directory",
        r"FileNotFoundError",
        r"No supported images found",
        r"No supported music found",
        r"Music track does not exist",
        r"unrecognized arguments",
        r"Generated clip failed identity",
        r"Generated clip failed .* validation",
    )
)

QUOTA_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"USER_PROJECT_DENIED",
        r"RESOURCE_EXHAUSTED",
        r"quota",
        r"rate.?limit",
        r"no available .*gpu",
        r"cannot .*gpu",
        r"temporar(?:y|ily)",
        r"backend.*unavailable",
        r"service unavailable",
        r"failed to assign.*backend",
        r"exceeded.*limit",
        r"not.*authorized.*colab",
    )
)

ERROR_LINE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\btraceback\b",
        r"\berror\b",
        r"\bexception\b",
        r"\bfailed\b",
        r"\bout of memory\b",
        r"\bOOM\b",
        r"\bquota\b",
        r"\bRESOURCE_EXHAUSTED\b",
        r"\bUSER_PROJECT_DENIED\b",
        r"\bunauthorized\b",
        r"\brefus(?:e|al|ed)\b",
    )
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    recent_output: list[str]
    timed_out: bool = False
    idle_timed_out: bool = False


class PipelineLogger:
    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_log = run_dir / "pipeline.log"
        self.error_log = run_dir / "pipeline_errors.log"

    def info(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with self.pipeline_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def error(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, file=sys.stderr, flush=True)
        with self.error_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        with self.pipeline_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def stream(self, stage: str, stream_name: str, line: str) -> None:
        rendered = f"[{timestamp()}] [{stage}:{stream_name}] {line.rstrip()}"
        print(rendered, flush=True)
        with self.pipeline_log.open("a", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        if stream_name == "stderr" or any(
            pattern.search(line) for pattern in ERROR_LINE_PATTERNS
        ):
            with self.error_log.open("a", encoding="utf-8") as handle:
                handle.write(rendered + "\n")


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in {None, ""}:
        return default
    return int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in {None, ""}:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def sleep_with_heartbeat(
    seconds: int,
    logger: PipelineLogger,
    *,
    stage: str,
    interval: int = 60,
) -> None:
    remaining = max(0, seconds)
    while remaining > 0:
        chunk = min(interval, remaining)
        logger.info(f"{stage}: waiting {chunk}s ({remaining}s remaining)")
        time.sleep(chunk)
        remaining -= chunk


def default_manifest_path() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT_DIR / "outputs" / stamp / "manifest.json"


def classify_failure(result: CommandResult) -> str:
    output = "\n".join(result.recent_output)
    if result.timed_out:
        return "stage-timeout"
    if result.idle_timed_out:
        return "idle-timeout"
    if any(pattern.search(output) for pattern in FATAL_PATTERNS):
        return "fatal"
    if any(pattern.search(output) for pattern in QUOTA_PATTERNS):
        return "colab-quota-or-refusal"
    return "transient-or-unknown"


def kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=20)


def enqueue_stream(
    stream: object,
    stream_name: str,
    lines: queue.Queue[tuple[str, str] | None],
) -> None:
    try:
        for line in stream:  # type: ignore[operator]
            lines.put((stream_name, str(line)))
    finally:
        lines.put(None)


def run_command(
    stage: str,
    command: list[str],
    logger: PipelineLogger,
    *,
    timeout_seconds: int,
    idle_timeout_seconds: int,
) -> CommandResult:
    logger.info(f"Command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    assert process.stderr is not None

    lines: queue.Queue[tuple[str, str] | None] = queue.Queue()
    stdout_thread = threading.Thread(
        target=enqueue_stream,
        args=(process.stdout, "stdout", lines),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=enqueue_stream,
        args=(process.stderr, "stderr", lines),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    recent: deque[str] = deque(maxlen=300)
    start = time.monotonic()
    last_output = start
    closed_streams = 0
    timed_out = False
    idle_timed_out = False

    while True:
        now = time.monotonic()
        if now - start > timeout_seconds:
            logger.error(f"{stage} exceeded timeout of {timeout_seconds}s")
            timed_out = True
            kill_process_group(process)
        if now - last_output > idle_timeout_seconds:
            logger.error(f"{stage} produced no output for {idle_timeout_seconds}s")
            idle_timed_out = True
            kill_process_group(process)

        try:
            item = lines.get(timeout=1)
        except queue.Empty:
            item = None
            if process.poll() is not None and closed_streams >= 2:
                break

        if item is None:
            if process.poll() is not None:
                closed_streams += 1
                if closed_streams >= 2:
                    break
            continue

        stream_name, line = item
        last_output = time.monotonic()
        rendered = f"{stream_name}: {line.rstrip()}"
        recent.append(rendered)
        logger.stream(stage, stream_name, line)

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    return CommandResult(
        returncode=process.returncode if process.returncode is not None else 1,
        recent_output=list(recent),
        timed_out=timed_out,
        idle_timed_out=idle_timed_out,
    )


def run_step(
    stage: str,
    command: list[str],
    logger: PipelineLogger,
    *,
    max_attempts: int,
    retry_sleep_seconds: int,
    quota_retry_sleep_seconds: int,
    long_quota_retry_seconds: int,
    quota_refusals_before_long_retry: int,
    infinite_quota_retry: bool,
    timeout_seconds: int,
    idle_timeout_seconds: int,
) -> None:
    attempt = 1
    quota_refusals = 0
    long_quota_mode = False
    while True:
        if long_quota_mode:
            attempt_label = f"{attempt} (long-term quota retry)"
        else:
            attempt_label = f"{attempt}/{max_attempts}"
        logger.info(f"Starting {stage} (attempt {attempt_label})")
        result = run_command(
            stage,
            command,
            logger,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
        )
        if result.returncode == 0 and not result.timed_out and not result.idle_timed_out:
            logger.info(f"Finished {stage}")
            return

        classification = classify_failure(result)
        logger.error(
            f"{stage} failed with return code {result.returncode}; "
            f"classification={classification}"
        )
        if result.recent_output:
            logger.error(f"{stage} recent output:")
            for line in result.recent_output[-40:]:
                logger.error(f"  {line}")

        if classification == "fatal":
            raise SystemExit(f"Fatal {stage} failure; see {logger.error_log}")
        if classification == "colab-quota-or-refusal":
            quota_refusals += 1
            if (
                infinite_quota_retry
                and quota_refusals >= quota_refusals_before_long_retry
            ):
                if not long_quota_mode:
                    logger.info(
                        f"{stage} appears quota/refusal limited after "
                        f"{quota_refusals} classified failure(s); entering "
                        "long-term retry mode"
                    )
                long_quota_mode = True
                logger.info(
                    f"Waiting {long_quota_retry_seconds}s before retrying "
                    f"{stage}; this will continue until the stage succeeds"
                )
                sleep_with_heartbeat(
                    long_quota_retry_seconds,
                    logger,
                    stage=stage,
                )
                attempt += 1
                continue
        if attempt >= max_attempts:
            raise SystemExit(
                f"{stage} failed after {attempt} attempt(s); see {logger.error_log}"
            )

        sleep_seconds = (
            quota_retry_sleep_seconds
            if classification == "colab-quota-or-refusal"
            else retry_sleep_seconds
        )
        logger.info(f"Waiting {sleep_seconds}s before retrying {stage}")
        time.sleep(sleep_seconds)
        attempt += 1


def validate_manifest_paths(manifest_path: Path, logger: PipelineLogger) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Manifest was not created: {manifest_path}") from exc

    music = Path(manifest.get("music", {}).get("path", ""))
    missing = []
    if not music.is_file():
        missing.append(str(music))
    for clip in manifest.get("clips", []):
        image = Path(str(clip.get("image_path", "")))
        if not image.is_file():
            missing.append(str(image))
    if missing:
        logger.error("Manifest references missing files:")
        for path in missing[:50]:
            logger.error(f"  {path}")
        if len(missing) > 50:
            logger.error(f"  ... and {len(missing) - 50} more")
        raise SystemExit("Manifest has missing input files; regenerate or repair it")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prepare, Colab generation, standard assembly, and transition assembly."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ["MANIFEST"])
        if os.environ.get("MANIFEST")
        else default_manifest_path(),
    )
    parser.add_argument(
        "--music",
        type=Path,
        default=Path(os.environ["MUSIC_TRACK"])
        if os.environ.get("MUSIC_TRACK")
        else None,
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(os.environ["IMAGE_DIR"])
        if os.environ.get("IMAGE_DIR")
        else None,
    )
    parser.add_argument(
        "--motion-style",
        choices=("rave", "little-queen"),
        default=os.environ.get("MOTION_STYLE", "rave"),
    )
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--preserve-image-order", action="store_true")
    parser.add_argument("--regenerate-prompts", action="store_true")
    parser.add_argument("--session", default=os.environ.get("COLAB_SESSION", "ltx-music-video"))
    parser.add_argument("--gpu", default=os.environ.get("COLAB_GPU", "T4"))
    parser.add_argument("--colab-authuser", type=int, default=env_int("COLAB_AUTHUSER", 1))
    parser.add_argument("--colab-attempts", type=int, default=env_int("COLAB_MAX_ATTEMPTS", 8))
    parser.add_argument("--batch-size", type=int, default=env_int("COLAB_BATCH_SIZE", 0))
    parser.add_argument("--colab-timeout-seconds", type=int, default=env_int("COLAB_TIMEOUT_SECONDS", DEFAULT_COLAB_TIMEOUT_SECONDS))
    parser.add_argument("--no-open-frontend", action="store_true")
    parser.add_argument("--playback-fps", type=int, default=env_int("PLAYBACK_FPS", DEFAULT_PLAYBACK_FPS))
    parser.add_argument("--output-fps", type=int, default=env_int("OUTPUT_FPS", DEFAULT_OUTPUT_FPS))
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=float(os.environ.get("TRANSITION_SECONDS", DEFAULT_TRANSITION_SECONDS)),
    )
    parser.add_argument("--retries", type=int, default=env_int("PIPELINE_RETRIES", DEFAULT_STAGE_RETRIES))
    parser.add_argument(
        "--retry-sleep-seconds",
        type=int,
        default=env_int("PIPELINE_RETRY_SLEEP_SECONDS", DEFAULT_RETRY_SLEEP_SECONDS),
    )
    parser.add_argument(
        "--quota-retry-sleep-seconds",
        type=int,
        default=env_int("PIPELINE_QUOTA_SLEEP_SECONDS", DEFAULT_QUOTA_RETRY_SLEEP_SECONDS),
    )
    parser.add_argument(
        "--long-quota-retry-seconds",
        type=int,
        default=env_int(
            "PIPELINE_LONG_QUOTA_RETRY_SECONDS",
            DEFAULT_LONG_QUOTA_RETRY_SECONDS,
        ),
    )
    parser.add_argument(
        "--quota-refusals-before-long-retry",
        type=int,
        default=env_int(
            "PIPELINE_QUOTA_REFUSALS_BEFORE_LONG_RETRY",
            DEFAULT_QUOTA_REFUSALS_BEFORE_LONG_RETRY,
        ),
    )
    parser.add_argument(
        "--infinite-quota-retry",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PIPELINE_INFINITE_QUOTA_RETRY", True),
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=env_int("PIPELINE_IDLE_TIMEOUT_SECONDS", DEFAULT_IDLE_TIMEOUT_SECONDS),
    )
    parser.add_argument(
        "--stage-timeout-seconds",
        type=int,
        default=env_int("PIPELINE_STAGE_TIMEOUT_SECONDS", DEFAULT_STAGE_TIMEOUT_SECONDS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    logger = PipelineLogger(manifest_path.parent)
    logger.info(f"Using manifest: {manifest_path}")
    logger.info(f"Pipeline log: {logger.pipeline_log}")
    logger.info(f"Error log: {logger.error_log}")

    prepare = ["ltx-music-video", "prepare", "--manifest", str(manifest_path)]
    if args.image_dir:
        prepare.extend(["--image-dir", str(args.image_dir.resolve())])
    prepare.extend(["--motion-style", args.motion_style])
    if args.music:
        prepare.extend(["--music", str(args.music.resolve())])
    if args.selection_seed is not None:
        prepare.extend(["--selection-seed", str(args.selection_seed)])
    if args.preserve_image_order:
        prepare.append("--preserve-image-order")
    if args.regenerate_prompts:
        prepare.append("--regenerate-prompts")

    run_step(
        "prepare",
        prepare,
        logger,
        max_attempts=args.retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        quota_retry_sleep_seconds=args.quota_retry_sleep_seconds,
        long_quota_retry_seconds=args.long_quota_retry_seconds,
        quota_refusals_before_long_retry=args.quota_refusals_before_long_retry,
        infinite_quota_retry=args.infinite_quota_retry,
        timeout_seconds=args.stage_timeout_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )
    validate_manifest_paths(manifest_path, logger)

    generate = [
        "ltx-music-video",
        "generate",
        "--manifest",
        str(manifest_path),
        "--batch-size",
        str(args.batch_size),
        "--session",
        args.session,
        "--gpu",
        args.gpu,
        "--colab-authuser",
        str(args.colab_authuser),
        "--max-attempts",
        str(args.colab_attempts),
        "--timeout-seconds",
        str(args.colab_timeout_seconds),
    ]
    if args.no_open_frontend:
        generate.append("--no-open-frontend")

    run_step(
        "generate",
        generate,
        logger,
        max_attempts=args.retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        quota_retry_sleep_seconds=args.quota_retry_sleep_seconds,
        long_quota_retry_seconds=args.long_quota_retry_seconds,
        quota_refusals_before_long_retry=args.quota_refusals_before_long_retry,
        infinite_quota_retry=args.infinite_quota_retry,
        timeout_seconds=args.stage_timeout_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )

    run_step(
        "assemble",
        ["ltx-music-video", "assemble", "--manifest", str(manifest_path)],
        logger,
        max_attempts=args.retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        quota_retry_sleep_seconds=args.quota_retry_sleep_seconds,
        long_quota_retry_seconds=args.long_quota_retry_seconds,
        quota_refusals_before_long_retry=args.quota_refusals_before_long_retry,
        infinite_quota_retry=args.infinite_quota_retry,
        timeout_seconds=args.stage_timeout_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )

    run_step(
        "assemble-transitions",
        [
            "ltx-music-video",
            "assemble-transitions",
            "--manifest",
            str(manifest_path),
            "--playback-fps",
            str(args.playback_fps),
            "--output-fps",
            str(args.output_fps),
            "--transition-seconds",
            str(args.transition_seconds),
        ],
        logger,
        max_attempts=args.retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        quota_retry_sleep_seconds=args.quota_retry_sleep_seconds,
        long_quota_retry_seconds=args.long_quota_retry_seconds,
        quota_refusals_before_long_retry=args.quota_refusals_before_long_retry,
        infinite_quota_retry=args.infinite_quota_retry,
        timeout_seconds=args.stage_timeout_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )

    logger.info("Pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
