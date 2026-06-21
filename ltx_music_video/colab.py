from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

from .manifest import save_manifest
from .media import conditioned_video_is_valid, video_has_motion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_CONTRACT_VERSION = 3


def generation_sha256(settings: dict[str, Any]) -> str:
    payload = {
        "contract_version": GENERATION_CONTRACT_VERSION,
        "settings": settings,
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prompt_cache_sha256(
    settings: dict[str, Any], clips: list[dict[str, Any]]
) -> str:
    payload = {
        "version": 1,
        "text_encoder_repo": settings.get(
            "text_encoder_repo", "city96/t5-v1_1-xxl-encoder-bf16"
        ),
        "tokenizer_repo": settings.get(
            "tokenizer_repo",
            settings.get(
                "text_encoder_repo", "city96/t5-v1_1-xxl-encoder-bf16"
            ),
        ),
        "negative_prompt": settings["negative_prompt"],
        "clips": [
            {"id": clip["id"], "prompt": clip["prompt"]} for clip in clips
        ],
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def pending_clips(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    pending = []
    for clip in manifest["clips"]:
        if generated_clip_is_valid(clip):
            clip["status"] = "generated"
        else:
            pending.append(clip)
    return pending


def generated_clip_is_valid(clip: dict[str, Any]) -> bool:
    output = Path(clip["clip_path"])
    report = output.with_suffix(".json")
    try:
        report_data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not conditioned_video_is_valid(
        output,
        Path(clip["image_path"]),
        sample_positions=(0.0,),
    ):
        return False
    motion = report_data.get("motion_validation", {})
    minimum_motion_mad = float(motion.get("minimum_motion_mad", math.inf))
    minimum_changed_percent = float(
        motion.get("minimum_changed_percent", math.inf)
    )
    pixel_delta = int(motion.get("pixel_delta", 3))
    if not motion.get("passed") or not video_has_motion(
        output,
        minimum_motion_mad=minimum_motion_mad,
        minimum_changed_percent=minimum_changed_percent,
        pixel_delta=pixel_delta,
    ):
        return False
    expected_prompt = hashlib.sha256(clip["prompt"].encode("utf-8")).hexdigest()
    return (
        report_data.get("prompt_sha256") == expected_prompt
        and report_data.get("generation_sha256") == clip.get("generation_sha256")
    )


def build_bundle(
    manifest: dict[str, Any],
    clips: list[dict[str, Any]],
    destination: Path,
    *,
    prompt_cache_key: str,
    prompt_cache_path: Path,
    seed_attempt_offset: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    job = {
        "version": 1,
        "settings": manifest["settings"]["ltx"],
        "prompt_cache_key": prompt_cache_key,
        "clips": [],
    }
    with tarfile.open(destination, "w:gz") as archive:
        if prompt_cache_path.is_file():
            archive.add(
                prompt_cache_path,
                arcname="prompt_embeddings.pt",
                recursive=False,
            )
        for clip in clips:
            source = Path(clip["image_path"])
            remote_name = f"inputs/{clip['id']}{source.suffix.lower()}"
            archive.add(source, arcname=remote_name, recursive=False)
            job["clips"].append(
                {
                    "id": clip["id"],
                    "image_path": remote_name,
                    "prompt": clip["prompt"],
                    "seed": clip["seed"],
                    "seed_attempt_offset": seed_attempt_offset,
                    "output_name": Path(clip["clip_path"]).name,
                    "generation_sha256": clip["generation_sha256"],
                }
            )
        job_path = destination.parent / "job.json"
        job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        archive.add(job_path, arcname="job.json", recursive=False)
    return destination


def generate_pending(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    session: str,
    gpu: str,
    colab_authuser: int,
    max_attempts: int,
    batch_size: int,
    timeout_seconds: int,
    open_frontend: bool,
) -> None:
    run_dir = manifest_path.parent
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    runner = PROJECT_ROOT / "colab" / "run_colab.sh"
    retry_delay = int(os.environ.get("LTX_COLAB_RETRY_DELAY", "20"))
    max_runtime_failures = int(
        os.environ.get("LTX_COLAB_MAX_RUNTIME_FAILURES", str(max_attempts * 3))
    )
    long_retry_after = int(os.environ.get("LTX_COLAB_LONG_RETRY_AFTER", "3"))
    long_retry_seconds = int(
        os.environ.get("LTX_COLAB_LONG_RETRY_SECONDS", "1800")
    )
    infinite_allocation_retry = os.environ.get(
        "LTX_COLAB_INFINITE_ALLOCATION_RETRY", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    runtime_failures = 0
    allocation_failure_codes = {75}
    transient_runtime_codes = {75, 76}
    active_batch: list[dict[str, Any]] = []
    prompt_cache_key = ""
    prompt_cache_path: Path | None = None
    job_checkpoint = run_dir / "jobs" / "job.json"
    try:
        previous_job = json.loads(job_checkpoint.read_text(encoding="utf-8"))
        previous_key = previous_job["prompt_cache_key"]
        previous_ids = {clip["id"] for clip in previous_job["clips"]}
        previous_cache = (
            run_dir / "jobs" / f"prompt-embeddings-{previous_key[:16]}.pt"
        )
        if previous_cache.is_file():
            previous_batch = [
                clip for clip in manifest["clips"] if clip["id"] in previous_ids
            ]
            expected_key = prompt_cache_sha256(
                manifest["settings"]["ltx"], previous_batch
            )
            if previous_key == expected_key:
                active_batch = previous_batch
                prompt_cache_key = previous_key
                prompt_cache_path = previous_cache
                print(
                    f"Recovered cached active batch with "
                    f"{len(active_batch)} clip(s)"
                )
            else:
                print(
                    "Ignoring stale prompt embedding cache because the "
                    "current LTX prompt/settings contract changed"
                )
    except (OSError, ValueError, KeyError, TypeError):
        pass

    if batch_size == 0:
        print(
            "Batch size is 0: each live Colab runtime receives all pending clips "
            "so LTX weights are loaded once and reused until the runtime finishes "
            "or Colab expires it."
        )
    else:
        print(
            f"Batch size is {batch_size}: this creates artificial model-reload "
            "boundaries after each completed batch. Use --batch-size 0 to keep "
            "one loaded LTX worker processing all pending clips in a live runtime."
        )

    attempt = 1
    while attempt <= max_attempts:
        pending = pending_clips(manifest)
        save_manifest(manifest_path, manifest)
        if not pending:
            return

        pending_by_id = {clip["id"]: clip for clip in pending}
        attempt_clips = [
            pending_by_id[clip["id"]]
            for clip in active_batch
            if clip["id"] in pending_by_id
        ]
        if not attempt_clips:
            active_batch = pending if batch_size == 0 else pending[:batch_size]
            attempt_clips = list(active_batch)
            prompt_cache_key = prompt_cache_sha256(
                manifest["settings"]["ltx"], active_batch
            )
            prompt_cache_path = (
                run_dir
                / "jobs"
                / f"prompt-embeddings-{prompt_cache_key[:16]}.pt"
            )
        assert prompt_cache_path is not None
        print(
            f"Colab attempt {attempt}/{max_attempts}: "
            f"{len(pending)} clip(s) remain, sending "
            f"{len(attempt_clips)} from the active batch"
        )
        for clip in attempt_clips:
            output = Path(clip["clip_path"])
            output.unlink(missing_ok=True)
            output.with_suffix(".json").unlink(missing_ok=True)
        bundle = build_bundle(
            manifest,
            attempt_clips,
            run_dir / "jobs" / "ltx-job-current.tar.gz",
            prompt_cache_key=prompt_cache_key,
            prompt_cache_path=prompt_cache_path,
            seed_attempt_offset=(attempt - 1)
            * int(manifest["settings"]["ltx"].get("generation_attempts_per_clip", 4)),
        )
        env = {
            "COLAB_SESSION": f"{session}-a{attempt}",
            "COLAB_GPU": gpu,
            "COLAB_AUTHUSER": str(colab_authuser),
            "LTX_BUNDLE": str(bundle.resolve()),
            "LTX_LOCAL_OUTPUT_DIR": str(clips_dir.resolve()),
            "LTX_PROMPT_CACHE": str(prompt_cache_path.resolve()),
            "LTX_GENERATE_TIMEOUT": str(timeout_seconds),
            "COLAB_OPEN_FRONTEND": "1" if open_frontend else "0",
        }
        completed = subprocess.run(
            [str(runner)],
            cwd=PROJECT_ROOT,
            env={**os.environ, **env},
            check=False,
        )
        newly_finished = 0
        for clip in manifest["clips"]:
            if generated_clip_is_valid(clip):
                if clip.get("status") != "generated":
                    newly_finished += 1
                clip["status"] = "generated"
                clip.pop("error", None)
        if newly_finished:
            runtime_failures = 0
        save_manifest(manifest_path, manifest)
        if not pending_clips(manifest):
            save_manifest(manifest_path, manifest)
            return
        if (
            completed.returncode in transient_runtime_codes
            and newly_finished == 0
        ):
            runtime_failures += 1
            if (
                completed.returncode in allocation_failure_codes
                and infinite_allocation_retry
                and runtime_failures >= long_retry_after
            ):
                print(
                    "Colab T4 allocation appears quota/refusal limited after "
                    f"{runtime_failures} runtime request failure(s); entering "
                    "long-term retry mode"
                )
                print(
                    f"Waiting {long_retry_seconds} seconds before requesting "
                    "another Colab T4 runtime"
                )
                time.sleep(long_retry_seconds)
                continue
            if runtime_failures > max_runtime_failures:
                raise RuntimeError(
                    "Colab runtime/allocation failed "
                    f"{runtime_failures} time(s) without producing clips; "
                    f"{len(pending_clips(manifest))} remain"
                )
            print(
                "Colab runtime/allocation failed before producing clips; "
                "retrying without consuming a generation attempt"
            )
            print(
                f"Waiting {retry_delay} seconds before requesting "
                "another Colab runtime"
            )
            time.sleep(retry_delay)
            continue
        if completed.returncode == 0 and newly_finished == 0:
            raise RuntimeError("Colab returned success without producing any clips")
        if newly_finished == 0 and attempt == max_attempts:
            raise RuntimeError(
                f"Colab produced no new clips on final attempt; "
                f"{len(pending_clips(manifest))} remain"
            )
        if attempt < max_attempts:
            print(
                f"Waiting {retry_delay} seconds before requesting "
                "another Colab runtime"
            )
            time.sleep(retry_delay)
        attempt += 1

    remaining = len(pending_clips(manifest))
    raise RuntimeError(f"Colab generation stopped with {remaining} clip(s) remaining")
