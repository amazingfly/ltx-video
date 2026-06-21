from __future__ import annotations

import base64
import json
import mimetypes
import re
import subprocess
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image


SYSTEM_PROMPT = """Look at the image and write one concise prompt for two seconds
of clearly visible scene motion and lighting or atmospheric effects. Describe
contained dance-like motion while the subject keeps the same identity, location,
and overall composition. Prefer in-place body rhythm: a hip sway, shoulder roll,
chest bounce, torso dip, pose pulse, or dramatic battle stance. Pair the body
motion with visible hair and clothing fabric movement when applicable. Never
describe walking, running, jumping, stepping, turning away, leaving the pose, or
single-limb-only motion like moving one arm. Never mention camera movement,
panning, reframing, zooming, or focus changes.

Use one clearly perceptible dance/body motion plus hair or loose fabric motion.
Add one or two effects that visibly evolve through the shot: flashing neon,
shifting colored light, curling fog, smoke, dust, sparks, rain, or pulsing
energy. Preserve every subject, object, outfit, and background. No scene change
and no new objects. Avoid words such as subtle, barely, faint, tiny, still,
static, or imperceptible. Use 16-30 words. Return only the prompt.

Good examples:
The woman sways her hips in place as long hair and loose fabric whip gently
while neon signs flare behind her.

The green fighter pulses into a battle stance as jacket fabric snaps, hair
shakes, and glowing dust swirls through the air.

The armored woman rolls her shoulders in place as cape edges ripple, hair
flicks, and energy sparks flash near her feet.

The pink-haired woman bounces rhythmically in place as loose hair tips and skirt
fabric sway while green energy bands pulse."""

FALLBACK_PROMPT = (
    "The subject blinks softly as loose hair or fabric stirs, existing lights "
    "pulse, and faint atmospheric particles drift through the scene."
)

FORBIDDEN_CAMERA_MOTION = re.compile(
    r"\b(?:camera|pan(?:s|ned|ning)?|zoom(?:s|ed|ing)?|dolly|trucking)\b"
    r"|\b(?:push[- ]?in|pull[- ]?back|rack focus)\b"
    r"|\b(?:view|frame|framing)\s+(?:moves?|shifts?|drifts?)\b",
    flags=re.IGNORECASE,
)

FORBIDDEN_LARGE_MOTION = re.compile(
    r"\b(?:walk(?:s|ed|ing)?|run(?:s|ning)?|turn(?:s|ed|ing)?|"
    r"flex(?:es|ed|ing)?|step(?:s|ped|ping)?|jump(?:s|ed|ing)?|"
    r"lean(?:s|ed|ing)?|reach(?:es|ed|ing)?|"
    r"(?:lift|raise)(?:s|d|ing)?\s+(?:an?\s+)?arm)\b"
    r"|\bshift(?:s|ed|ing)?\s+(?:his|her|their|its)?\s*weight\b",
    flags=re.IGNORECASE,
)

LTX_PROMPT_PREFIX = (
    "The camera remains locked with unchanged framing. Preserve the original "
    "subject's identity, appearance, clothing, location, and background while "
    "the described contained subject motion happens clearly. "
)
LTX_PROMPT_SUFFIX = (
    " Make the contained body, hair, fabric, lighting, and atmospheric "
    "effects clearly perceptible throughout the shot; no camera movement, "
    "reframing, new subjects, or scene change."
)


def build_ltx_prompt(motion_prompt: str) -> str:
    validate_motion_prompt(motion_prompt)
    return f"{LTX_PROMPT_PREFIX}{motion_prompt}{LTX_PROMPT_SUFFIX}"


class GemmaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        start_command: Path | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.start_command = start_command
        self.timeout = timeout
        self.started_server = False

    def ensure_ready(self) -> None:
        if self._healthy():
            return
        if self.start_command is None:
            raise RuntimeError(f"Gemma server is not reachable at {self.base_url}")
        subprocess.run([str(self.start_command)], check=True)
        self.started_server = True
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self._healthy():
                return
            time.sleep(1)
        raise RuntimeError(f"Gemma server did not become ready at {self.base_url}")

    def describe_motion(self, image_path: Path) -> str:
        mime_type, image_bytes = prepare_image(image_path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        correction = ""
        last_error: RuntimeError | None = None
        for _ in range(3):
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT + correction,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{encoded}"
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0.25,
                "top_p": 0.9,
                "max_tokens": 96,
            }
            request = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
            try:
                text = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Unexpected Gemma response: {body}") from exc
            prompt = clean_prompt(text)
            try:
                validate_motion_prompt(prompt)
                return prompt
            except RuntimeError as exc:
                last_error = exc
                correction = (
                    "\nYour previous answer violated the rules. Rewrite it without "
                    "camera language, walking, jumping, stepping, turning away, "
                    "or single-limb-only motion. Keep the subject in place and "
                    "describe contained dance/body motion with hair, fabric, "
                    "lighting, fog, smoke, dust, sparks, rain, or energy."
                )
        raise last_error or RuntimeError("Gemma did not return a usable prompt")

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/health", timeout=2
            ) as response:
                data = json.load(response)
            return data.get("status") == "ok"
        except (OSError, ValueError, urllib.error.URLError):
            return False


def clean_prompt(text: str) -> str:
    prompt = " ".join(text.strip().split())
    prompt = re.sub(
        r"^(?:prompt|image-to-video prompt|video prompt)\s*:\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    return prompt.strip(" \"'`")


def validate_motion_prompt(prompt: str) -> None:
    word_count = len(prompt.split())
    if word_count < 12 or word_count > 36:
        raise RuntimeError(
            f"Gemma prompt must contain 12-36 words, got {word_count}: {prompt!r}"
        )
    if FORBIDDEN_CAMERA_MOTION.search(prompt):
        raise RuntimeError(f"Gemma prompt contains camera motion: {prompt!r}")
    if FORBIDDEN_LARGE_MOTION.search(prompt):
        raise RuntimeError(f"Gemma prompt contains large body motion: {prompt!r}")


def prepare_image(path: Path, max_edge: int = 384) -> tuple[str, bytes]:
    guessed_type = mimetypes.guess_type(path.name)[0] or "image/png"
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    mime_type = "image/jpeg" if guessed_type.startswith("image/") else guessed_type
    return mime_type, buffer.getvalue()
