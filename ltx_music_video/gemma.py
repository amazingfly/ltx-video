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
localized motion while the subject keeps the same identity, location, and
overall composition. Generally describe a dynamic motion, a single elaborate dance move, full body motions, bouncing, spinning, twirling, moving hips, a single gymnastic move, a dramatic battle pose. Focus on the move, and the movement of hair and or clothes if applicable. never use simple single limb motions like moving one arm. Never mention camera movement,
panning reframing, or focus changes.

Use one small but clearly perceptible character motion such as hair or loose
fabric blowing, blinking, breathing, or fingers tightening. Add one or two
effects that visibly evolve through the shot: flashing neon, shifting colored
light, curling fog, smoke, dust, sparks, rain, or pulsing energy. Preserve every
subject, object, outfit, and background. No scene change and no new objects.
Avoid words such as subtle, barely, faint, tiny, still, static, or
imperceptible. Use 16-30 words. Return only the prompt.

Good examples:
The woman's long hair and loose fabric blow gently in the wind as neon signs
flare behind her and thin fog curls around her boots.

The green fighter takes a slow breath as city lights flicker behind him and
glowing dust swirls softly through the air.

The armored woman's cape edge and loose hair ripple gently as energy sparks
flash near her feet and colored light rolls across her armor.

The pink-haired woman's loose hair tips sway as green energy bands pulse and
neon reflections shimmer across her unchanged pose."""

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
    "the described local motion happens clearly. "
)
LTX_PROMPT_SUFFIX = (
    " Make the local character motion and evolving lighting or atmospheric "
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
                    "camera language or any whole-body movement. Keep the subject "
                    "fixed and describe only hair, fabric, blinking, breathing, "
                    "fingers, lighting, fog, smoke, dust, sparks, rain, or energy."
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
