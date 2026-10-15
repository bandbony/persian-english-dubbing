"""Persian to English translation with Claude, sized to fit each line's time slot."""

import json

import anthropic
from pydantic import BaseModel

from ..config import Settings
from ..models import Segment

# Natural English speech runs at roughly 14-16 characters per second.
CHARS_PER_SECOND = 15
BATCH = 60

SYSTEM = """You translate Persian video transcripts into English for dubbing.

Each line will be spoken by an English voice in the same time slot as the Persian original, so:
- Keep each translation close to its character budget (max_chars). Going over makes the voice rush; prefer a shorter natural phrasing over a literal one.
- Translate meaning and tone, not word by word. Keep names, numbers and technical terms accurate.
- Use the full transcript for context (who is speaking, what "it" refers to), but return exactly one translation per requested id.
- The transcript comes from speech recognition and may contain errors; translate what was most likely said.
- Return natural spoken English, no notes or brackets."""


class Line(BaseModel):
    id: int
    en: str


class Translation(BaseModel):
    lines: list[Line]


def budget(seg: Segment) -> int:
    return max(8, round(seg.duration * CHARS_PER_SECOND))


def translate(segments: list[Segment], settings: Settings) -> list[Segment]:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    context = "\n".join(f"[{s.id}] {s.fa}" for s in segments)
    by_id = {s.id: s for s in segments}

    for i in range(0, len(segments), BATCH):
        batch = segments[i:i + BATCH]
        request = "\n".join(
            json.dumps({"id": s.id, "seconds": round(s.duration, 1), "max_chars": budget(s), "fa": s.fa},
                       ensure_ascii=False)
            for s in batch
        )
        response = client.beta.messages.parse(
            model=settings.claude_model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": settings.claude_effort},
            output_format=Translation,
            system=SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"<transcript>\n{context}\n</transcript>\n\n"
                    f"Translate these lines:\n{request}"
                ),
            }],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined to translate part of this transcript.")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise RuntimeError("Translation came back incomplete.")
        for line in response.parsed_output.lines:
            if line.id in by_id:
                by_id[line.id].en = line.en.strip()

    missing = [s.id for s in segments if not s.en]
    if missing:
        raise RuntimeError(f"No translation for lines {missing}")
    return segments
