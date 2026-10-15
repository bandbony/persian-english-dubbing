"""English speech with ElevenLabs, optionally in a clone of the original speaker's voice."""

from pathlib import Path

import httpx

from ..config import Settings
from ..models import Segment

API = "https://api.elevenlabs.io/v1"


def _client(settings: Settings) -> httpx.Client:
    return httpx.Client(base_url=API, headers={"xi-api-key": settings.elevenlabs_api_key}, timeout=120)


def clone_voice(sample_mp3: Path, name: str, settings: Settings) -> str:
    """Instant voice clone from a clean vocals sample. Returns the new voice id."""
    with _client(settings) as client, sample_mp3.open("rb") as f:
        r = client.post(
            "/voices/add",
            data={"name": name, "description": "Created by persian-english-dubbing"},
            files={"files": (sample_mp3.name, f, "audio/mpeg")},
        )
        r.raise_for_status()
        return r.json()["voice_id"]


def delete_voice(voice_id: str, settings: Settings) -> None:
    with _client(settings) as client:
        client.delete(f"/voices/{voice_id}")


def synthesize(segments: list[Segment], voice_id: str, out_dir: Path, settings: Settings) -> list[Path]:
    """One mp3 per segment, named by segment id."""
    out_dir.mkdir(exist_ok=True)
    paths = []
    with _client(settings) as client:
        for i, seg in enumerate(segments):
            r = client.post(
                f"/text-to-speech/{voice_id}",
                params={"output_format": "mp3_44100_128"},
                json={
                    "text": seg.en,
                    "model_id": settings.elevenlabs_model,
                    # Neighbouring lines keep intonation consistent across clips.
                    "previous_text": segments[i - 1].en if i > 0 else None,
                    "next_text": segments[i + 1].en if i + 1 < len(segments) else None,
                },
            )
            r.raise_for_status()
            path = out_dir / f"{seg.id:05d}.mp3"
            path.write_bytes(r.content)
            paths.append(path)
    return paths
