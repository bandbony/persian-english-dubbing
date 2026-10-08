"""Persian speech to timed text segments."""

from pathlib import Path

from ..config import Settings
from ..models import Segment


def transcribe(audio_mp3: Path, settings: Settings) -> list[Segment]:
    if settings.transcriber == "openai":
        return _openai(audio_mp3, settings)
    if settings.transcriber == "whisperx":
        return _whisperx(audio_mp3)
    raise ValueError(f"unknown transcriber: {settings.transcriber}")


def _openai(audio_mp3: Path, settings: Settings) -> list[Segment]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key or None)
    # The API accepts files up to 25 MB; a 64 kbps mono mp3 is about 50 minutes.
    with audio_mp3.open("rb") as f:
        result = client.audio.transcriptions.create(
            model=settings.openai_transcribe_model,
            file=f,
            language="fa",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return _clean([(s.start, s.end, s.text) for s in result.segments or []])


def _whisperx(audio_mp3: Path) -> list[Segment]:
    import whisperx  # optional, needs a GPU to be practical

    device = "cuda"
    model = whisperx.load_model("large-v3", device, language="fa")
    audio = whisperx.load_audio(str(audio_mp3))
    result = model.transcribe(audio, batch_size=16)
    align_model, meta = whisperx.load_align_model(language_code="fa", device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device)
    return _clean([(s["start"], s["end"], s["text"]) for s in result["segments"]])


def _clean(raw: list[tuple[float, float, str]]) -> list[Segment]:
    segments = []
    for start, end, text in raw:
        text = text.strip()
        if text and end > start:
            segments.append(Segment(id=len(segments), start=round(start, 3), end=round(end, 3), fa=text))
    return segments
