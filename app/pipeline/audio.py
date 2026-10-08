"""ffmpeg helpers and vocal/background separation."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

SAMPLE_RATE = 44100


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr[-2000:]}")
    return proc.stdout


def duration(path: Path) -> float:
    out = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    return float(json.loads(out)["format"]["duration"])


def extract_audio(video: Path, out_wav: Path) -> Path:
    run([
        "ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "2",
        "-ar", str(SAMPLE_RATE), str(out_wav),
    ])
    return out_wav


def to_mono_mp3(wav: Path, out_mp3: Path) -> Path:
    """Small mono copy for upload to the transcription and cloning APIs."""
    run(["ffmpeg", "-y", "-i", str(wav), "-ac", "1", "-ar", "16000", "-b:a", "64k", str(out_mp3)])
    return out_mp3


def separate(audio: Path, work: Path, method: str) -> tuple[Path, Path | None]:
    """Split speech from music/effects. Returns (vocals, background or None)."""
    if method == "demucs":
        out = work / "demucs"
        run([
            sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
            "-o", str(out), str(audio),
        ])
        stem_dir = out / "htdemucs" / audio.stem
        vocals, background = work / "vocals.wav", work / "background.wav"
        shutil.move(stem_dir / "vocals.wav", vocals)
        shutil.move(stem_dir / "no_vocals.wav", background)
        shutil.rmtree(out, ignore_errors=True)
        return vocals, background
    if method == "none":
        return audio, None
    raise ValueError(f"unknown separator: {method}")


def cut(audio: Path, start: float, end: float, out: Path) -> Path:
    run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(audio), str(out)])
    return out


def concat(parts: list[Path], out: Path) -> Path:
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])
    listing.unlink()
    return out
