"""Fit each English clip into the time slot of its Persian line and build one voice track."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..models import Segment
from .audio import SAMPLE_RATE


@dataclass
class Placement:
    start: float   # seconds into the video
    speed: float   # playback speed, 1.0 = unchanged
    length: float = 0.0  # seconds of audio after the speed change


def plan(segments: list[Segment], clip_lengths: list[float], total: float, max_speedup: float) -> list[Placement]:
    """Decide where each clip starts and how much to speed it up.

    A clip may use its own slot plus any silence before the next line. If it is still
    too long it is sped up, up to max_speedup. Beyond that it runs over and the following
    lines start late; the delay is absorbed by the next pause in speech.
    """
    placements = []
    cursor = 0.0
    for i, (seg, length) in enumerate(zip(segments, clip_lengths)):
        start = max(seg.start, cursor)
        next_start = segments[i + 1].start if i + 1 < len(segments) else total
        room = max(next_start - start, 0.05)
        speed = min(max(length / room, 1.0), max_speedup)
        placements.append(Placement(start=round(start, 3), speed=round(speed, 3), length=round(length / speed, 3)))
        cursor = start + length / speed
    return placements


def decode(path: Path, speed: float = 1.0) -> np.ndarray:
    """Decode to mono float32 at SAMPLE_RATE, time-stretched without changing pitch."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if abs(speed - 1.0) > 1e-3:
        cmd += ["-filter:a", f"atempo={speed:.4f}"]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def build_track(segments: list[Segment], clips: list[Path], total: float, max_speedup: float,
                out_wav: Path) -> list[Placement]:
    lengths = [len(decode(c)) / SAMPLE_RATE for c in clips]
    placements = plan(segments, lengths, total, max_speedup)
    audio = [decode(c, p.speed) for c, p in zip(clips, placements)]
    end = max([total] + [p.start + len(a) / SAMPLE_RATE for p, a in zip(placements, audio)])
    track = np.zeros(int(end * SAMPLE_RATE) + 1, dtype=np.float32)
    for p, a in zip(placements, audio):
        i = int(p.start * SAMPLE_RATE)
        track[i:i + len(a)] += a
    sf.write(out_wav, np.clip(track, -1, 1), SAMPLE_RATE)
    return placements
