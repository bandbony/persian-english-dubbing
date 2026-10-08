"""English subtitles and the final mix."""

from pathlib import Path

from ..models import Segment
from .align import Placement
from .audio import run


def _ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt(segments: list[Segment], placements: list[Placement] | None = None) -> str:
    """Subtitles timed to the dubbed voice when placements are given, else to the original."""
    blocks = []
    for n, seg in enumerate(segments):
        start, end = seg.start, seg.end
        if placements:
            p = placements[n]
            start, end = p.start, max(p.start + seg.duration / p.speed, seg.end)
            if n + 1 < len(placements):
                end = min(end, placements[n + 1].start)
        blocks.append(f"{n + 1}\n{_ts(start)} --> {_ts(end)}\n{seg.en}\n")
    return "\n".join(blocks)


def mix(video: Path, voice: Path, background: Path | None, subtitles: Path, total: float,
        out_mp4: Path, out_audio: Path) -> Path:
    """Voice over background, muxed into the original video with soft English subtitles."""
    if background:
        run([
            "ffmpeg", "-y", "-i", str(voice), "-i", str(background), "-filter_complex",
            "[0:a]volume=1.0[v];[1:a]volume=0.9[b];[v][b]amix=inputs=2:duration=longest:normalize=0[a]",
            "-map", "[a]", "-t", f"{total:.3f}", str(out_audio),
        ])
    else:
        run(["ffmpeg", "-y", "-i", str(voice), "-t", f"{total:.3f}", str(out_audio)])
    run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(out_audio), "-i", str(subtitles),
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
        "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng",
        "-t", f"{total:.3f}", str(out_mp4),
    ])
    return out_mp4
