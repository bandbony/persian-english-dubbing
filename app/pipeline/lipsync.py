"""Lip sync with sync.so, applied only to the parts of the video where someone speaks.

sync.so bills by the length of video it processes, so we cut out the stretches with
English speech, lip sync those, and splice them back between the untouched parts.
sync.so downloads its inputs itself, so they are served by this app at public URLs.
"""

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from ..config import Settings
from .align import Placement
from .audio import run

API = "https://api.sync.so/v2"

# Pauses shorter than this stay inside one window: fewer seams and fewer jobs.
MERGE_GAP = 2.0
# Extra time around speech so mouths open and close naturally.
PAD = 0.3


def speech_windows(placements: list[Placement], total: float,
                   merge_gap: float = MERGE_GAP, pad: float = PAD) -> list[tuple[float, float]]:
    windows: list[list[float]] = []
    for p in sorted(placements, key=lambda p: p.start):
        start, end = max(0.0, p.start - pad), min(total, p.start + p.length + pad)
        if windows and start - windows[-1][1] < merge_gap:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    # Snap to the edges when the leftover piece would be tiny.
    if windows and windows[0][0] < merge_gap:
        windows[0][0] = 0.0
    if windows and total - windows[-1][1] < merge_gap:
        windows[-1][1] = total
    return [(round(s, 3), round(e, 3)) for s, e in windows]


def _video_info(path: Path) -> tuple[int, int, str]:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,r_frame_rate", "-of", "json", str(path)])
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"], s["r_frame_rate"]


def _encode_piece(src: Path, start: float | None, end: float | None, size: tuple[int, int, str], out: Path) -> Path:
    """Re-encode (a part of) a video to one common format so the pieces can be joined."""
    width, height, fps = size
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
    cmd += ["-an", "-vf", f"scale={width}:{height},setsar=1,fps={fps}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    run(cmd)
    return out


def generate(video_url: str, audio_url: str, out_mp4: Path, settings: Settings,
             poll_seconds: float = 10, timeout_seconds: float = 3 * 3600) -> Path:
    """One sync.so generation: video + audio in, lip-synced video out."""
    headers = {"x-api-key": settings.sync_api_key}
    with httpx.Client(headers=headers, timeout=60) as client:
        r = client.post(f"{API}/generate", json={
            "model": settings.sync_model,
            "input": [{"type": "video", "url": video_url}, {"type": "audio", "url": audio_url}],
            "options": {"sync_mode": "cut_off"},
        })
        r.raise_for_status()
        generation_id = r.json()["id"]

        deadline = time.monotonic() + timeout_seconds
        while True:
            g = client.get(f"{API}/generate/{generation_id}").json()
            status = g.get("status")
            if status == "COMPLETED":
                break
            if status in ("FAILED", "REJECTED"):
                raise RuntimeError(f"sync.so {status}: {g.get('error') or g}")
            if time.monotonic() > deadline:
                raise RuntimeError("sync.so took too long")
            time.sleep(poll_seconds)

        url = g.get("output_url") or g.get("outputUrl")
        with client.stream("GET", url) as download, out_mp4.open("wb") as f:
            download.raise_for_status()
            for chunk in download.iter_bytes():
                f.write(chunk)
    return out_mp4


def lipsync(video: Path, voice: Path, placements: list[Placement], total: float, work: Path,
            url_for: Callable[[Path], str], settings: Settings,
            generate_fn: Callable[..., Path] = generate) -> tuple[Path, float]:
    """Lip sync the speech windows and splice them into the original video.

    Returns the new video (no audio track) and the seconds sent to sync.so.
    """
    windows = speech_windows(placements, total)
    size = _video_info(video)

    def do(i: int, start: float, end: float) -> Path:
        clip_in = _encode_piece(video, start, end, size, work / f"lip_in_{i:03d}.mp4")
        voice_in = work / f"lip_in_{i:03d}.wav"
        run(["ffmpeg", "-y", "-i", str(voice), "-ss", f"{start:.3f}", "-to", f"{end:.3f}", str(voice_in)])
        out = generate_fn(url_for(clip_in), url_for(voice_in), work / f"lip_out_{i:03d}.mp4", settings)
        return _encode_piece(out, None, None, size, work / f"lip_piece_{i:03d}.mp4")

    with ThreadPoolExecutor(max_workers=max(1, settings.sync_concurrency)) as pool:
        synced = list(pool.map(lambda w: do(w[0], *w[1]), enumerate(windows)))

    # Untouched gaps between windows, re-encoded to match.
    pieces, cursor = [], 0.0
    for i, ((start, end), clip) in enumerate(zip(windows, synced)):
        if start - cursor > 0.01:
            pieces.append(_encode_piece(video, cursor, start, size, work / f"lip_gap_{i:03d}.mp4"))
        pieces.append(clip)
        cursor = end
    if total - cursor > 0.01:
        pieces.append(_encode_piece(video, cursor, total, size, work / "lip_gap_end.mp4"))

    listing = work / "lip_concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in pieces))
    out = work / "lipsync_video.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])

    for p in work.glob("lip_*"):
        p.unlink()
    return out, round(sum(e - s for s, e in windows), 1)
