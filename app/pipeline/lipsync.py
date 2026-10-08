"""Lip sync with sync.so. sync.so downloads the inputs itself, so they need public URLs."""

import time
from pathlib import Path

import httpx

from ..config import Settings

API = "https://api.sync.so/v2"


def lipsync(video_url: str, audio_url: str, out_mp4: Path, settings: Settings,
            poll_seconds: float = 10, timeout_seconds: float = 3 * 3600) -> Path:
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
