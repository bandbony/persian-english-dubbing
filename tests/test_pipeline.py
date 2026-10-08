"""Runs the whole app on a generated video, with the paid APIs replaced by stubs."""

import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from app import config, store
from app.models import Segment
from app.pipeline import audio


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEPARATOR", "none")
    monkeypatch.setenv("CLONE_VOICE", "false")
    config.get_settings.cache_clear()

    def fake_transcribe(mp3, settings):
        return [Segment(id=0, start=0.5, end=2.0, fa="سلام"), Segment(id=1, start=3.0, end=4.5, fa="خداحافظ")]

    def fake_translate(segments, settings):
        for s in segments:
            s.en = {"سلام": "Hello", "خداحافظ": "Goodbye"}[s.fa]
        return segments

    def fake_synthesize(segments, voice_id, out_dir, settings):
        out_dir.mkdir(exist_ok=True)
        paths = []
        for s in segments:
            p = out_dir / f"{s.id:05d}.mp3"
            audio.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.8", str(p)])
            paths.append(p)
        return paths

    monkeypatch.setattr("app.pipeline.transcribe.transcribe", fake_transcribe)
    monkeypatch.setattr("app.pipeline.translate.translate", fake_translate)
    monkeypatch.setattr("app.pipeline.tts.synthesize", fake_synthesize)

    from app.main import app
    with TestClient(app) as c:
        yield c
    config.get_settings.cache_clear()


def wait_for(job_id, status, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.load(job_id)
        if job.status == status:
            return job
        assert job.status != "failed", job.error
        time.sleep(0.2)
    raise AssertionError(f"timed out, status {job.status}")


def test_upload_review_dub_download(client, tmp_path):
    video = tmp_path / "in.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=6", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video),
    ], check=True, capture_output=True)

    with video.open("rb") as f:
        r = client.post("/jobs", files={"video": ("clip.mp4", f, "video/mp4")}, follow_redirects=False)
    assert r.status_code == 303
    job_id = r.headers["location"].rsplit("/", 1)[1]

    job = wait_for(job_id, "review")
    assert [s.en for s in job.segments] == ["Hello", "Goodbye"]
    page = client.get(f"/jobs/{job_id}")
    assert "Hello" in page.text and "سلام" in page.text

    r = client.post(f"/jobs/{job_id}/segments",
                    data={"en_0": "Hi there", "en_1": "Goodbye", "fa_0": "سلام", "fa_1": "خداحافظ", "action": "dub"},
                    follow_redirects=False)
    assert r.status_code == 303

    job = wait_for(job_id, "done")
    srt = client.get(f"/files/{job_id}/english.srt").text
    assert "Hi there" in srt
    out = store.job_dir(job_id) / job.outputs["video"]
    assert abs(audio.duration(out) - 6.0) < 0.3
    streams = audio.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)])
    assert sorted(streams.split()) == ["audio", "subtitle", "video"]

    assert client.get(f"/files/{job_id}/job.json").status_code == 404
    assert client.get(f"/files/{job_id}/..%2Fjob.json").status_code == 404


def test_password(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    config.get_settings.cache_clear()
    from app.main import app
    c = TestClient(app)
    assert c.get("/").status_code == 401
    assert c.get("/", auth=("x", "wrong")).status_code == 401
    assert c.get("/", auth=("x", "s3cret")).status_code == 200
    assert c.get("/files/abc/english.srt").status_code == 404  # open path, but no such job
    config.get_settings.cache_clear()
