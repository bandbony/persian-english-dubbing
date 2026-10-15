import shutil
import subprocess

from app.config import Settings
from app.pipeline import audio
from app.pipeline.align import Placement
from app.pipeline.lipsync import lipsync, speech_windows


def p(start, length):
    return Placement(start=start, speed=1.0, length=length)


def test_windows_skip_long_silences():
    w = speech_windows([p(10, 5), p(100, 5)], total=600)
    assert w == [(9.7, 15.3), (99.7, 105.3)]


def test_short_pauses_merge_and_edges_snap():
    w = speech_windows([p(1, 3), p(4.5, 2), p(590, 9)], total=600)
    assert w == [(0.0, 6.8), (589.7, 600.0)]


def test_lipsync_only_sends_speech_and_keeps_length(tmp_path):
    video = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=12",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
    voice = tmp_path / "voice.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=12", str(voice)],
                   check=True, capture_output=True)
    sent = []

    def fake_generate(video_url, audio_url, out, settings):
        sent.append((video_url, audio_url))
        shutil.copy(video_url.removeprefix("file://"), out)
        return out

    out, seconds = lipsync(video, voice, [p(3, 2), p(8, 1)], 12.0, tmp_path,
                           lambda path: f"file://{path}", Settings(), generate_fn=fake_generate)
    assert len(sent) == 2 and sent[0][0].endswith("lip_in_000.mp4") and sent[0][1].endswith("lip_in_000.wav")
    assert seconds == 4.2
    assert abs(audio.duration(out) - 12.0) < 0.2
    assert not list(tmp_path.glob("lip_*"))
