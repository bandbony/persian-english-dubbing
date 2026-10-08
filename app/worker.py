"""Background worker. Runs the two halves of the pipeline:

1. prepare: extract audio, separate vocals, transcribe, translate -> status "review"
2. dub: voice, timing, mix, subtitles, optional lip sync -> status "done"

The user checks and edits the transcript and translation between the two.
"""

import logging
import queue
import threading
import traceback
from pathlib import Path

from . import store
from .config import get_settings
from .models import Job, Status
from .pipeline import align, audio, output, transcribe, translate, tts

log = logging.getLogger(__name__)

_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()


def enqueue(job_id: str, phase: str) -> None:
    _queue.put((job_id, phase))


def _step(job: Job, text: str) -> None:
    job.step = text
    store.save(job)
    log.info("job %s: %s", job.id, text)


def source_video(job: Job) -> Path:
    return store.job_dir(job.id) / f"source{Path(job.filename).suffix.lower() or '.mp4'}"


def prepare(job: Job) -> None:
    s = get_settings()
    work = store.job_dir(job.id)
    job.status = Status.transcribing

    _step(job, "استخراج صدا")
    original = audio.extract_audio(source_video(job), work / "original.wav")

    _step(job, "جدا کردن گفتار از موسیقی و افکت")
    vocals, _ = audio.separate(original, work, s.separator)

    _step(job, "تبدیل گفتار فارسی به متن")
    speech_mp3 = audio.to_mono_mp3(vocals, work / "speech.mp3")
    job.segments = transcribe.transcribe(speech_mp3, s)
    if not job.segments:
        raise RuntimeError("No speech found in this video.")
    store.save(job)

    _step(job, "ترجمه به انگلیسی")
    job.segments = translate.translate(job.segments, s)

    job.status = Status.review
    _step(job, "منتظر بازبینی شما")


def dub(job: Job) -> None:
    s = get_settings()
    work = store.job_dir(job.id)
    video = source_video(job)
    total = audio.duration(video)
    vocals = work / "vocals.wav" if (work / "vocals.wav").exists() else work / "original.wav"
    background = work / "background.wav" if (work / "background.wav").exists() else None
    job.status = Status.dubbing

    voice_id, cloned = s.elevenlabs_default_voice_id, False
    if s.clone_voice:
        _step(job, "کلون کردن صدای گوینده")
        try:
            # About two minutes of clean speech is plenty for an instant clone.
            sample = audio.cut(vocals, 0, min(total, 120), work / "clone_sample.mp3")
            voice_id, cloned = tts.clone_voice(sample, f"dub-{job.id[:8]}", s), True
        except Exception:
            log.exception("voice clone failed, using default voice")
    job.voice_id = voice_id

    try:
        _step(job, "ساخت صدای انگلیسی")
        clips = tts.synthesize(job.segments, voice_id, work / "clips", s)
    finally:
        if cloned:
            tts.delete_voice(voice_id, s)

    _step(job, "همزمان‌سازی با زمان‌بندی اصلی")
    placements = align.build_track(job.segments, clips, total, s.max_speedup, work / "voice_en.wav")

    _step(job, "میکس صدا و ساخت زیرنویس")
    subs = work / "english.srt"
    subs.write_text(output.srt(job.segments, placements), encoding="utf-8")
    dubbed = output.mix(video, work / "voice_en.wav", background, subs, total,
                        work / "dubbed.mp4", work / "audio_en.m4a")
    job.outputs = {"video": dubbed.name, "subtitles": subs.name, "audio": "audio_en.m4a"}

    if job.lipsync:
        _step(job, "لب‌خوانی با sync.so")
        from .pipeline import lipsync

        base = s.public_base_url.rstrip("/")
        if not base:
            raise RuntimeError("Lip sync needs PUBLIC_BASE_URL so sync.so can download the files.")
        out, job.lipsync_seconds = lipsync.lipsync(
            video, work / "voice_en.wav", placements, total, work,
            lambda p: f"{base}/files/{job.id}/{p.name}", s,
        )
        # sync.so gets the voice alone, which lip syncs best; put the full mix and subtitles back on.
        output.mix(out, work / "voice_en.wav", background, subs, total,
                   work / "dubbed_lipsync.mp4", work / "audio_en.m4a")
        job.outputs["video"] = "dubbed_lipsync.mp4"

    job.status = Status.done
    _step(job, "تمام شد")


def _run(job_id: str, phase: str) -> None:
    job = store.load(job_id)
    if not job:
        return
    try:
        (prepare if phase == "prepare" else dub)(job)
    except Exception as e:
        log.error("job %s failed:\n%s", job_id, traceback.format_exc())
        job.status, job.error = Status.failed, f"{job.step}: {e}"
        store.save(job)


def _loop() -> None:
    while True:
        job_id, phase = _queue.get()
        _run(job_id, phase)
        _queue.task_done()


def start() -> None:
    """Start the worker and pick up jobs that were interrupted by a restart."""
    for job in store.all_jobs():
        if job.status in (Status.queued, Status.transcribing):
            enqueue(job.id, "prepare")
        elif job.status == Status.dubbing:
            enqueue(job.id, "dub")
    threading.Thread(target=_loop, daemon=True, name="dub-worker").start()
