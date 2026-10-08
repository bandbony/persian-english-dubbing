import base64
import logging
import secrets
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from . import store, worker
from .config import get_settings
from .models import Job, Status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
# Files that may be served from a job folder (downloads, and inputs fetched by sync.so).
SERVABLE = {"dubbed.mp4", "dubbed_lipsync.mp4", "english.srt", "audio_en.m4a", "voice_en.wav"} | {
    f"source{ext}" for ext in VIDEO_TYPES
}

STATUS_FA = {
    Status.queued: "در صف",
    Status.transcribing: "در حال پردازش",
    Status.review: "آماده‌ی بازبینی",
    Status.dubbing: "در حال دوبله",
    Status.done: "تمام شد",
    Status.failed: "خطا",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker.start()
    yield


app = FastAPI(title="Persian → English dubbing", lifespan=lifespan)


@app.middleware("http")
async def password(request: Request, call_next):
    expected = get_settings().app_password
    # /files/ stays open so sync.so can fetch inputs; job ids are random and unguessable.
    if not expected or request.url.path.startswith("/files/"):
        return await call_next(request)
    try:
        scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
        given = base64.b64decode(encoded).decode().partition(":")[2] if scheme.lower() == "basic" else ""
    except ValueError:
        given = ""
    if secrets.compare_digest(given.encode(), expected.encode()):
        return await call_next(request)
    return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="dubbing"'})


templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.globals["STATUS_FA"] = STATUS_FA


def _job(job_id: str) -> Job:
    job = store.load(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"jobs": store.all_jobs()})


@app.post("/jobs")
def create_job(video: UploadFile = File(...), lipsync: bool = Form(False)):
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_TYPES:
        raise HTTPException(400, f"Unsupported file type {suffix!r}")
    job = store.new_job(video.filename or f"video{suffix}", lipsync)
    with worker.source_video(job).open("wb") as f:
        shutil.copyfileobj(video.file, f)
    worker.enqueue(job.id, "prepare")
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.get("/jobs/{job_id}")
def job_page(request: Request, job_id: str):
    return templates.TemplateResponse(request, "job.html", {"job": _job(job_id)})


@app.get("/api/jobs/{job_id}")
def job_json(job_id: str) -> Job:
    return _job(job_id)


@app.post("/jobs/{job_id}/segments")
async def save_segments(request: Request, job_id: str):
    """Save the edited transcript/translation; with action=dub, also start dubbing."""
    job = _job(job_id)
    if job.status != Status.review:
        raise HTTPException(409, "This job is not waiting for review")
    form = await request.form()
    for seg in job.segments:
        seg.fa = str(form.get(f"fa_{seg.id}", seg.fa)).strip()
        seg.en = str(form.get(f"en_{seg.id}", seg.en)).strip()
    job.segments = [s for s in job.segments if s.en]
    job.lipsync = form.get("lipsync") == "on"
    if form.get("action") == "dub":
        job.approved = True
        job.status, job.step = Status.dubbing, "در صف دوبله"
        store.save(job)
        worker.enqueue(job.id, "dub")
    else:
        store.save(job)
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.post("/jobs/{job_id}/retry")
def retry(job_id: str):
    job = _job(job_id)
    if job.status != Status.failed:
        raise HTTPException(409, "Only failed jobs can be retried")
    # A job that failed after review keeps its translation and only redoes the dub.
    phase = "dub" if job.approved else "prepare"
    job.status, job.error = (Status.dubbing if phase == "dub" else Status.queued), ""
    store.save(job)
    worker.enqueue(job.id, phase)
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.get("/files/{job_id}/{name}")
def files(job_id: str, name: str):
    if name not in SERVABLE:
        raise HTTPException(404)
    _job(job_id)
    path = store.job_dir(job_id) / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, filename=name)
