"""Jobs are stored as one folder per job with a job.json inside."""

import threading
import uuid
from pathlib import Path

from .config import get_settings
from .models import Job

_lock = threading.Lock()


def jobs_root() -> Path:
    root = get_settings().data_dir / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dir(job_id: str) -> Path:
    # Job ids are hex uuids; refuse anything else so a URL can't escape the data dir.
    if not job_id.isalnum():
        raise ValueError("bad job id")
    return jobs_root() / job_id


def new_job(filename: str, lipsync: bool) -> Job:
    job = Job(id=uuid.uuid4().hex, filename=filename, lipsync=lipsync)
    job_dir(job.id).mkdir(parents=True)
    save(job)
    return job


def save(job: Job) -> None:
    path = job_dir(job.id) / "job.json"
    tmp = path.with_suffix(".tmp")
    with _lock:
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)


def load(job_id: str) -> Job | None:
    try:
        path = job_dir(job_id) / "job.json"
    except ValueError:
        return None
    if not path.exists():
        return None
    return Job.model_validate_json(path.read_text(encoding="utf-8"))


def all_jobs() -> list[Job]:
    jobs = [load(p.name) for p in jobs_root().iterdir() if p.is_dir()]
    return sorted((j for j in jobs if j), key=lambda j: j.created_at, reverse=True)
