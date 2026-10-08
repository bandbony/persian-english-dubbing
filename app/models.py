from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    queued = "queued"          # waiting for transcription
    transcribing = "transcribing"
    review = "review"          # waiting for the user to check the transcript and translation
    dubbing = "dubbing"        # voice, timing, mix and optional lip sync
    done = "done"
    failed = "failed"


class Segment(BaseModel):
    id: int
    start: float
    end: float
    fa: str
    en: str = ""
    speaker: str = "S1"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Job(BaseModel):
    id: str
    filename: str
    status: Status = Status.queued
    step: str = ""
    error: str = ""
    lipsync: bool = False
    approved: bool = False     # the user reviewed the translation and started dubbing
    voice_id: str = ""
    segments: list[Segment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Paths are relative to the job directory.
    outputs: dict[str, str] = Field(default_factory=dict)
