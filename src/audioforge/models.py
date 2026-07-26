"""Domain models and enums for AudioForge jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class JobStatus(StrEnum):
    """Lifecycle status of a build job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(StrEnum):
    """Pipeline stage currently active or last reached."""

    INGEST = "ingest"
    PREP = "prep"
    TTS = "tts"
    PACKAGE = "package"


class BuildOptions(BaseModel):
    """Per-build knobs validated at CLI/API boundaries."""

    model_config = ConfigDict(extra="forbid")

    source: str
    voice: str = "af_heart"
    prep_model: str = "llama3.2:3b"
    skip_prep: bool = False
    resume: bool = True
    force: bool = False
    fictionreaper_bin: str = "fictionreaper"
    output_dir: Path | None = None
    job_id: str | None = None


class ChapterRef(BaseModel):
    """Reference to a discovered FictionReaper chapter file."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1)
    title: str
    source_path: Path
    slug: str


class ChapterProgress(BaseModel):
    """Per-chapter completion flags and optional error."""

    model_config = ConfigDict(extra="forbid")

    chapter_index: int = Field(ge=1)
    prep_done: bool = False
    audio_done: bool = False
    error: str | None = None


class JobState(BaseModel):
    """Authoritative on-disk job state (``job.json``)."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    source: str
    options: BuildOptions
    status: JobStatus
    stage: JobStage | None = None
    chapters: list[ChapterRef] = Field(default_factory=list)
    progress: list[ChapterProgress] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class ArtifactManifest(BaseModel):
    """Paths to produced audio artifacts for a job."""

    model_config = ConfigDict(extra="forbid")

    chapter_audio: list[Path]
    m4b_path: Path | None = None
