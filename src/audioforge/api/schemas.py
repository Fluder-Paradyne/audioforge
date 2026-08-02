"""HTTP request/response models for the AudioForge API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from audioforge.models import JobStatus


class CreateJobRequest(BaseModel):
    """Body for ``POST /jobs`` (maps onto :class:`~audioforge.models.BuildOptions`)."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        min_length=1,
        description="Local chapter directory or fiction URL.",
    )
    voice: str = "af_heart"
    prep_model: str = "llama3.2:3b"
    skip_prep: bool = False
    resume: bool = True
    force: bool = False
    fictionreaper_bin: str = "fictionreaper"
    job_id: str | None = None
    subtitles: bool = True
    skip_align: bool = False


class CreateJobResponse(BaseModel):
    """Immediate response after accepting a job (async or sync)."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    status: JobStatus


class HealthResponse(BaseModel):
    """Liveness plus configuration hints (no live dependency probes)."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    ffmpeg_configured: bool
    ollama_base_url: str
