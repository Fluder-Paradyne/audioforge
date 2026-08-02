"""Pipeline stages: ingest, prep, tts, align, package, orchestrator."""

from __future__ import annotations

from audioforge.pipeline.align import AlignStageError, align_chapters
from audioforge.pipeline.ingest import IngestError, ingest
from audioforge.pipeline.orchestrator import (
    PipelineError,
    resolve_job_paths,
    run_package,
    run_pipeline,
    run_prepare,
    run_synthesize,
)
from audioforge.pipeline.package import PackageError, package_book
from audioforge.pipeline.prep import PrepError, prep_chapters, select_prep_backend
from audioforge.pipeline.subtitles import SubtitleError, build_book_vtt
from audioforge.pipeline.tts import TtsError, synthesize_chapters

__all__ = [
    "AlignStageError",
    "IngestError",
    "PackageError",
    "PipelineError",
    "PrepError",
    "SubtitleError",
    "TtsError",
    "align_chapters",
    "build_book_vtt",
    "ingest",
    "package_book",
    "prep_chapters",
    "resolve_job_paths",
    "run_package",
    "run_pipeline",
    "run_prepare",
    "run_synthesize",
    "select_prep_backend",
    "synthesize_chapters",
]
