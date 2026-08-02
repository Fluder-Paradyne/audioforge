"""Per-job workspace path layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class JobPaths:
    """Directory layout under ``work/<job_id>/`` for a single build job."""

    root: Path
    source: Path
    prepared: Path
    audio: Path
    out: Path
    job_json: Path
    job_log: Path

    @classmethod
    def for_job(cls, work_dir: Path, job_id: str) -> JobPaths:
        """Build path layout for *job_id* under *work_dir* (does not create dirs)."""
        root = Path(work_dir) / job_id
        return cls(
            root=root,
            source=root / "source",
            prepared=root / "prepared",
            audio=root / "audio",
            out=root / "out",
            job_json=root / "job.json",
            job_log=root / "job.log",
        )

    def ensure(self) -> JobPaths:
        """Create workspace directories (not ``job.json``) and return self."""
        for directory in (self.root, self.source, self.prepared, self.audio, self.out):
            directory.mkdir(parents=True, exist_ok=True)
        return self
