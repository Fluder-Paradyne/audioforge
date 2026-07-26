"""Atomic load/save of :class:`~audioforge.models.JobState` to ``job.json``."""

from __future__ import annotations

from pathlib import Path

from audioforge.models import JobState


def save_job(state: JobState, path: Path) -> None:
    """Persist *state* to *path* using a temp file + atomic replace.

    Creates parent directories as needed. On success no ``.tmp`` sibling remains.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_job(path: Path) -> JobState:
    """Load and validate a :class:`JobState` from *path*.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValidationError: If JSON does not match the schema.
        json.JSONDecodeError / ValueError: If the file is not valid JSON.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    return JobState.model_validate_json(raw)
