# AudioForge Design

**Date:** 2026-07-26  
**Status:** Approved for implementation  
**Repo:** public GitHub `audioforge` (MIT)  
**Workflow:** issues + PRs, squash merge to `main`

## 1. Problem

FictionReaper downloads Royal Road fiction as Markdown/EPUB. Users want a **local** pipeline on Apple Silicon (M2) that turns that text into a listenable audiobook without cloud TTS.

## 2. Goals (v1)

- Ingest FictionReaper chapter Markdown (folder-first).
- Optionally wrap FictionReaper when given a Royal Road URL.
- Light **text prep** via **Ollama** (rule-based fallback when Ollama unavailable or `--skip-prep`).
- Single-voice **TTS** via **Kokoro** (pluggable backend interface).
- Output **per-chapter audio** and one **chaptered M4B**.
- **CLI + small local FastAPI**.
- Resume-friendly on-disk jobs for long books.
- Multi-voice casting deferred to a later phase (extension points only).

## 3. Non-goals (v1)

- Multi-character voices / LLM casting
- Cloud TTS or cloud LLMs
- Desktop GUI
- Windows-first support (macOS Apple Silicon primary; Linux best-effort later)
- Perfect prosody / emotional acting

## 4. Architecture

Stage pipeline; CLI and API are thin shells over one library.

```
[RR URL] --optional--> fictionreaper
                            |
[FictionReaper folder] -----+
                            v
                     Ingest -> Prep -> TTS -> Package
```

| Stage | Responsibility | Default backend |
|-------|----------------|-----------------|
| Ingest | Discover chapters + metadata; optional FictionReaper invoke | Filesystem + subprocess |
| Prep | Clean text for speech | Ollama + rule fallback |
| TTS | Synthesize per chapter | Kokoro |
| Package | Chapter audio files + chaptered M4B | FFmpeg |

### Process boundaries

- **Package:** `audioforge` (src layout)
- **CLI:** `audioforge` entry point
- **API:** FastAPI app under `audioforge.api`
- **Job state:** disk `job.json` (source of truth)

### Workspace layout (per job)

```text
work/<job-id>/
  source/           # FictionReaper-style chapter .md
  prepared/         # cleaned text per chapter
  audio/            # per-chapter audio
  out/              # final .m4b
  job.json          # JobState
```

## 5. Engineering standards (mandatory)

| Rule | Enforcement |
|------|-------------|
| **uv** | `pyproject.toml`, `uv.lock`, `uv sync`, `uv run` |
| **Python** | `>=3.12` |
| **Full typing** | Annotate all functions and meaningful variables; `mypy --strict` on `src` and `tests` |
| **Pydantic** | Validate CLI/API/config/job boundaries with Pydantic v2 models |
| **100% coverage** | `pytest` + `pytest-cov`, `--cov-fail-under=100` in CI |
| **Lint/format** | `ruff check` + `ruff format` |
| **CI** | GitHub Actions on every PR |

### Testing strategy for 100%

- Protocol/ABC backends with fakes (Ollama, Kokoro, FFmpeg, FictionReaper).
- Golden fixtures for Markdown → prepared text.
- Package tests with tiny synthetic audio; FFmpeg stubbed or optional real binary behind markers.
- Live integration tests (`@pytest.mark.integration`) optional; not required to meet the coverage gate (covered lines via fakes).

## 6. Domain models (Pydantic)

- `AppSettings` — env/defaults (`AUDIOFORGE_*`): work dir, Ollama base URL, FFmpeg path, defaults for voice/model
- `BuildOptions` — per-build knobs: source, voice, prep_model, skip_prep, resume, force, fictionreaper_bin
- `ChapterRef` — index, title, source_path, slug
- `JobStatus` — `pending | running | completed | failed | cancelled`
- `JobStage` — `ingest | prep | tts | package`
- `ChapterProgress` — prep_done, audio_done, error
- `JobState` — job_id, source, options, status, stage, chapters, error, timestamps
- `ArtifactManifest` — chapter audio paths, m4b path

Pipeline stages accept validated models only (no untyped dicts at boundaries).

## 7. CLI

| Command | Purpose |
|---------|---------|
| `audioforge build <path-or-url>` | Full pipeline |
| `audioforge prepare <path>` | Ingest + prep |
| `audioforge synthesize <path>` | TTS for prepared chapters |
| `audioforge package <path>` | Chapter audio + M4B |
| `audioforge status <job>` | Job progress |
| `audioforge serve` | Local API (default `127.0.0.1:8765`) |
| `audioforge --version` | Version |

Flags (validated into `BuildOptions` / `AppSettings`): `--output-dir`, `--voice`, `--prep-model`, `--skip-prep`, `--resume`, `--force`, `--fictionreaper-bin`.

## 8. HTTP API

| Method | Path | Behavior |
|--------|------|----------|
| `POST` | `/jobs` | Create job, start pipeline async; 202 + `job_id` |
| `GET` | `/jobs/{job_id}` | Status / progress |
| `GET` | `/jobs/{job_id}/artifacts` | Artifact paths when ready |
| `GET` | `/health` | Liveness + dependency hints |

v1: in-process background task, single-user local. Disk `job.json` is authoritative.

## 9. Backends

### Text prep

1. Rule-based: strip common RR artifacts, normalize whitespace/punctuation for TTS.
2. Ollama (default when available): rewrite chapter text for spoken narration without changing plot/dialogue meaning; temperature low; structured system prompt.

Interface: `TextPrepBackend.prepare(chapter_text: str, *, options: BuildOptions) -> str`

### TTS

Interface: `TtsBackend.synthesize(text: str, *, voice: str, out_path: Path) -> Path`

Default: Kokoro (exact package pin chosen at implementation; must run on macOS arm64).

### Package

FFmpeg: concat chapter audio, embed chapter markers, produce `.m4b` (AAC in M4B/M4A container).

## 10. Resume & errors

- `--resume` (default true for API rebuilds of same job dir): skip chapters with completed audio.
- `--force`: rebuild all.
- Default **fail-fast** on first chapter error; error recorded on `JobState` and chapter progress.
- Missing FictionReaper when URL given → clear error with install hint.
- Missing FFmpeg / Kokoro weights → fail at validation with install hints.
- Ollama down → rule fallback unless configured to require Ollama.

## 11. Multi-voice extension points (not v1)

- Future `CastBackend` after prep producing speaker-tagged segments.
- `TtsBackend` extended to accept segment + voice map.
- No schema changes that block adding `segments/` later.

## 12. Repo & workflow

- **Name:** `audioforge`
- **Visibility:** public
- **License:** MIT
- **Default branch:** `main`
- **Merges:** squash
- **Branches:** `issue/<n>-short-slug`
- **PRs:** one issue per PR; CI green required

### Issue breakdown (epic children)

1. Scaffolding (uv, package, CI, typing, coverage gate)
2. Domain models + job store
3. Ingest (+ optional FictionReaper wrap)
4. Text prep (rules + Ollama)
5. TTS interface + Kokoro
6. Package (chapters + M4B)
7. CLI
8. FastAPI
9. End-to-end fake-backend test + README

## 13. Success criteria (v1)

- From a FictionReaper-style chapter folder, `audioforge build` produces chapter audio + chaptered M4B using fakes in CI and real Kokoro/Ollama on a developer M2 when available.
- `mypy --strict` and **100%** line coverage on `src/audioforge` pass in CI.
- Public docs explain install (uv), Ollama model pull, FFmpeg, and ethics (personal offline use; respect RR ToS/copyright).

## 14. Ethics

Same posture as FictionReaper: personal archival/offline listening of content the user may access; be polite to remote sites; respect copyright and Royal Road ToS.
