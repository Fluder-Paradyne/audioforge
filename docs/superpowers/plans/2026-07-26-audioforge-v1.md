# AudioForge v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local uv-managed Python package that turns FictionReaper Markdown into single-voice audiobooks (Kokoro + Ollama prep) with CLI, FastAPI, resumeable jobs, full typing, Pydantic validation, and 100% test coverage.

**Architecture:** Disk-backed stage pipeline (ingest → prep → tts → package). CLI and FastAPI call the same orchestrator. Backends behind protocols with fakes for tests.

**Tech Stack:** Python 3.12+, uv, Pydantic v2, Typer (or argparse+pydantic), FastAPI, httpx, pytest, pytest-cov, mypy strict, ruff, FFmpeg (external), Ollama (external), Kokoro TTS.

## Global Constraints

- Python `>=3.12`
- Package manager: **uv** only
- Every function parameter, return type, and meaningful variable annotated
- `mypy --strict` on `src` and `tests`
- Pydantic v2 at all CLI/API/config/job boundaries
- **100%** coverage on `audioforge` (`--cov-fail-under=100`)
- License MIT; public repo; squash merges; branch `issue/<n>-slug`
- macOS Apple Silicon primary
- No multi-voice in v1

## File map

```text
pyproject.toml
uv.lock
README.md
LICENSE
.github/workflows/ci.yml
src/audioforge/
  __init__.py              # __version__
  py.typed
  models.py                # all Pydantic models + enums
  settings.py              # AppSettings
  jobstore.py              # load/save JobState
  pipeline/
    __init__.py
    orchestrator.py        # run stages
    ingest.py
    prep.py
    tts.py
    package.py
  backends/
    __init__.py
    protocols.py           # TextPrepBackend, TtsBackend, FictionReaperRunner
    rules_prep.py
    ollama_prep.py
    kokoro_tts.py
    fake.py                # test fakes (also used in e2e unit tests)
    fictionreaper.py
  io/
    __init__.py
    chapters.py            # discover FictionReaper chapter files
    paths.py               # job workspace paths
  package_ffmpeg.py        # or pipeline/package.py internals
  cli.py
  api/
    __init__.py
    app.py
    schemas.py             # request/response models if separate from models.py
tests/
  conftest.py
  unit/
    test_models.py
    test_settings.py
    test_jobstore.py
    test_chapters.py
    test_paths.py
    test_rules_prep.py
    test_ollama_prep.py
    test_kokoro_tts.py
    test_ingest.py
    test_prep_stage.py
    test_tts_stage.py
    test_package.py
    test_orchestrator.py
    test_cli.py
    test_api.py
    test_fictionreaper.py
  fixtures/
    sample_book/
      0001-chapter-one.md
      0002-chapter-two.md
```

---

### Task 1: Scaffolding (uv, CI, coverage gate)

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `README.md`, `.github/workflows/ci.yml`, `src/audioforge/__init__.py`, `src/audioforge/py.typed`, `tests/conftest.py`, `tests/unit/test_version.py`
- Produce: installable package `audioforge`, CI green with 100% on minimal code

**Interfaces:**
- Produces: `audioforge.__version__: str`

- [ ] **Step 1:** Create `pyproject.toml` with project name `audioforge`, requires-python `>=3.12`, deps later tasks will add; dev group: pytest, pytest-cov, mypy, ruff, httpx; scripts entry `audioforge = audioforge.cli:app` (cli stub ok); tool config for ruff, mypy strict, coverage source=`audioforge` fail-under=100, branch=true optional.

- [ ] **Step 2:** MIT LICENSE; README skeleton (AudioForge, FictionReaper → audiobook, install via uv).

- [ ] **Step 3:** `src/audioforge/__init__.py` with `__version__ = "0.1.0"` and `py.typed`.

- [ ] **Step 4:** Minimal `cli.py` that exposes Typer/app or function printing version (full CLI later).

- [ ] **Step 5:** `tests/unit/test_version.py` asserts version string matches semver-ish pattern.

- [ ] **Step 6:** `uv sync --all-groups && uv run ruff check . && uv run mypy src tests && uv run pytest --cov=audioforge --cov-fail-under=100`

- [ ] **Step 7:** CI workflow running the same; commit on branch; open PR for scaffolding issue.

---

### Task 2: Domain models + settings + job store

**Files:**
- Create: `src/audioforge/models.py`, `src/audioforge/settings.py`, `src/audioforge/jobstore.py`, `src/audioforge/io/paths.py`
- Test: `tests/unit/test_models.py`, `test_settings.py`, `test_jobstore.py`, `test_paths.py`

**Interfaces:**
- Produces:
  - Enums `JobStatus`, `JobStage`
  - Models `BuildOptions`, `ChapterRef`, `ChapterProgress`, `JobState`, `ArtifactManifest`
  - `AppSettings` (pydantic-settings)
  - `JobStore.save(state: JobState, path: Path) -> None` / `load(path: Path) -> JobState`
  - `JobPaths` dataclass/model for workspace dirs

- [ ] **Step 1:** Write failing tests for model validation (invalid status, empty job_id, chapter index >= 1).

- [ ] **Step 2:** Implement models with strict types (`Path` where appropriate via validation).

- [ ] **Step 3:** Tests for `AppSettings` env prefix `AUDIOFORGE_`.

- [ ] **Step 4:** Implement jobstore atomic write (write temp + replace).

- [ ] **Step 5:** `JobPaths.for_job(root: Path, job_id: str) -> JobPaths` creating directory helpers.

- [ ] **Step 6:** Coverage 100%; commit; PR.

---

### Task 3: Ingest + FictionReaper optional wrap

**Files:**
- Create: `src/audioforge/io/chapters.py`, `src/audioforge/backends/protocols.py`, `src/audioforge/backends/fictionreaper.py`, `src/audioforge/pipeline/ingest.py`
- Test: `tests/unit/test_chapters.py`, `test_fictionreaper.py`, `test_ingest.py`
- Fixtures: `tests/fixtures/sample_book/*.md`

**Interfaces:**
- `discover_chapters(source_dir: Path) -> list[ChapterRef]` — sorted by numeric prefix `0001-*.md`
- `FictionReaperRunner.run(url: str, output_dir: Path, *, bin_path: str) -> Path`
- `ingest(source: str, paths: JobPaths, options: BuildOptions, runner: FictionReaperRunner | None) -> list[ChapterRef]`

- [ ] **Step 1:** Fixture two markdown chapters with titles in first heading.

- [ ] **Step 2:** Tests for discover order, skip non-md, title extraction.

- [ ] **Step 3:** Implement discovery.

- [ ] **Step 4:** Fake FictionReaper runner for URL path; real runner shells out to `fictionreaper download`.

- [ ] **Step 5:** Ingest copies/links sources into `paths.source` and returns chapter refs; updates job chapters.

- [ ] **Step 6:** PR.

---

### Task 4: Text prep (rules + Ollama)

**Files:**
- Create: `src/audioforge/backends/rules_prep.py`, `ollama_prep.py`, `src/audioforge/pipeline/prep.py`
- Test: `test_rules_prep.py`, `test_ollama_prep.py`, `test_prep_stage.py`

**Interfaces:**
- Protocol `TextPrepBackend`: `def prepare(self, text: str, *, options: BuildOptions) -> str`
- `RulesTextPrep`, `OllamaTextPrep` (httpx to `{base}/api/chat` or `/api/generate`)
- `prep_stage(...)` writes `prepared/NNNN-slug.txt`, updates progress

- [ ] **Step 1:** Rules: collapse blank lines, strip HTML remnants, normalize quotes; tests with fixtures.

- [ ] **Step 2:** Ollama client with injected `httpx.Client`; mock transport tests for success + HTTP error.

- [ ] **Step 3:** Stage selects Ollama unless `skip_prep` or health check fails then rules if fallback allowed.

- [ ] **Step 4:** PR.

---

### Task 5: TTS protocol + Kokoro + fake

**Files:**
- Create: `src/audioforge/backends/kokoro_tts.py`, update `protocols.py`, `fake.py`, `src/audioforge/pipeline/tts.py`
- Test: `test_kokoro_tts.py`, `test_tts_stage.py`

**Interfaces:**
- Protocol `TtsBackend`: `def synthesize(self, text: str, *, voice: str, out_path: Path) -> Path`
- `FakeTtsBackend` writes minimal valid WAV (or silent raw + header) for tests
- `KokoroTtsBackend` wraps chosen Kokoro library; unit tests mock the synthesizer object

- [ ] **Step 1:** Fake WAV writer tested.
- [ ] **Step 2:** TTS stage reads prepared text, writes `audio/NNNN-slug.wav`, resume skips existing unless force.
- [ ] **Step 3:** Kokoro backend with injectable engine; if import missing, clear error type `KokoroNotInstalledError`.
- [ ] **Step 4:** PR.

---

### Task 6: Package chapters + M4B

**Files:**
- Create: `src/audioforge/pipeline/package.py`
- Test: `test_package.py`

**Interfaces:**
- `package_book(paths: JobPaths, chapters: list[ChapterRef], *, ffmpeg: str) -> ArtifactManifest`
- Builds `out/` chapter copies or final audio list + `out/{slug}.m4b`
- FFmpeg runner injectable: `FfmpegRunner.run(args: list[str]) -> None`

- [ ] **Step 1:** Fake FFmpeg runner records commands; assert chapter metadata file / concat list generated correctly.
- [ ] **Step 2:** Real optional integration mark if ffmpeg present.
- [ ] **Step 3:** PR.

---

### Task 7: Orchestrator

**Files:**
- Create: `src/audioforge/pipeline/orchestrator.py`
- Test: `test_orchestrator.py`

**Interfaces:**
- `run_pipeline(options: BuildOptions, settings: AppSettings, *, prep: TextPrepBackend, tts: TtsBackend, ffmpeg: FfmpegRunner, fictionreaper: FictionReaperRunner | None) -> JobState`

- [ ] **Step 1:** Test full flow with fakes on sample_book → completed JobState + manifest paths set.
- [ ] **Step 2:** Test fail-fast sets status failed and error message.
- [ ] **Step 3:** Test resume skips synth when audio exists.
- [ ] **Step 4:** PR.

---

### Task 8: CLI

**Files:**
- Modify: `src/audioforge/cli.py`
- Test: `test_cli.py` via Typer CliRunner

**Interfaces:**
- Commands: build, prepare, synthesize, package, status, serve, version

- [ ] **Step 1:** Wire build to orchestrator with backend factory.
- [ ] **Step 2:** status reads jobstore.
- [ ] **Step 3:** serve imports uvicorn run of app (factory).
- [ ] **Step 4:** PR.

---

### Task 9: FastAPI

**Files:**
- Create: `src/audioforge/api/app.py`, `schemas.py`
- Test: `test_api.py` with TestClient; override backends with fakes

**Interfaces:**
- `create_app(settings: AppSettings | None = None, ...) -> FastAPI`
- POST /jobs, GET /jobs/{id}, GET /jobs/{id}/artifacts, GET /health

- [ ] **Step 1:** Tests for create job + poll completed with fake background (sync mode injectable for tests).
- [ ] **Step 2:** Health reports configured paths.
- [ ] **Step 3:** PR.

---

### Task 10: README polish + release hygiene

**Files:**
- Modify: `README.md`
- Ensure: examples, ethics, uv install, Ollama model, FFmpeg, Kokoro notes

- [ ] **Step 1:** Document end-to-end developer flow.
- [ ] **Step 2:** Confirm CI badge / coverage still 100%.
- [ ] **Step 3:** PR; tag v0.1.0 when epic complete (optional).

---

## Execution notes

- One GitHub issue per task; branch `issue/<n>-...`; squash merge.
- Prefer TDD: failing test → implement → coverage.
- Do not call real Ollama/Kokoro/network in default unit tests.
- Pin versions in pyproject; commit `uv.lock`.
