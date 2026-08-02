# AudioForge

[![CI](https://github.com/Fluder-Paradyne/audioforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Fluder-Paradyne/audioforge/actions/workflows/ci.yml)

Local **FictionReaper Markdown → single-voice audiobook** pipeline for Apple Silicon.

AudioForge turns FictionReaper-style chapter Markdown (or a fiction URL via FictionReaper) into listenable audiobooks using **local** tooling only:

1. **Ingest** — chapter folder, or optional FictionReaper download for a URL  
2. **Prep** — light text cleanup for speech (Ollama when available; rule-based fallback)  
3. **TTS** — single-voice synthesis via Kokoro  
4. **Package** — per-chapter audio + one chaptered **M4B** via FFmpeg  

CLI and a small local FastAPI surface share the same disk-backed job pipeline (`work/<job-id>/`).

> **Status:** v0.1.0 (alpha). Single-voice pipeline, CLI, and local jobs API are in place. Multi-voice is future work.

## Features (v1)

- Folder-first ingest of FictionReaper chapter Markdown (`NNNN-slug.md`)
- Optional FictionReaper wrap when `source` is a URL
- Text prep: Ollama (default model `llama3.2:3b`) with rules fallback / `--skip-prep`
- Single-voice TTS via optional Kokoro extra (`af_heart` default voice)
- Chapter audio + chaptered M4B packaging (FFmpeg)
- Resume-friendly on-disk jobs (`job.json`, `--resume` / `--force`)
- CLI: `build`, `prepare`, `synthesize`, `package`, `status`, `serve`
- Local FastAPI: `POST /jobs`, job status, artifacts, `/health`
- 100% test coverage gate, `mypy --strict`, Ruff

## Requirements

| Requirement | Notes |
|-------------|--------|
| **macOS Apple Silicon** | Primary target (M-series). Linux is best-effort. |
| **Python ≥ 3.12** | Managed via [uv](https://docs.astral.sh/uv/) |
| **[FFmpeg](https://ffmpeg.org/)** | On `PATH` (or set `AUDIOFORGE_FFMPEG_PATH`) for packaging |
| **[Ollama](https://ollama.com/)** (optional) | For LLM text prep; without it, rules prep is used |
| **FictionReaper** (optional) | Only needed when `source` is a fiction URL |
| **Kokoro** (optional extra) | Real TTS; install the `tts` extra (see below) |

Install system tools yourself (Homebrew examples):

```bash
brew install ffmpeg
# Optional: https://ollama.com — then:
ollama pull llama3.2:3b
# Optional: install FictionReaper so `fictionreaper` is on PATH
```

## Install

### CLI tool (recommended for users)

```bash
# From the public GitHub repo
uv tool install git+https://github.com/Fluder-Paradyne/audioforge

audioforge --version
```

With real Kokoro TTS (optional extra):

```bash
uv tool install "audioforge[tts] @ git+https://github.com/Fluder-Paradyne/audioforge"
```

### Development checkout

```bash
git clone https://github.com/Fluder-Paradyne/audioforge.git
cd audioforge
uv sync --all-groups          # runtime + dev (ruff, mypy, pytest)
# Optional real TTS stack (platform-specific; Apple Silicon):
uv sync --all-groups --extra tts
```

## Quick start

### 1. From a local chapter folder

FictionReaper-style chapters (see `tests/fixtures/sample_book/`):

```text
my-book/
  0001-chapter-one.md
  0002-chapter-two.md
```

**Without Ollama / prep** (rules still run unless you skip; skip is fastest smoke path):

```bash
# Dev checkout
uv run audioforge build ./my-book --skip-prep

# Or installed tool
audioforge build ./my-book --skip-prep
```

**With Ollama prep** (Ollama running, model pulled):

```bash
audioforge build ./my-book --prep-model llama3.2:3b
```

**Without Kokoro installed** (silent fake TTS for plumbing checks only — not real audio):

```bash
AUDIOFORGE_ALLOW_FAKE_TTS=1 audioforge build ./my-book --skip-prep
```

### 2. From FictionReaper download, then build

```bash
fictionreaper download "https://www.royalroad.com/fiction/..." --output-dir ./downloaded-book
audioforge build ./downloaded-book --skip-prep
```

Or let AudioForge invoke FictionReaper when given a URL:

```bash
audioforge build "https://www.royalroad.com/fiction/..." --fictionreaper-bin fictionreaper
```

### 3. Staged workflow

```bash
audioforge prepare ./my-book --skip-prep
audioforge synthesize <job-id-or-path>
audioforge package <job-id-or-path>
audioforge status <job-id-or-path>
audioforge status <job-id-or-path> --json
```

Jobs live under `work/<job-id>/` by default (`AUDIOFORGE_WORK_DIR`).

### 4. Local API

```bash
audioforge serve                  # default 127.0.0.1:8765
audioforge serve --host 127.0.0.1 --port 8765
curl -s http://127.0.0.1:8765/health
# {"status":"ok"}
```

## CLI reference

| Command | Purpose |
|---------|---------|
| `audioforge --version` / `-V` | Print version and exit |
| `audioforge build <source>` | Full pipeline: ingest → prep → TTS → package |
| `audioforge prepare <source>` | Ingest + text prep only |
| `audioforge synthesize <job>` | TTS for an existing job |
| `audioforge package <job>` | Chapter audio → chaptered M4B |
| `audioforge status <job>` | Human summary; `--json` for full `job.json` |
| `audioforge serve` | Start local HTTP API (uvicorn) |

`<source>` is a local chapter directory or fiction URL.  
`<job>` is a job id under the work dir, a job folder path, or a path to `job.json`.

### `build` / `prepare` options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir PATH` | Optional output directory hint | — |
| `--voice STR` | TTS voice id | `AUDIOFORGE_DEFAULT_VOICE` / `af_heart` |
| `--prep-model STR` | Ollama prep model name | `AUDIOFORGE_DEFAULT_PREP_MODEL` / `llama3.2:3b` |
| `--skip-prep` | Skip LLM/rules text prep | off |
| `--resume` / `--no-resume` | Skip work already on disk | resume |
| `--force` / `--no-force` | Re-run stages even if artifacts exist | no-force |
| `--fictionreaper-bin STR` | Path to FictionReaper binary | `fictionreaper` |
| `--job-id STR` | Explicit job id under the work directory | auto |

### `serve` options

| Option | Description | Default |
|--------|-------------|---------|
| `--host STR` | Bind host | `AUDIOFORGE_HOST` / `127.0.0.1` |
| `--port INT` | Bind port | `AUDIOFORGE_PORT` / `8765` |

## HTTP API (brief)

`audioforge serve` loads `audioforge.api.app:create_app`.

| Method | Path | Status |
|--------|------|--------|
| `GET` | `/health` | Liveness + config flags (`ffmpeg_configured`, `ollama_base_url`) |
| `POST` | `/jobs` | Create job, start pipeline (202 + `job_id` / status) |
| `GET` | `/jobs/{job_id}` | Full job state from `job.json` |
| `GET` | `/jobs/{job_id}/artifacts` | Chapter audio + M4B paths when ready |

Disk `job.json` remains the source of truth for job state. See the [design spec](docs/superpowers/specs/2026-07-26-audioforge-design.md) for the full contract.

## Configuration (`AUDIOFORGE_*`)

Settings are loaded via Pydantic Settings (`AppSettings`). Environment variables use the `AUDIOFORGE_` prefix:

| Variable | Meaning | Default |
|----------|---------|---------|
| `AUDIOFORGE_WORK_DIR` | Job workspace root | `work` |
| `AUDIOFORGE_OLLAMA_BASE_URL` | Ollama HTTP base URL | `http://127.0.0.1:11434` |
| `AUDIOFORGE_FFMPEG_PATH` | FFmpeg executable | `ffmpeg` |
| `AUDIOFORGE_DEFAULT_VOICE` | Default Kokoro voice id | `af_heart` |
| `AUDIOFORGE_DEFAULT_PREP_MODEL` | Default Ollama model | `llama3.2:3b` |
| `AUDIOFORGE_HOST` | API bind host | `127.0.0.1` |
| `AUDIOFORGE_PORT` | API bind port | `8765` |
| `AUDIOFORGE_LOG_LEVEL` | Console log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`) | `INFO` |
| `AUDIOFORGE_LOG_FORMAT` | Console/job log format: `text` or `json` | `text` |

`AUDIOFORGE_LOG_LEVEL` controls the **console** handler. The package logger stays at DEBUG so handlers can filter independently (e.g. a verbose `job.log` while console stays INFO). Invalid values like `NOTSET` or typos are rejected.

For machine-readable logs (JSON Lines on stderr and in `job.log`):

```bash
AUDIOFORGE_LOG_FORMAT=json AUDIOFORGE_LOG_LEVEL=DEBUG audioforge build ./my-book --skip-prep
```

Additional (factory, not in `AppSettings`):

| Variable | Meaning |
|----------|---------|
| `AUDIOFORGE_ALLOW_FAKE_TTS=1` | If Kokoro is not installed, use a silent fake TTS backend instead of failing (dev/CI plumbing only) |

### Workspace layout (per job)

```text
work/<job-id>/
  source/           # FictionReaper-style chapter .md
  prepared/         # cleaned text per chapter
  audio/            # per-chapter audio
  out/              # final .m4b
  job.json          # JobState (source of truth)
  job.log           # Structured pipeline log for this job
```

## Development

```bash
uv sync --all-groups

uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=audioforge --cov-fail-under=100
```

- **uv** only — commit `uv.lock`
- **Python ≥ 3.12**, full typing, `mypy --strict`
- **Pydantic v2** at CLI / API / config / job boundaries
- **100%** line coverage on `audioforge` (fakes for Ollama, Kokoro, FFmpeg, FictionReaper)
- Branches: `issue/<n>-short-slug`; squash merge to `main`

Default unit tests do **not** call real Ollama, Kokoro, or the network.

## Ethics / legal

AudioForge is intended for **personal, offline** listening of material you are allowed to access.

- Respect **copyright** and the **Royal Road Terms of Service** (and any other source site’s rules).
- Be polite to remote sites when using FictionReaper; prefer local chapter folders once downloaded.
- Do not use this project to redistribute commercial or unauthorized audiobooks.
- Same posture as FictionReaper: personal archival / offline use, not bulk scraping or piracy tooling.

You are responsible for complying with applicable law and site policies.

## Design & plan

- [Design spec](docs/superpowers/specs/2026-07-26-audioforge-design.md) — architecture, models, CLI/API, ethics  
- [Implementation plan](docs/superpowers/plans/2026-07-26-audioforge-v1.md) — task breakdown and constraints  

## License

[MIT](LICENSE) © 2026 Fluder-Paradyne
