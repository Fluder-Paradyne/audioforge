# AudioForge

Local **FictionReaper → audiobook** pipeline for Apple Silicon.

AudioForge turns FictionReaper chapter Markdown (or a Royal Road URL via FictionReaper) into single-voice audiobooks using local tooling: light text prep (Ollama with rule-based fallback), Kokoro TTS, and FFmpeg packaging to chaptered M4B.

> **Status:** early scaffolding (v0.1.0). The full pipeline is under construction.

## Requirements

- Python **≥ 3.12**
- [uv](https://docs.astral.sh/uv/) package manager
- (Later stages) FFmpeg, Ollama, Kokoro — not required for install/CI of this scaffold

## Install (development)

```bash
git clone https://github.com/Fluder-Paradyne/audioforge.git
cd audioforge
uv sync --all-groups
```

## CLI

```bash
uv run audioforge --version
uv run audioforge --help
```

Full commands (`build`, `prepare`, `synthesize`, `package`, `status`, `serve`) land in later milestones.

## Development checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=audioforge --cov-fail-under=100
```

## Design

See the approved design and plan:

- [Design spec](docs/superpowers/specs/2026-07-26-audioforge-design.md)
- [Implementation plan](docs/superpowers/plans/2026-07-26-audioforge-v1.md)

## License

[MIT](LICENSE) © 2026 Fluder-Paradyne
