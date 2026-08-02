# Contributing to AudioTrove

## Setup

**Prerequisites:** Python 3.10+, ffmpeg, git.

Install ffmpeg for your platform:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows (via conda)
conda install -c conda-forge ffmpeg
```

Clone and install in editable mode with all dev dependencies:

```bash
git clone https://github.com/onepizzateam/AudioTrove.git
cd AudioTrove
pip install -e ".[dev]"
```

Verify the install:

```bash
audiotrove --version
pytest tests/ -x -q
```

---

## Running tests

Full suite with coverage:

```bash
pytest tests/ -v --cov=audiotrove --cov-report=term-missing
```

Single file:

```bash
pytest tests/test_vad.py -v
```

Tests use synthetic audio (sine waves generated via scipy) and mock the Silero VAD model
— no internet connection is required.

---

## Linting and type checking

```bash
# Check for lint errors and show auto-fix suggestions
ruff check audiotrove/ --show-fixes

# Format code in-place
ruff format audiotrove/

# Type checking
mypy audiotrove/
```

CI runs all three on every push. Fix any `ruff check` errors before opening a PR; `mypy`
warnings on third-party stubs are acceptable.

---

## Adding a new filter

1. Subclass `AudioFilter` from [`audiotrove/base.py`](audiotrove/base.py) and implement:
   - `name: str` — unique snake_case identifier used in checkpoint keys
   - `filter(doc: AudioDocument) -> bool` — return `True` to keep, `False` to reject;
     write a rejection reason to `doc.metadata` for observability
2. Add it to the appropriate pipeline in `audiotrove/pipelines/`.
3. Add unit tests covering at least one pass case and one fail case.
4. Add an integration test in `tests/test_tts_pipeline.py` if the filter affects the TTS
   pipeline.

---

## Adding a new transformer

1. Subclass `AudioTransformer` (1-to-1) or `AudioFanOutTransformer` (1-to-many) from
   [`audiotrove/base.py`](audiotrove/base.py) and implement `transform()`.
2. **Fan-out transformers:** `doc_id` values for child documents must be derived
   deterministically from the parent `doc_id` and segment metadata (e.g.
   `f"{parent_id}_seg{index}"`). This is required for checkpoint safety — re-running a
   pipeline must produce the same IDs so already-processed segments are skipped.
3. Optional heavy dependencies (model weights, external libs) must be imported lazily
   inside `transform()` or `__init__`, never at module import time.

---

## PR checklist

- [ ] Tests pass locally (`pytest tests/ -x -q`)
- [ ] `ruff check audiotrove/` passes with no errors
- [ ] New components have focused pass and fail unit tests
- [ ] New optional dependencies are added to the correct extra in `pyproject.toml`
- [ ] `ARCHITECTURE.md` updated if any component contracts or pipeline ordering changes
