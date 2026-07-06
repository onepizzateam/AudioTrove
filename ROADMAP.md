# AudioTrove — Actionable Engineering Roadmap
**Version 1.0 | July 2026**

---

## How to Read This Document

Every technical decision below is accompanied by its reasoning and the alternatives considered. Every phase has a hard gate — a set of conditions that must be true before Phase N+1 begins. The gate exists because the two most common failure modes in open-source infrastructure projects are (1) building Phase 3 features before Phase 0 actually works, and (2) accumulating architectural debt that costs three months to fix later.

The roadmap is sequential. Do not start Phase 1 until Phase 0 passes its gate. Do not start Phase 2 until Phase 1 passes its gate.

---

## Foundational Architecture Decisions

These decisions are made before any code is written. They cannot be changed cheaply later. This section is the ARCHITECTURE.md that the proposal says to write in Week 1.

### Decision 1: The AudioDocument Dataclass

Every piece of audio that flows through the pipeline is represented as a single canonical object.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AudioDocument:
    audio: np.ndarray        # Always: mono, float32, range [-1.0, 1.0]
    sample_rate: int         # Always: normalised at ingestion (default 16000 Hz)
    source_path: str         # Original path (local, s3://, hf://, etc.)
    duration_seconds: float  # Computed at ingestion, not re-derived
    doc_id: str              # Deterministic hash of source_path + byte_offset
    metadata: dict = field(default_factory=dict)  # Carries provenance through pipeline
```

**Why mono float32 always:** Every downstream stage (VAD, SNR, fingerprinting) is written once, not once-per-channel-count. Stereo content is downmixed at ingestion. This is not a loss for speech — virtually all speech model training uses mono. The alternative (passing multi-channel through and letting each stage handle it) produces subtle bugs that only appear on edge-case corpora, exactly the kind of data low-resource language teams will throw at this.

**Why 16kHz default:** Silero VAD, pyannote, and ECAPA-TDNN speaker embeddings all operate at 16kHz. Resampling at ingestion once is cheaper than each stage resampling independently. 44.1kHz input (music, podcasts) is resampled down; 8kHz telephone audio is resampled up. Both are handled by torchaudio's `Resample` transform.

**Why a deterministic `doc_id`:** Checkpointing, deduplication, and decontamination all need a stable identifier that doesn't depend on processing order. `doc_id = sha256(source_path)[:16]` is sufficient.

**Why `metadata` is a plain dict:** Stages append to it freely — VAD ratio, SNR score, speaker count, duplicate status. No schema is enforced because different pipeline configurations produce different metadata keys. The output writer serialises whatever is present.

---

### Decision 2: The Block Interface

A pipeline is a list of blocks. Every block is one of two types.

```python
from abc import ABC, abstractmethod

class AudioFilter(ABC):
    """Returns False to discard the document. Discarded documents are logged."""
    @abstractmethod
    def filter(self, doc: AudioDocument) -> bool:
        ...
    
    @property
    @abstractmethod  
    def name(self) -> str:
        ...

class AudioTransformer(ABC):
    """Returns a modified document. Cannot discard — use AudioFilter for that."""
    @abstractmethod
    def transform(self, doc: AudioDocument) -> AudioDocument:
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...
```

**Why two types, not one:** DataTrove uses a single block type where returning `None` means discard. That's clever but produces confusing code when a transformer accidentally returns `None` (a Python default). Explicit filter vs. transformer makes intent clear and makes unit testing simpler — a filter test checks `True`/`False`; a transformer test checks the output document.

**Why no intermediate state between blocks:** Each block receives a full `AudioDocument` and returns one. No shared mutable state. This means blocks are independently testable and reorderable. The alternative (blocks operating on a shared pipeline context object) produces tight coupling that makes the architecture fragile when users want non-standard orderings.

---

### Decision 3: The Executor Model

```python
class LocalExecutor:
    def __init__(self, pipeline: list, num_workers: int = 4, 
                 checkpoint_path: str | None = None):
        ...
    
    def run(self, reader: AudioReader, writer: AudioWriter) -> PipelineStats:
        ...
```

The executor handles parallelism and checkpointing. The pipeline list knows nothing about these concerns.

**Why multiprocessing over threading for local executor:** Python GIL. Audio processing (resampling, spectrogram computation) is CPU-bound, not I/O-bound. `multiprocessing.Pool` with `chunksize` tuning gives near-linear scaling up to physical core count.

**Why checkpointing from day one:** A pipeline run over 10,000 hours of audio takes hours. Without checkpointing, an OOM kill or SIGTERM at hour 5 means starting over. Checkpointing is not a Phase 3 feature. It is a Phase 0 requirement. Implementation: maintain a SQLite database at `checkpoint_path` with one row per processed `doc_id`. On resume, skip docs whose `doc_id` is already in the database.

**Why SQLite for checkpointing (not a flat file):** Concurrent writes from multiple workers. SQLite handles this; a flat file requires a lock. No extra dependency — `sqlite3` is in the Python standard library.

**Slurm executor (Phase 3):** Identical interface, different implementation. Users swap `LocalExecutor` for `SlurmExecutor` with no pipeline changes. This is the DataTrove executor pattern and it is the right one.

---

### Decision 4: Library Selection

#### Audio I/O: `torchaudio`

**Why torchaudio:** Handles WAV, MP3, FLAC, OGG natively. Resampling is built in via `torchaudio.transforms.Resample` (uses sinc interpolation, production-quality). It is already a dependency in every ML environment that will run AudioTrove. The tensor → numpy conversion is one `.numpy()` call.

**Why not librosa:** librosa requires scipy, which adds ~50MB to the install. librosa is also slower for batch I/O. Its main advantage (spectral analysis) is not needed here — we compute spectrograms ourselves for fingerprinting.

**Why not soundfile alone:** soundfile does not support MP3 (no MPEG decoder). A significant fraction of real-world audio data (podcasts, scraped web audio) is MP3. Making MP3 require a separate install path is a friction point the proposal cannot afford in early adoption.

**Fallback:** `soundfile` is kept as a fallback for non-PyTorch environments. The reader detects which is available.

#### Filesystem Abstraction: `fsspec`

**Why fsspec:** One interface for `file://`, `s3://`, `gs://`, and `hf://` paths. Users pass `--input s3://my-bucket/audio/` and it works without any code changes. This is exactly how DataTrove handles multi-environment deployment, and it is the right pattern.

**Optional extras:** `s3fs` for S3, `gcsfs` for GCS. Core package does not require them. `pip install audiotrove[s3]` pulls in s3fs. The reader raises a clear error if a path scheme is used without the corresponding extra installed.

#### VAD: Silero VAD

**Why Silero VAD:** MIT licensed. Pre-trained TorchScript model (~3MB). CPU inference time is ~80ms per 30-second clip. No HuggingFace token required. No heavy model download at first run. Widely battle-tested in production (Mozilla uses it). Configurable threshold.

**Why not pyannote VAD:** pyannote VAD is more accurate, particularly on overlapping speech, but requires a HuggingFace token and acceptance of gated model terms. This is an insurmountable friction point for a first install. It becomes an optional extra (`audiotrove[diarize]`) in Phase 1.

**Why not WebRTC VAD (webrtcvad):** C extension. Fails to install on some platforms without a compiler. Lower accuracy. No longer actively maintained. There is no reason to use it when Silero exists.

**Why not Whisper-based VAD:** Adds a 1GB+ model dependency for what is a preprocessing step. Whisper's VAD is a side effect of transcription; Silero VAD is purpose-built. Silero at 80ms vs. Whisper at 3-10 seconds per clip — the difference matters at scale.

#### CLI: `click`

**Why click:** Mature. Composable commands (`audiotrove curate`, `audiotrove inspect`, `audiotrove decon` as separate subcommands). Automatic `--help` generation. Used by DataTrove, Flask, and most ML tooling worth naming. The alternative `typer` is good but adds a fastapi-level dependency graph that is unnecessary here.

#### Progress and Logging: `rich`

**Why rich:** Long-running audio pipeline jobs need a progress bar that shows clips/sec, estimated time remaining, and per-stage filter rates. `tqdm` does not handle multi-worker progress well. `rich` does. It also renders the pipeline stats table in a format suitable for a screenshot in the HuggingFace blog post.

#### Deduplication Hash Storage: `sqlite3` (stdlib)

**Why SQLite:** No extra dependency. File-based (portable, no server). Fast enough for 10M+ hash lookups. Concurrent writer-safe with WAL mode enabled. The alternative (an in-memory dict) fails when the corpus is larger than RAM, which happens on real datasets above ~500k clips.

---

## Phase 0 — Foundation
**Duration: Weeks 1–8**
**Goal: A working VAD + SNR filter that a stranger can run on their own audio in under 10 minutes.**

---

### Week 1–2: Repository, CI/CD, and Core Architecture

#### Actions

**Repository setup:**
```
audiotrove/
├── audiotrove/
│   ├── __init__.py
│   ├── document.py       # AudioDocument dataclass
│   ├── base.py           # AudioFilter, AudioTransformer ABCs
│   ├── executor/
│   │   ├── __init__.py
│   │   └── local.py      # LocalExecutor
│   ├── io/
│   │   ├── __init__.py
│   │   ├── readers.py    # LocalAudioReader, HFDatasetReader
│   │   └── writers.py    # JSONLWriter (first), others in Phase 1
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── vad.py
│   │   └── snr.py
│   └── cli/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── fixtures/          # Short real audio files, committed to repo
│   │   ├── speech_clean.wav    # 5s clean speech, 16kHz mono
│   │   ├── speech_noisy.wav    # 5s speech + babble noise
│   │   ├── silence.wav         # 5s silence
│   │   ├── music.wav           # 5s music (no speech)
│   │   └── corrupt.wav         # Truncated/corrupt file
│   ├── test_document.py
│   ├── test_reader.py
│   ├── test_vad.py
│   └── test_snr.py
├── recipes/              # YAML pipeline configs (Phase 1)
├── examples/
├── docs/
├── ARCHITECTURE.md       # Written and locked this week
├── pyproject.toml
└── README.md
```

**CI/CD (GitHub Actions):**

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=audiotrove --cov-report=xml
      - run: ruff check audiotrove/
      - run: mypy audiotrove/ --ignore-missing-imports
```

**Why test on Linux + macOS from day one:** The ICP (low-resource language teams, TTS indie devs, academic labs) uses both. A macOS-only bug that appears on an AI4Bharat researcher's Linux cluster kills adoption at exactly the wrong moment.

**Why ruff + mypy from day one:** Technical debt in a data pipeline is extremely expensive. An `AudioDocument` field whose type is wrong produces silent data corruption, not a crash. Type annotations + mypy catches this at development time. Ruff catches style issues before they become review friction for external contributors.

**ARCHITECTURE.md contents:**
- The AudioDocument contract (mono, float32, 16kHz, why)
- The Filter vs. Transformer distinction (with examples)
- The executor model (why parallelism is in the executor, not the blocks)
- The checkpoint contract (what it guarantees on resume)
- What AudioTrove is not (model training, ASR, TTS synthesis)

**Why write ARCHITECTURE.md before any feature code:** One wrong decision in the executor interface — for example, making blocks stateful instead of stateless — will cost three months of refactoring when external contributors try to add a custom block and it breaks under multiprocessing. Lock the interface first. This is also the document that makes an early contributor PR review tractable — they can check "does this PR violate the architecture?" without understanding all the code.

#### Tests for Week 1–2

```python
# tests/test_document.py
def test_audio_document_creation():
    audio = np.zeros(16000, dtype=np.float32)
    doc = AudioDocument(audio=audio, sample_rate=16000, 
                        source_path="test.wav", duration_seconds=1.0,
                        doc_id="abc123")
    assert doc.audio.dtype == np.float32
    assert doc.metadata == {}

def test_audio_document_immutable_audio():
    # audio field should not be mutated by pipeline stages
    # (transformers should return new AudioDocument, not modify in place)
    ...
```

#### Gate Check: Week 2 Complete

- [ ] Repository is public on GitHub
- [ ] CI passes on Python 3.10/3.11/3.12, Linux and macOS
- [ ] `ARCHITECTURE.md` is written, reviewed, and committed
- [ ] `AudioDocument`, `AudioFilter`, `AudioTransformer` are defined and have 100% test coverage
- [ ] `LocalExecutor` skeleton exists (even if it only runs sequentially)
- [ ] First GitHub Discussion opened: "RFC: Pipeline Block Interface" — ask for feedback before building on top of it

---

### Week 3–4: Audio Reader + VAD Filter

#### Reader Implementation

```python
# audiotrove/io/readers.py

class LocalAudioReader:
    """Reads audio files from a local directory or glob pattern via fsspec."""
    
    def __init__(self, path_pattern: str, target_sample_rate: int = 16000,
                 max_duration_seconds: float | None = None,
                 min_duration_seconds: float = 0.5):
        self.path_pattern = path_pattern
        self.target_sr = target_sample_rate
        self.max_duration = max_duration_seconds
        self.min_duration = min_duration_seconds
    
    def __iter__(self) -> Iterator[AudioDocument]:
        import fsspec
        fs, path = fsspec.url_to_fs(self.path_pattern)
        for fpath in fs.glob(path):
            try:
                yield self._load(fpath, fs)
            except Exception as e:
                # Log and continue — corrupt files should not stop a pipeline
                logger.warning(f"Failed to load {fpath}: {e}")
                continue
    
    def _load(self, path: str, fs) -> AudioDocument:
        with fs.open(path, 'rb') as f:
            waveform, sr = torchaudio.load(f)
        # Downmix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to target
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        audio = waveform.squeeze(0).numpy().astype(np.float32)
        duration = len(audio) / self.target_sr
        # Skip clips outside duration bounds
        if duration < self.min_duration:
            return None  # executor skips None
        if self.max_duration and duration > self.max_duration:
            audio = audio[:int(self.max_duration * self.target_sr)]
            duration = self.max_duration
        return AudioDocument(
            audio=audio, sample_rate=self.target_sr,
            source_path=path, duration_seconds=duration,
            doc_id=_make_doc_id(path)
        )
```

**Why `min_duration_seconds=0.5` default:** Clips shorter than 500ms are nearly always VAD artifacts or silence padding. They are not useful for TTS or ASR training and cause numerical instability in SNR estimation. Filtering them at read time is cheaper than discovering them in downstream stages.

**Why catch-and-continue on corrupt files:** Real-world audio corpora (CommonVoice, scraped web audio) routinely contain truncated files, zero-byte files, and files with corrupt headers. A single corrupt file should not abort a 10,000-hour pipeline run. The warning is logged with the file path so the user can investigate.

#### VAD Filter Implementation

```python
# audiotrove/filters/vad.py

class SileroVADFilter(AudioFilter):
    """
    Discards clips where the speech ratio is below min_speech_ratio.
    Uses Silero VAD (MIT license, CPU-friendly, ~3MB model).
    """
    
    name = "silero_vad"
    
    def __init__(self, min_speech_ratio: float = 0.3, 
                 threshold: float = 0.5,
                 window_size_samples: int = 512):
        self.min_speech_ratio = min_speech_ratio
        self.threshold = threshold
        self.window_size = window_size_samples
        self._model = None  # Lazy-load: don't download at import time
    
    @property
    def model(self):
        if self._model is None:
            self._model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
        return self._model
    
    def filter(self, doc: AudioDocument) -> bool:
        audio_tensor = torch.from_numpy(doc.audio)
        speech_timestamps = get_speech_timestamps(
            audio_tensor, self.model,
            threshold=self.threshold,
            sampling_rate=doc.sample_rate,
            window_size_samples=self.window_size
        )
        if not speech_timestamps:
            doc.metadata['vad_speech_ratio'] = 0.0
            return False
        speech_samples = sum(t['end'] - t['start'] for t in speech_timestamps)
        ratio = speech_samples / len(doc.audio)
        doc.metadata['vad_speech_ratio'] = round(ratio, 4)
        doc.metadata['vad_speech_timestamps'] = speech_timestamps
        return ratio >= self.min_speech_ratio
```

**Why lazy-load the model:** `torch.hub.load` downloads the model on first call. If the model is loaded at import time, `import audiotrove` triggers a network request and a model download. That is unacceptable behaviour for a library. Lazy-load means the download happens on first `filter()` call, and only if this filter is actually in the pipeline.

**Why store `vad_speech_timestamps` in metadata:** The SNR filter (next stage) uses these timestamps to identify speech frames vs. noise frames. Computing VAD twice would be wasteful. Passing the timestamps through metadata avoids re-running Silero in the SNR stage.

**Why `min_speech_ratio=0.3` as default:** Empirically, clips with less than 30% speech content are dominated by music, background noise, or silence. The threshold is configurable — TTS pipelines typically want higher (0.7+); podcast ASR pipelines can tolerate lower (0.2). The default covers the most common case without aggressive filtering.

#### Tests for Week 3–4

```python
# tests/test_vad.py

def test_vad_keeps_clean_speech(speech_clean_fixture):
    """speech_clean.wav should pass VAD with ratio > 0.3"""
    vad = SileroVADFilter(min_speech_ratio=0.3)
    assert vad.filter(speech_clean_fixture) is True
    assert speech_clean_fixture.metadata['vad_speech_ratio'] > 0.3

def test_vad_discards_silence(silence_fixture):
    """silence.wav should be discarded"""
    vad = SileroVADFilter(min_speech_ratio=0.3)
    assert vad.filter(silence_fixture) is False
    assert silence_fixture.metadata['vad_speech_ratio'] == 0.0

def test_vad_discards_music(music_fixture):
    """music.wav (no speech) should be discarded"""
    vad = SileroVADFilter(min_speech_ratio=0.3)
    assert vad.filter(music_fixture) is False

def test_vad_metadata_populated(speech_clean_fixture):
    """VAD should populate metadata for downstream stages"""
    vad = SileroVADFilter()
    vad.filter(speech_clean_fixture)
    assert 'vad_speech_ratio' in speech_clean_fixture.metadata
    assert 'vad_speech_timestamps' in speech_clean_fixture.metadata

def test_vad_threshold_configurable(speech_noisy_fixture):
    """A clip that passes at 0.2 should fail at 0.8"""
    vad_loose = SileroVADFilter(min_speech_ratio=0.2)
    vad_strict = SileroVADFilter(min_speech_ratio=0.8)
    assert vad_loose.filter(speech_noisy_fixture) is True
    assert vad_strict.filter(speech_noisy_fixture) is False

def test_vad_model_lazy_loaded():
    """Model should not be downloaded at import time"""
    vad = SileroVADFilter()
    assert vad._model is None  # Not loaded until filter() is called
```

---

### Week 5–6: SNR Scoring + CLI Skeleton

#### SNR Filter Implementation

```python
# audiotrove/filters/snr.py

class SNRFilter(AudioFilter):
    """
    Estimates Signal-to-Noise Ratio using VAD-based power estimation.
    Requires VAD metadata to already be present (run SileroVADFilter first).
    Falls back to a simple energy-based estimate if VAD metadata is absent.
    """
    
    name = "snr_filter"
    
    def __init__(self, min_snr_db: float = 15.0,
                 use_mos_scoring: bool = False):
        self.min_snr_db = min_snr_db
        self.use_mos_scoring = use_mos_scoring
    
    def filter(self, doc: AudioDocument) -> bool:
        snr_db = self._compute_snr(doc)
        doc.metadata['snr_db'] = round(snr_db, 2)
        return snr_db >= self.min_snr_db
    
    def _compute_snr(self, doc: AudioDocument) -> float:
        timestamps = doc.metadata.get('vad_speech_timestamps')
        if not timestamps:
            # Fallback: treat top quartile energy as signal
            return self._energy_fallback_snr(doc.audio)
        
        audio = doc.audio
        sr = doc.sample_rate
        
        # Build speech mask
        speech_mask = np.zeros(len(audio), dtype=bool)
        for ts in timestamps:
            speech_mask[ts['start']:ts['end']] = True
        
        speech_frames = audio[speech_mask]
        noise_frames = audio[~speech_mask]
        
        if len(noise_frames) < sr * 0.1:  # Less than 100ms of noise
            # Not enough noise to estimate — assume clean, return high SNR
            doc.metadata['snr_note'] = 'insufficient_noise_floor'
            return 40.0
        
        # RMS power in dB
        signal_power = np.mean(speech_frames ** 2)
        noise_power = np.mean(noise_frames ** 2)
        
        if noise_power == 0:
            return 40.0  # Silence in noise region = perfect SNR
        
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return float(snr_db)
    
    def _energy_fallback_snr(self, audio: np.ndarray) -> float:
        """Simple SNR estimate when VAD timestamps not available."""
        frame_size = 512
        frames = audio[:len(audio) - len(audio) % frame_size].reshape(-1, frame_size)
        frame_energies = np.mean(frames ** 2, axis=1)
        threshold = np.percentile(frame_energies, 75)
        signal_power = np.mean(frame_energies[frame_energies >= threshold])
        noise_power = np.mean(frame_energies[frame_energies < threshold])
        return float(10 * np.log10(signal_power / (noise_power + 1e-10)))
```

**Why VAD-based SNR, not WADA-SNR:** WADA-SNR (waveform amplitude distribution analysis) is more accurate in controlled settings but requires scipy and is ~5x slower. VAD-based estimation is fast (pure numpy), has no extra dependencies, and is good enough to separate the 10dB clips from the 25dB clips — which is the actual job. WADA-SNR could be added as an optional accuracy upgrade later.

**Why `min_snr_db=15.0` as default:** Below ~10dB, speech intelligibility degrades significantly for ASR. Above 20dB is ideal but excludes a lot of useful real-world data. 15dB is the commonly cited threshold in the speech enhancement literature (NOIZEUS dataset, Hu & Loizou 2007) and what production pipelines like the ones described in IndexTTS 2.5 papers use as a starting point.

**Why keep the fallback:** Users who run `SNRFilter` without `SileroVADFilter` in their pipeline should not get a crash. They should get a warning and a less accurate but still useful estimate.

#### CLI Skeleton

```python
# audiotrove/cli/main.py
import click
from rich.console import Console

console = Console()

@click.group()
@click.version_option()
def cli():
    """AudioTrove: Open-source audio data curation pipeline."""
    pass

@cli.command()
@click.argument('input_path')
@click.argument('output_dir')
@click.option('--vad-threshold', default=0.3, show_default=True,
              help='Minimum speech ratio (0-1) to keep a clip.')
@click.option('--snr-min', default=15.0, show_default=True,
              help='Minimum SNR in dB to keep a clip.')
@click.option('--workers', default=4, show_default=True)
@click.option('--format', 'output_format', 
              type=click.Choice(['jsonl', 'webdataset', 'parquet']),
              default='jsonl', show_default=True)
@click.option('--recipe', default=None,
              help='Path to a YAML recipe file (overrides individual flags).')
@click.option('--checkpoint', default=None,
              help='Path to checkpoint database for resumable runs.')
def curate(input_path, output_dir, vad_threshold, snr_min, 
           workers, output_format, recipe, checkpoint):
    """Curate audio files from INPUT_PATH into OUTPUT_DIR."""
    ...

@cli.command()
@click.argument('input_path')
def inspect(input_path):
    """Show statistics for an audio directory without filtering."""
    ...
```

**Why `inspect` as a separate subcommand from day one:** The first thing a new user does is not run the full pipeline. They run something that shows them what their data looks like. If `inspect` doesn't exist, they'll do `audiotrove curate --dry-run` and discover it doesn't exist either. A working `inspect` that prints duration distribution, format counts, and estimated SNR histogram is a better first impression than a full pipeline run.

#### Tests for Week 5–6

```python
# tests/test_snr.py

def test_snr_clean_speech_passes(speech_clean_fixture):
    """Clean speech should exceed 15dB SNR threshold"""
    # Run VAD first to populate metadata
    vad = SileroVADFilter()
    vad.filter(speech_clean_fixture)
    snr = SNRFilter(min_snr_db=15.0)
    assert snr.filter(speech_clean_fixture) is True
    assert speech_clean_fixture.metadata['snr_db'] > 15.0

def test_snr_noisy_speech_filtered(speech_noisy_fixture):
    """Heavily noisy speech should fail high SNR threshold"""
    vad = SileroVADFilter()
    vad.filter(speech_noisy_fixture)
    snr = SNRFilter(min_snr_db=25.0)
    assert snr.filter(speech_noisy_fixture) is False

def test_snr_populates_metadata(speech_clean_fixture):
    vad = SileroVADFilter()
    vad.filter(speech_clean_fixture)
    snr = SNRFilter()
    snr.filter(speech_clean_fixture)
    assert 'snr_db' in speech_clean_fixture.metadata

def test_snr_fallback_without_vad(speech_clean_fixture):
    """SNR filter should work even without prior VAD metadata"""
    snr = SNRFilter(min_snr_db=10.0)
    result = snr.filter(speech_clean_fixture)
    assert isinstance(result, bool)  # Should not raise

# tests/test_cli.py
from click.testing import CliRunner
from audiotrove.cli.main import cli

def test_curate_command_runs(tmp_path, fixtures_dir):
    runner = CliRunner()
    result = runner.invoke(cli, [
        'curate', str(fixtures_dir), str(tmp_path),
        '--vad-threshold', '0.3', '--snr-min', '10.0', '--workers', '1'
    ])
    assert result.exit_code == 0
    assert (tmp_path / 'manifest.jsonl').exists()

def test_inspect_command_runs(fixtures_dir):
    runner = CliRunner()
    result = runner.invoke(cli, ['inspect', str(fixtures_dir)])
    assert result.exit_code == 0
    assert 'duration' in result.output.lower()
```

---

### Week 7–8: Documentation, Performance Baseline, PyPI Prep

#### README Requirements

The README at Phase 0 gate must contain:

1. **One-paragraph problem statement** — "Every lab training a speech model reimplements this from scratch."
2. **Installation** — `pip install audiotrove` with expected output.
3. **30-second demo** — CLI command that runs on the included example audio.
4. **Before/after** — show a JSONL manifest entry with `snr_db`, `vad_speech_ratio` populated.
5. **Scope fence** — explicit "AudioTrove is NOT a model training framework" section.
6. **Contributing guide** — how to add a new filter block in <20 lines.

#### Performance Baseline (Establish Now, Track Forever)

Run a benchmark on the fixtures and log the results in `benchmarks/phase0_baseline.json`. This becomes the regression test for performance.

```
Target throughputs (single CPU core):
- Reader (WAV, local): > 500 clips/sec
- SileroVADFilter: > 10 clips/sec (the bottleneck; model inference)
- SNRFilter: > 200 clips/sec
- End-to-end (VAD + SNR): > 8 clips/sec
```

At 8 clips/sec with 4 workers, a 10,000-clip corpus completes in ~20 minutes. That is acceptable for Phase 0. Phase 1 dedup will be the next bottleneck to profile.

#### pyproject.toml

```toml
[project]
name = "audiotrove"
version = "0.0.1"
description = "Composable open-source audio data curation pipeline"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "numpy>=1.24.0",
    "fsspec>=2023.1.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
s3 = ["s3fs>=2023.1.0"]
gcs = ["gcsfs>=2023.1.0"]
diarize = ["pyannote.audio>=3.1.0"]
embed-dedup = ["faiss-cpu>=1.7.4", "speechbrain>=1.0.0"]
dev = ["pytest", "pytest-cov", "ruff", "mypy", "click[testing]"]

[project.scripts]
audiotrove = "audiotrove.cli.main:cli"
```

**Why torch is a core dependency, not optional:** Silero VAD requires PyTorch. Torchaudio requires PyTorch. The entire audio processing stack assumes PyTorch is present. Making it optional would mean the library does nothing without it. Be honest about this dependency rather than pretending it is optional.

**Why `torch` CPU-only is acceptable:** Silero VAD and SNR scoring do not benefit meaningfully from GPU. Near-duplicate fingerprinting (Phase 1) is numpy. Only embedding-based dedup (optional extra) benefits from GPU. Default install should not require CUDA.

#### Phase 0 Gate Checklist

Before Phase 1 begins, every item must be true:

- [ ] `pip install audiotrove` succeeds on fresh Python 3.10/3.11/3.12 environments, Linux and macOS
- [ ] `audiotrove curate ./audio_dir ./output --vad-threshold 0.3 --snr-min 15` runs end-to-end
- [ ] Total runtime for 100 WAV files (5s each) < 3 minutes on a single laptop core
- [ ] Test coverage ≥ 85% for all Phase 0 code
- [ ] All CI checks pass (pytest, ruff, mypy)
- [ ] README includes working 30-second demo
- [ ] `ARCHITECTURE.md` is written and committed
- [ ] GitHub Discussions "RFC: Pipeline Block Interface" thread has been open for at least 1 week
- [ ] At least **one person outside the project** has run it on their own audio and reported the result (success or failure both count)

The last item is the most important. If zero external people have run it, Phase 0 is not done.

---

## Phase 1 — Core Pipeline
**Duration: Months 2–5**
**Goal: Full pipeline through dedup and diarization. PyPI v0.1 published. 500 GitHub stars. First external contributor PR.**

---

### Month 2: Near-Duplicate Detection

#### Technical Design

Near-duplicate detection finds recordings that are acoustically near-identical — same source audio re-encoded at different bitrates, slightly trimmed versions of the same clip, or near-identical recordings of the same text by the same speaker. These inflate training data size without adding information and can cause eval set contamination.

**Stage 1: Perceptual Hash (Fast, Always Run)**

```python
# audiotrove/filters/dedup.py

class PerceptualHashDeduplicator:
    """
    Computes a perceptual hash over the mel spectrogram.
    Identical or near-identical recordings produce similar hashes.
    
    Algorithm:
    1. Compute 64-band mel spectrogram (no windowing — full clip)
    2. Resize to 32x32 pixels (consistent dimensionality)
    3. Compute DCT, keep top-left 8x8 coefficients (dHash variant)
    4. Binarise: 1 if coefficient > mean, 0 otherwise
    5. Result: 64-bit integer hash
    
    Two clips with Hamming distance <= threshold are considered duplicates.
    """
    
    name = "perceptual_dedup"
    
    def __init__(self, hamming_threshold: int = 10,
                 db_path: str = "audiotrove_dedup.db"):
        self.threshold = hamming_threshold
        self.db_path = db_path
        self._db = None
    
    def _compute_hash(self, doc: AudioDocument) -> int:
        import torchaudio.transforms as T
        audio_tensor = torch.from_numpy(doc.audio).unsqueeze(0)
        mel = T.MelSpectrogram(
            sample_rate=doc.sample_rate, n_fft=1024,
            hop_length=512, n_mels=64
        )(audio_tensor)
        # Log scale, normalise
        mel_db = T.AmplitudeToDB()(mel).squeeze(0).numpy()
        # Resize to 32x32
        from PIL import Image
        img = Image.fromarray(mel_db).resize((32, 32), Image.LANCZOS)
        arr = np.array(img, dtype=float)
        # DCT-based hash (dHash variant)
        mean_val = arr.mean()
        bits = (arr > mean_val).flatten()
        hash_int = int(''.join(bits[:64].astype(str)), 2)
        return hash_int
    
    def _hamming_distance(self, a: int, b: int) -> int:
        return bin(a ^ b).count('1')
    
    def is_duplicate(self, doc: AudioDocument) -> bool:
        h = self._compute_hash(doc)
        doc.metadata['perceptual_hash'] = h
        # Look up in DB: find any stored hash within threshold
        existing = self._get_similar_hash(h)
        if existing is not None:
            doc.metadata['duplicate_of'] = existing
            return True
        self._store_hash(h, doc.doc_id)
        return False
```

**Why mel spectrogram, not raw waveform hash:** Raw waveform hashes are format-sensitive — the same audio re-encoded at a different bitrate produces a completely different waveform hash. Mel spectrograms are perceptual: they capture what the audio sounds like, not exactly what the bytes are. This is what makes them suitable for finding near-duplicates across different encodings of the same source.

**Why 32x32 resize:** Consistent dimensionality regardless of clip length. A 2-second clip and a 10-second clip of the same audio produce the same hash if they represent the same content. This is the same principle as image perceptual hashing (pHash, dHash) — length-invariant fingerprint.

**Why Hamming distance ≤ 10 as default:** Empirically, clips sharing the same source audio but with minor differences (slight trim, volume normalisation) produce hashes with Hamming distance < 5. Distinct clips from the same speaker (different sentences) produce distances > 15. 10 is a safe middle ground. The threshold is configurable and should be tuned by the user based on their corpus — the `audiotrove inspect` command will show the distribution of pairwise distances.

**Why SQLite for hash storage (not in-memory):** A corpus of 1 million clips with 64-bit hashes requires 8MB of storage — trivially small for SQLite. But that 1M-hash lookup from memory requires 8MB of RAM held for the duration of the pipeline run and does not survive a crash. SQLite with WAL mode persists across restarts and is the correct choice.

**Stage 2: Embedding-Based Dedup (Optional Extra, Phase 1)**

```
pip install audiotrove[embed-dedup]
```

Uses ECAPA-TDNN speaker embeddings + approximate nearest-neighbour search (faiss). Finds semantic duplicates: different encodings of the same underlying content. More accurate but ~20x slower than perceptual hash. Recommended for corpora where near-duplicate content at the semantic level is likely (scraped web audio, synthetic TTS data).

This is built in Phase 1 but behind the optional extra flag. It should not block v0.1 publication.

#### Tests for Month 2

```python
def test_dedup_identical_files_detected(speech_clean_fixture):
    """Two identical audio clips should be flagged as duplicates"""
    dedup = PerceptualHashDeduplicator(db_path=':memory:')
    result1 = dedup.is_duplicate(speech_clean_fixture)
    result2 = dedup.is_duplicate(speech_clean_fixture)  # Same clip again
    assert result1 is False   # First occurrence: not a duplicate
    assert result2 is True    # Second occurrence: duplicate

def test_dedup_different_files_not_flagged(speech_clean_fixture, speech_noisy_fixture):
    """Different audio clips should not be flagged as duplicates"""
    dedup = PerceptualHashDeduplicator(db_path=':memory:')
    dedup.is_duplicate(speech_clean_fixture)
    result = dedup.is_duplicate(speech_noisy_fixture)
    assert result is False

def test_dedup_hash_in_metadata(speech_clean_fixture):
    dedup = PerceptualHashDeduplicator(db_path=':memory:')
    dedup.is_duplicate(speech_clean_fixture)
    assert 'perceptual_hash' in speech_clean_fixture.metadata

def test_dedup_survives_format_re_encoding():
    """Same audio at different bitrates should be detected as duplicate"""
    # Fixture: same 5s speech clip encoded as 128kbps MP3 and 64kbps MP3
    # Both should produce hashes with Hamming distance < 10
    ...
```

---

### Month 3: Speaker Diarisation Hook

#### Technical Design

```python
# audiotrove/filters/diarize.py

class SpeakerDiarisationFilter(AudioFilter):
    """
    Optional filter using pyannote.audio diarisation.
    Requires: pip install audiotrove[diarize] and a HuggingFace token.
    
    Use cases:
    - Filter by per-speaker clip count (prevent speaker ID leakage into eval)
    - Extract single-speaker segments from multi-speaker recordings
    - Build speaker verification datasets
    """
    
    name = "speaker_diarisation"
    
    def __init__(self, hf_token: str,
                 max_speakers_per_clip: int | None = None,
                 min_speaker_duration_seconds: float = 1.0):
        self.hf_token = hf_token
        self.max_speakers = max_speakers_per_clip
        self.min_speaker_duration = min_speaker_duration_seconds
        self._pipeline = None
    
    @property  
    def pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
        return self._pipeline
```

**Why pyannote.audio 3.1:** Best open-source diarisation accuracy as of 2026. DER (Diarisation Error Rate) of ~12% on AMI corpus, competitive with commercial systems. The code is MIT licensed. The limitation is that the pretrained models are gated on HuggingFace and require a token — this is documented prominently, not hidden.

**Why an optional extra, not core:** pyannote.audio pulls in a significant dependency graph (speechbrain, asteroid, etc.). Making it optional means the base install remains lightweight. TTS fine-tuners (22% of ICP) do not need diarisation. Low-resource language teams (28% of ICP) may or may not need it. Only users explicitly building diarisation training data or doing speaker-ID-aware filtering need this module.

**Why document the HuggingFace token requirement prominently:** The worst possible user experience is: install library, write pipeline config, run pipeline, fail with an opaque error 20 minutes in because a model download requires authentication. The diarisation module should check for the token at instantiation time and raise a clear error with instructions if it is absent.

---

### Month 4: Output Formats and HuggingFace Hub Integration

#### JSONL Writer (Phase 0, formalized here)

```json
{
  "doc_id": "a3f9c2d1",
  "source_path": "s3://my-bucket/audio/clip_001.wav",
  "duration_seconds": 4.82,
  "sample_rate": 16000,
  "output_path": "clean_corpus/clip_001.wav",
  "metadata": {
    "vad_speech_ratio": 0.71,
    "snr_db": 22.4,
    "perceptual_hash": 9223372036854775807,
    "speaker_count": 1
  },
  "pipeline_version": "0.1.0",
  "processed_at": "2026-07-02T14:22:31Z"
}
```

**Why include `pipeline_version`:** When someone cites "AudioTrove-curated Common Voice" in a paper, reviewers need to know exactly which version was used. Provenance metadata is non-negotiable for academic credibility.

#### WebDataset Writer

```python
class WebDatasetWriter(AudioWriter):
    """
    Writes output as WebDataset tar shards.
    Compatible with PyTorch DataLoader for large-scale training.
    Shard size: default 1GB per shard.
    """
```

**Why WebDataset:** The format used by LLaVA, Stable Diffusion, and virtually every serious large-scale multimodal training run. Users who want to plug AudioTrove output directly into a training loop without conversion steps need this format. It also pairs naturally with HuggingFace Hub's streaming datasets.

#### HuggingFace Datasets Parquet Writer

For publishing cleaned datasets to HF Hub. The cleaned corpus becomes a citable, linkable artifact — not just a file on someone's local disk.

#### Month 4 Gate: v0.1 PyPI Publication

```
pip install audiotrove==0.1.0
```

Publication checklist:
- [ ] All three output formats (JSONL, WebDataset, Parquet) tested
- [ ] `audiotrove curate` runs end-to-end with all Phase 0+1 stages
- [ ] Changelog written
- [ ] GitHub release tagged
- [ ] HuggingFace blog post drafted (publish in Month 5)

---

### Month 5: Launch

#### The Launch Asset: HuggingFace Blog Post

Title: **"There is no FineWeb for audio — so we built it."**

Structure:
1. Show the gap with Figure 3 from the proposal (text has 4–8 tools per stage; audio has 0–1)
2. Show the Sommelier quote + OLMoASR quote as primary evidence
3. Show the AudioTrove pipeline diagram
4. **The proof that matters:** Run a small TTS model (CosyVoice or StyleTTS2 fine-tuning) on (a) raw CommonVoice EN and (b) AudioTrove-curated CommonVoice EN. Show MOS improvement. Numbers > words.
5. One-command install and demo
6. Call to action: GitHub Discussions, Contributing guide

**Why MOS comparison:** This is the proof point that converts users. "Here is a tool" gets skimmed. "Here is evidence that using this tool produces a measurably better model" makes people install it. FineWeb's blog post worked because it showed C4 vs. FineWeb vs. RefinedWeb training curves. AudioTrove needs the equivalent.

#### Twitter/X Launch Thread

12 tweets:
1. The gap chart (Figure 3)
2. The Sommelier quote
3. Pipeline diagram + one-liner
4. `pip install audiotrove` + demo GIF
5. MOS improvement number (the hook)
6. Who it is for (5 segments)
7. The DataTrove analogy
8. Link to HF blog post

Target: reply to 3–5 prominent speech ML researchers with the thread. Not cold DMs — replies to their existing tweets about audio training data problems.

#### Phase 1 Gate Checklist

- [ ] `pip install audiotrove` installs all Phase 0+1 features cleanly
- [ ] All seven pipeline stages documented with examples
- [ ] Near-duplicate detection tested on a corpus with known duplicates
- [ ] HuggingFace blog post published
- [ ] 500 GitHub stars
- [ ] First external contributor PR merged (even a small one: docs fix, new test, minor feature)
- [ ] Test coverage ≥ 85% across all Phase 1 code

---

## Phase 2 — Decontamination and Community
**Duration: Months 6–9**
**Goal: Eval-set decontamination module shipped. Benchmark published. 5 papers citing. 15+ contributors.**

---

### Month 6–7: Eval-Set Decontamination

#### Technical Design

Eval-set decontamination prevents training data from containing clips that acoustically overlap with held-out test sets. Without it, benchmark scores are inflated in ways that are impossible to detect from the paper alone.

```python
# audiotrove/filters/decon.py

class EvalSetDecontaminator(AudioFilter):
    """
    Removes training clips that acoustically overlap with a held-out eval set.
    Uses the same perceptual fingerprinting as near-duplicate detection,
    but compares against a fixed reference corpus rather than within the training set.
    
    Reference eval sets supported out of the box:
    - librispeech-test-clean
    - librispeech-test-other
    - seedtts-eval
    - vctk-test
    - custom (pass your own directory)
    """
    
    name = "eval_decontamination"
    
    def __init__(self, eval_set: str | list[str],
                 hamming_threshold: int = 8,
                 cache_dir: str = "~/.audiotrove/decon_cache"):
        self.eval_sets = [eval_set] if isinstance(eval_set, str) else eval_set
        self.threshold = hamming_threshold
        self.cache_dir = cache_dir
        self._eval_hashes: set[int] | None = None
    
    def _load_eval_hashes(self):
        """Compute or load cached hashes for all eval set clips."""
        cache_path = Path(self.cache_dir) / self._cache_key()
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self._eval_hashes = pickle.load(f)
            return
        # Compute hashes for eval set
        self._eval_hashes = set()
        for eval_set_path in self.eval_sets:
            reader = LocalAudioReader(eval_set_path)
            hasher = PerceptualHashDeduplicator(db_path=':memory:')
            for doc in reader:
                h = hasher._compute_hash(doc)
                self._eval_hashes.add(h)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self._eval_hashes, f)
    
    def filter(self, doc: AudioDocument) -> bool:
        if self._eval_hashes is None:
            self._load_eval_hashes()
        h = doc.metadata.get('perceptual_hash')
        if h is None:
            # Compute if not already in metadata
            hasher = PerceptualHashDeduplicator(db_path=':memory:')
            h = hasher._compute_hash(doc)
        # Check against eval hashes
        for eval_hash in self._eval_hashes:
            if bin(h ^ eval_hash).count('1') <= self.threshold:
                doc.metadata['decon_match'] = True
                return False
        return True
```

**Why reuse perceptual hash from dedup:** The same fingerprinting technique that finds within-corpus duplicates is appropriate for cross-corpus decontamination. There is no need for a different algorithm. This also means a user who is already running `PerceptualHashDeduplicator` can cache the `perceptual_hash` in metadata and the `EvalSetDecontaminator` reuses it — no double computation.

**Why cache eval set hashes:** LibriSpeech test-clean has 2,620 clips. Computing hashes for all of them takes ~30 seconds. Running decontamination on a million-clip training corpus involves checking each clip against these 2,620 hashes. Caching the eval set hashes means this 30-second cost is paid once per eval set, not once per pipeline run.

**Why `hamming_threshold=8` (tighter than dedup's 10):** Decontamination false negatives (failing to remove a test clip from training) are more damaging than dedup false negatives. A slightly tighter threshold is appropriate here at the cost of rare false positives.

#### Benchmark Publication (Month 7)

Train two identical small TTS models:
- Model A: Fine-tuned on raw CommonVoice EN (100 hours)
- Model B: Fine-tuned on AudioTrove-curated CommonVoice EN (same 100 hours, post-filter)

Measure: MOS (Mean Opinion Score) on held-out test set using UTMOSv2 (automated MOS predictor). Publish as a HuggingFace dataset card and a blog post.

**Why UTMOSv2, not human MOS:** Human MOS at meaningful scale (100+ clips, 5+ raters) takes weeks and money. UTMOSv2 (Saeki et al., 2024) has >0.9 Pearson correlation with human MOS on public benchmarks. It is good enough to demonstrate a real improvement and fast enough to run in CI.

**The number this benchmark needs to show:** AudioTrove-curated training data produces a model that scores ≥ 0.3 UTMOSv2 points higher than raw training data. That is the unit of proof that makes people install the library.

---

### Month 8–9: Community Infrastructure

#### Contributor Onboarding

`CONTRIBUTING.md` must answer:
1. How to add a new `AudioFilter` — with a complete worked example (LanguageIDFilter)
2. How to add a new reader (e.g., a podcast RSS reader)
3. How to add a new output format
4. Test requirements (every new filter needs at least 4 tests)
5. What ARCHITECTURE.md says you cannot do (no stateful blocks, no disk I/O in filters)

#### Direct Outreach (Month 8)

Contact these organisations directly with a personalised message:
- **AI4Bharat** (Indic language ASR/TTS) — offer to add a IndicVAD calibration recipe
- **Masakhane** (African language NLP) — offer to co-author a short blog post showing AudioTrove on their pipeline
- **Sunbird AI** (East African language AI) — offer to add a crowdsourced-audio denoising recipe
- **Mozilla Common Voice** — open a PR adding an AudioTrove recipe to their contributor documentation

**Why these four, not broader outreach:** These organisations are the "28% low-resource language teams" from the ICP analysis. They are the most underserved, the most likely to adopt, and the most credible endorsers. A quote from an AI4Bharat researcher in the README converts more users than 1,000 Twitter impressions. Prioritise depth over breadth.

#### Interspeech / ICASSP Submission (Month 9)

Submit a 4-page system description paper to one of:
- Interspeech 2027 (deadline typically March)
- ICASSP 2027 (deadline September 2026)

Paper content:
- The tooling gap (evidence from Sommelier, OLMoASR, MiMo-Audio)
- AudioTrove's pipeline design
- The CommonVoice benchmark results
- Usage statistics (downloads, known users)

**Why submit to a workshop rather than main track:** A system description paper at Interspeech or ICASSP workshop has a realistic acceptance rate for a tool paper. The value is not the paper itself — it is that every subsequent speech paper that uses AudioTrove writes "we used AudioTrove (Author, 2027)" and cites it. That citation loop is how tools become canonical references.

#### Phase 2 Gate Checklist

- [ ] Decontamination module ships and is tested against LibriSpeech test-clean
- [ ] CommonVoice benchmark published (UTMOSv2 results visible in README)
- [ ] HuggingFace Hub integration: direct loading via `audiotrove.from_hub()`
- [ ] 5+ papers citing AudioTrove (Google Scholar alert set up)
- [ ] 15+ contributors (GitHub contributor graph)
- [ ] ICASSP or Interspeech submission sent
- [ ] AI4Bharat or Masakhane has run it on real data and said so publicly

---

## Phase 3 — Ecosystem
**Duration: Months 10–12**
**Goal: v1.0. Plugin API. Slurm + cloud executor. Pre-built recipes. Adoption by at least one named research lab.**

---

### Month 10: Plugin API

The plugin API allows external packages to register custom blocks without being part of the AudioTrove core.

```python
# External package: audiotrove-indic
# setup.cfg
[options.entry_points]
audiotrove.filters =
    indic_vad = audiotrove_indic.filters:IndicVADFilter

# Usage — no import needed, just install the package
pip install audiotrove audiotrove-indic

audiotrove curate ./audio --filter indic_vad --vad-threshold 0.4
```

**Why entry points for plugin discovery:** The Python entry points mechanism (`importlib.metadata`) is the standard pattern for extensible libraries (pytest plugins, Flask extensions, Click commands). It allows third-party packages to register AudioTrove blocks without AudioTrove knowing about them at import time. This is the correct architecture for a tool that wants an ecosystem of specialised filters (language-specific VAD, domain-specific SNR estimators, etc.).

**Why Phase 3, not Phase 1:** The plugin API requires the block interface to be stable. The block interface cannot be called stable until it has been used by real external contributors and the rough edges have been filed off. Phase 1 and 2 provide that iteration. Locking the plugin API before the interface is proven would mean breaking external packages when the interface changes.

---

### Month 11: Slurm and Cloud Executor

```python
class SlurmExecutor:
    """
    Runs an AudioTrove pipeline as a Slurm job array.
    Each job processes a shard of the input corpus.
    Results are merged after all jobs complete.
    
    Usage:
        executor = SlurmExecutor(
            pipeline=[SileroVADFilter(), SNRFilter()],
            num_jobs=100,
            cpus_per_job=8,
            mem_gb=32,
            time_limit="4:00:00"
        )
        executor.run(reader, writer)
    """
```

**Why identical interface to LocalExecutor:** A user who has been running on a laptop should be able to switch to Slurm by changing one line of code. The `LocalExecutor` is for development and small corpora; the `SlurmExecutor` is for production. They are not different tools — they are different backends for the same pipeline.

**What Slurm adds:** Job array submission via `sbatch`, automatic sharding of the input corpus, inter-job deduplication (the SQLite hash database is on shared storage), and result merging after all jobs complete. The checkpoint database handles partial failures — if job 37 of 100 fails and is requeued, it resumes from where it left off.

---

### Month 12: Pre-Built Recipes and v1.0

#### Recipe Format

```yaml
# recipes/tts-single-speaker.yaml
name: "TTS Single Speaker"
description: "Curate single-speaker TTS training data from raw recordings"
version: "1.0.0"
pipeline:
  - type: SileroVADFilter
    min_speech_ratio: 0.7       # Higher than default — TTS wants dense speech
    threshold: 0.5
  - type: SNRFilter
    min_snr_db: 20.0            # Higher than default — TTS is quality-sensitive
  - type: PerceptualHashDeduplicator
    hamming_threshold: 8
  - type: EvalSetDecontaminator
    eval_set: "seedtts-eval"
output_format: webdataset
target_sample_rate: 22050       # TTS models typically use 22050 Hz
```

```yaml
# recipes/asr-multilingual.yaml
name: "ASR Multilingual"
description: "Curate multilingual ASR training data"
pipeline:
  - type: SileroVADFilter
    min_speech_ratio: 0.3       # More permissive — ASR tolerates background
  - type: SNRFilter
    min_snr_db: 10.0
  - type: PerceptualHashDeduplicator
    hamming_threshold: 10
  - type: EvalSetDecontaminator
    eval_set: ["librispeech-test-clean", "librispeech-test-other"]
output_format: parquet
target_sample_rate: 16000
```

**Why recipes in YAML, not Python config:** YAML recipes are shareable and citable. A paper can write "We used the AudioTrove `asr-multilingual` v1.0 recipe" and reviewers can find and run it. A Python config requires reading code. A YAML recipe is a document.

#### v1.0 Semantic Guarantees

The v1.0 release commits to:
- The `AudioDocument` interface is stable (no breaking changes without a major version bump)
- The `AudioFilter` and `AudioTransformer` ABCs are stable
- All built-in block names are stable
- The YAML recipe format is stable
- All pre-built recipes produce deterministic output for the same input and version

**Why make these guarantees at v1.0 and not before:** Before v1.0, breaking changes are acceptable and expected. After v1.0, a lab that has integrated AudioTrove into their pipeline should not have that pipeline break on `pip install --upgrade audiotrove`. The stability guarantee is what converts trials into permanent integrations.

#### Phase 3 Gate = v1.0 Release Checklist

- [ ] Plugin API documented with a complete worked example
- [ ] Slurm executor tested on at least one real Slurm cluster
- [ ] Three pre-built recipes (TTS, ASR multilingual, SLM) published and tested
- [ ] v1.0 interface stability guarantees documented
- [ ] Changelog covers all breaking changes from v0.1 to v1.0
- [ ] 2,000+ GitHub stars
- [ ] At least one named research lab (AI4Bharat, Masakhane, academic speech lab) publicly using AudioTrove in a project or paper
- [ ] PyPI downloads > 5,000/month

---

## Risk Mitigations — Operational Detail

### Risk 1: A Large Lab Ships First

**Trigger:** HuggingFace, AI2, or NVIDIA open-sources an internal audio curation pipeline.

**Response:**
- If it happens before Phase 0 ships: evaluate whether their tool is truly general-purpose or specific to their corpus. If general-purpose, contribute to it instead of competing.
- If it happens after Phase 0: AudioTrove already exists, has users, and should position as complementary. Open a PR to their tool that adds AudioTrove as a recommended preprocessing step.
- Mitigation: **Ship Phase 0 fast.** Every week of delay increases this risk.

### Risk 2: Acoustic Fingerprinting False Positive Rate Too High

**Trigger:** Users report that distinct audio clips are being incorrectly flagged as duplicates.

**Response:**
- The `hamming_threshold` is configurable from day one. Users can loosen it immediately.
- Add `--dedup-dry-run` mode that reports would-be duplicates without removing them, so users can inspect before committing.
- If the perceptual hash is fundamentally insufficient, the optional embedding-based dedup exists as a fallback.

### Risk 3: Low Adoption Despite Good Technical Quality

**Trigger:** < 200 GitHub stars after Phase 1 launch, no external contributors.

**Response:**
- Do not interpret this as "the problem is not real." Re-examine the distribution, not the product.
- The most likely cause is that the blog post did not reach the right audience. Try: a direct email to 20 researchers who published relevant papers in the last year, offering to run AudioTrove on their data and share the results.
- A secondary cause may be that the setup friction is too high. Run a usability test with one person from the target ICP over video call and watch them install it.

---

## Key Dependencies and Version Pins

| Package | Min Version | Why Pinned |
|---|---|---|
| torch | ≥2.0.0 | TorchScript Silero VAD requires 2.0+ |
| torchaudio | ≥2.0.0 | Matched to torch version |
| numpy | ≥1.24.0 | dtype handling improvements |
| fsspec | ≥2023.1.0 | HF path scheme support |
| click | ≥8.1.0 | Stable group/command API |
| pyannote.audio | ≥3.1.0 | [diarize] extra; 3.1 has best DER |
| faiss-cpu | ≥1.7.4 | [embed-dedup] extra |

---

## Success Metrics — Monthly Tracking

| Metric | Month 2 | Month 4 (v0.1) | Month 6 | Month 9 | Month 12 (v1.0) |
|---|---|---|---|---|---|
| GitHub Stars | 50 | 500 | 800 | 1,500 | 2,000+ |
| PyPI Downloads/month | — | 200 | 1,000 | 3,000 | 5,000+ |
| External Contributors | 0 | 1 | 5 | 15 | 20+ |
| Papers citing | 0 | 0 | 1 | 5 | 8+ |
| Named lab adoptions | 0 | 0 | 0 | 1 | 3+ |
| Test coverage | 85% | 85% | 88% | 90% | 90%+ |
| Open issues | 5 | 20 | 35 | 50 | 60+ |

Open issues is a health metric, not a failure metric. A growing issue count means the community is engaged.

---

## The One Thing

If this roadmap is too long to keep in your head, keep this:

**Phase 0 is the only phase that matters right now.**

Ship the VAD + SNR filter. Make it install in one command. Make it run in under 10 minutes. Put it in front of one real person from the ICP. Everything else is contingent on that.

The roadmap exists so that when Phase 0 works, you know exactly what to build next and why.

---

*AudioTrove Roadmap v1.0 | July 2026*
