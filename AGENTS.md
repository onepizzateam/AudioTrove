# AudioTrove: Full Upgrade Specification
## From Preprocessing Tool to End-to-End Audio ML Platform

**Goal:** Make AudioTrove faster and more end-to-end than audio.cpp by adding:
1. A Rust/PyO3 extension that eliminates the GIL bottleneck in the executor and I/O
2. GPU acceleration across the entire curation pipeline
3. Integrated inference layer (TTS, ASR, voice conversion) matching audio.cpp's model surface
4. GPU-aware training integration (F5-TTS, StyleTTS2, Piper, Matcha-TTS)
5. A single `audiotrove run` command: **raw audio → trained voice model**, one command

---

## Table of Contents
1. [New File Tree](#new-file-tree)
2. [Rust Extension (`audiotrove-core`)](#1-rust-extension-audiotrove-core)
3. [GPU Acceleration Layer](#2-gpu-acceleration-layer)
4. [Executor Overhaul](#3-executor-overhaul)
5. [GPU-Accelerated Filters & Transformers](#4-gpu-accelerated-filters--transformers)
6. [Inference Layer](#5-inference-layer)
7. [Training Layer](#6-training-layer)
8. [End-to-End Pipeline](#7-end-to-end-pipeline)
9. [CLI Overhaul](#8-cli-overhaul)
10. [`pyproject.toml` and `Cargo.toml`](#9-pyprojecttoml-and-cargotoml)
11. [Tests](#10-tests)
12. [Priority Order](#priority-order)

---

## New File Tree

Files marked `[NEW]` are created from scratch. Files marked `[MOD]` are modified.
Files marked `[RUST]` belong to the Rust extension crate.

```
AudioTrove/
├── Cargo.toml                              [NEW][RUST]  workspace root
├── Cargo.lock                              [NEW][RUST]  generated
├── audiotrove-core/                        [NEW][RUST]  Rust crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── glob.rs          fast recursive file discovery
│       ├── checkpoint.rs    lock-free SQLite checkpoint writes
│       └── decode.rs        parallel Symphonia audio decode
├── audiotrove/
│   ├── __init__.py                         [MOD] bump version, expose gpu flag
│   ├── base.py                             [MOD] add GPUFilter / GPUTransformer base classes
│   ├── document.py                         [MOD] add gpu_tensor field (Optional[torch.Tensor])
│   ├── gpu/                                [NEW]
│   │   ├── __init__.py
│   │   └── device.py        device selection, dtype helpers, memory reporting
│   ├── executor/
│   │   └── local.py                        [MOD] use Rust checkpoint + parallel decode
│   ├── filters/
│   │   ├── vad.py                          [MOD] GPU-pinned Silero VAD path
│   │   ├── snr.py                          [MOD] torch-based SNR on GPU
│   │   └── duration.py                     [UNCHANGED]
│   ├── transformers/
│   │   ├── silence_trim.py                 [MOD] tensor path for silence trim
│   │   ├── whisper_transcribe.py           [MOD] faster-whisper + CUDA
│   │   └── diarize.py                      [MOD] GPU diarization path
│   ├── io/
│   │   ├── readers.py                      [MOD] delegate to Rust decode when available
│   │   └── writers.py                      [UNCHANGED]
│   ├── exporters/
│   │   └── tts_manifest.py                 [MOD] add Parquet export format
│   ├── inference/                          [NEW]  — compete with audio.cpp inference surface
│   │   ├── __init__.py
│   │   ├── base.py          InferenceSession ABC
│   │   ├── tts.py           TTS runners: F5-TTS, StyleTTS2, Piper, Chatterbox, Matcha
│   │   ├── asr.py           ASR runners: faster-whisper, Qwen3-ASR, Parakeet
│   │   ├── vc.py            Voice conversion: SeedVC, RVC wrappers
│   │   ├── vad.py           Standalone VAD inference session
│   │   └── server.py        Minimal HTTP server wrapping inference sessions
│   ├── training/                           [NEW]
│   │   ├── __init__.py
│   │   ├── base.py          BaseTrainer ABC
│   │   ├── f5tts.py         F5-TTS trainer wrapper
│   │   ├── styletts2.py     StyleTTS2 trainer wrapper
│   │   ├── piper.py         Piper trainer wrapper
│   │   ├── matcha.py        Matcha-TTS trainer wrapper
│   │   └── gpu.py           Multi-GPU setup, DDP helpers
│   ├── pipelines/
│   │   ├── tts.py                          [MOD] accept device/train/infer kwargs
│   │   └── e2e.py                          [NEW] curate → transcribe → train → infer
│   └── cli/
│       └── main.py                         [MOD] add train / infer / run commands
├── pyproject.toml                          [MOD] new extras: gpu, train, infer, rust
└── tests/
    ├── test_gpu_vad.py                     [NEW]
    ├── test_rust_checkpoint.py             [NEW]
    ├── test_inference_tts.py               [NEW]
    ├── test_inference_asr.py               [NEW]
    ├── test_training_f5tts.py              [NEW]
    └── test_e2e_pipeline.py                [NEW]
```

---

## 1. Rust Extension (`audiotrove-core`)

### Why Rust here

The current 4-worker scaling is only 1.57× (vs expected 4×) because:
- `LocalExecutor._run_parallel` holds the GIL for the SQLite checkpoint writes in the main loop
- `LocalAudioReader` iterates files in Python, paying per-file Python overhead
- `as_completed` in the parallel executor drains futures in the main thread, adding serialisation

Rust with PyO3 removes all three bottlenecks without changing the Python API surface.

---

### `Cargo.toml` (workspace root) — **NEW**

```toml
[workspace]
members = ["audiotrove-core"]
resolver = "2"
```

---

### `audiotrove-core/Cargo.toml` — **NEW**

```toml
[package]
name = "audiotrove-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "audiotrove_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
rusqlite = { version = "0.31", features = ["bundled"] }
symphonia = { version = "0.5", features = ["all"] }
walkdir = "2"
rayon = "1"
```

---

### `audiotrove-core/src/lib.rs` — **NEW**

**Scope:** PyO3 module root. Registers three sub-modules: `glob`, `checkpoint`, `decode`.

```rust
use pyo3::prelude::*;

mod checkpoint;
mod decode;
mod glob;

#[pymodule]
fn audiotrove_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<checkpoint::CheckpointStore>()?;
    m.add_function(wrap_pyfunction!(glob::discover_files, m)?)?;
    m.add_function(wrap_pyfunction!(decode::decode_audio_batch, m)?)?;
    Ok(())
}
```

---

### `audiotrove-core/src/glob.rs` — **NEW**

**Scope:** `discover_files(patterns: Vec<str>, extensions: Vec<str>) -> Vec<str>`

Uses `walkdir` with Rayon parallel iteration to walk directory trees and return matching paths.
Replaces `fsspec` glob in `LocalAudioReader` — no GIL hold during directory traversal.

Key logic:
- Accepts the same glob pattern list that `LocalAudioReader` currently builds
- Filters by extension set
- Returns `Vec<String>` of absolute paths sorted for determinism
- Called from Python as `audiotrove_core.discover_files(patterns, extensions)`

---

### `audiotrove-core/src/checkpoint.rs` — **NEW**

**Scope:** `CheckpointStore` — a thread-safe SQLite wrapper exposed to Python.

```rust
#[pyclass]
pub struct CheckpointStore {
    conn: Mutex<Connection>,
}

#[pymethods]
impl CheckpointStore {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> { ... }

    pub fn is_processed(&self, doc_id: &str) -> PyResult<bool> { ... }

    pub fn mark_processed(&self, doc_id: &str) -> PyResult<()> { ... }

    pub fn mark_batch(&self, doc_ids: Vec<String>) -> PyResult<()> {
        // single transaction for a batch — key for throughput
    }
}
```

The `mark_batch` method is the critical addition: it lets the executor accumulate a batch of completed doc IDs and commit them in one SQLite transaction rather than one commit per document. This is the main reason parallel executor throughput stalls at scale.

Replaces `LocalExecutor._init_db`, `_is_processed`, `_mark_processed` completely.

---

### `audiotrove-core/src/decode.rs` — **NEW**

**Scope:** `decode_audio_batch(paths: Vec<str>, target_sr: u32) -> Vec<(ndarray, u32, str)>`

Uses Symphonia to decode audio (WAV, FLAC, MP3, OGG, OPUS) in parallel via Rayon thread pool.
Returns raw f32 sample arrays as numpy arrays via PyO3's `numpy` bridge.

Each returned tuple is `(samples: np.ndarray, actual_sr: u32, path: str)`.
Downmix to mono happens inside Rust. Resampling to `target_sr` uses linear interpolation
(rubato integration optional for higher quality).

This replaces the Python `torchaudio.load` call in `LocalAudioReader.read` and the per-file
overhead of Python iteration.

---

## 2. GPU Acceleration Layer

### `audiotrove/gpu/__init__.py` — **NEW**

Empty init, re-exports `get_device`.

---

### `audiotrove/gpu/device.py` — **NEW**

**Scope:** Centralised device resolution used by all GPU-aware components.

```python
import torch

def get_device(preference: str = "auto") -> torch.device:
    """
    preference: "auto" | "cuda" | "mps" | "cpu"
    "auto" picks CUDA > MPS > CPU in that order.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")

def device_info() -> dict:
    """Return a dict of detected backend info for CLI display."""
    ...

def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor with correct dtype for the backend (fp16 on CUDA, fp32 on MPS/CPU)."""
    ...
```

---

### `audiotrove/base.py` — **MOD**

**Scope:** Add two new abstract base classes after the existing three.

```python
class GPUFilter(AudioFilter):
    """AudioFilter that receives a device and can operate on GPU tensors."""

    @property
    @abstractmethod
    def device(self) -> "torch.device":
        pass

    def to(self, device: "torch.device") -> "GPUFilter":
        """Move internal models to device. Returns self."""
        return self


class GPUTransformer(AudioTransformer):
    """AudioTransformer that operates on GPU tensors."""

    @property
    @abstractmethod
    def device(self) -> "torch.device":
        pass

    def to(self, device: "torch.device") -> "GPUTransformer":
        return self
```

No changes to existing `AudioFilter`, `AudioTransformer`, `AudioFanOutTransformer`.

---

### `audiotrove/document.py` — **MOD**

**Scope:** Add optional `gpu_tensor` field to `AudioDocument`.

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class AudioDocument:
    audio: np.ndarray
    sample_rate: int
    source_path: str
    duration_seconds: float
    doc_id: str
    metadata: dict = field(default_factory=dict)
    gpu_tensor: Optional["torch.Tensor"] = field(default=None, repr=False)  # NEW
```

Rules for `gpu_tensor`:
- Set by `GPUFilter`/`GPUTransformer` components that move audio to GPU to avoid redundant host↔device transfers across adjacent GPU steps
- Cleared (set to `None`) before pickling (add `__getstate__` / `__setstate__` like VAD does)
- The CPU `audio` field always remains the source of truth; `gpu_tensor` is a cache

---

## 3. Executor Overhaul

### `audiotrove/executor/local.py` — **MOD**

**What changes:**

#### 3a. Swap SQLite backend

```python
# BEFORE (top of __init__):
self._conn = None  # sqlite3.Connection

# AFTER:
try:
    from audiotrove_core import CheckpointStore
    self._store = CheckpointStore(checkpoint_path) if checkpoint_path else None
    self._use_rust_checkpoint = True
except ImportError:
    self._use_rust_checkpoint = False
    self._conn = None   # falls back to existing sqlite3 path
```

Replace all calls to `_init_db`, `_is_processed`, `_mark_processed` with:

```python
def _is_processed(self, doc_id: str) -> bool:
    if self._use_rust_checkpoint:
        return self._store.is_processed(doc_id) if self._store else False
    # existing sqlite3 path unchanged
    ...

def _mark_processed(self, doc_id: str) -> None:
    if self._use_rust_checkpoint:
        if self._store:
            self._store.mark_processed(doc_id)
        return
    # existing path unchanged
    ...
```

New method `_mark_batch(doc_ids: list[str])` — calls `self._store.mark_batch(doc_ids)` for
the parallel path, committing all completed docs in one transaction per batch of futures.

#### 3b. Batch future draining in `_run_parallel`

```python
# BEFORE: drain one future at a time
for future in as_completed(future_to_doc):
    ...
    self._mark_processed(doc_id)

# AFTER: drain in chunks, commit checkpoint in one batch per chunk
BATCH = 64
futures = list(future_to_doc.keys())
for i in range(0, len(futures), BATCH):
    chunk = futures[i : i + BATCH]
    completed_ids = []
    for future in as_completed(chunk):
        docs, keep, error, error_block, rejected = future.result()
        # ... existing result handling ...
        completed_ids.append(future_to_doc[future].doc_id)
    self._mark_batch(completed_ids)   # one SQLite transaction per 64 docs
```

#### 3c. GPU pipeline mode

Add `device: str = "cpu"` parameter to `LocalExecutor.__init__`.

Before the pipeline loop in `_run_sequential` and before submitting to the executor in
`_run_parallel`, call:

```python
def _move_pipeline_to_device(self, device: torch.device):
    from audiotrove.base import GPUFilter, GPUTransformer
    for block in self.pipeline:
        if isinstance(block, (GPUFilter, GPUTransformer)):
            block.to(device)
```

In `_run_sequential`, if `self.device != "cpu"` and a doc comes back from a GPU block,
reuse its `gpu_tensor` as input to the next GPU block rather than round-tripping through numpy.

---

## 4. GPU-Accelerated Filters & Transformers

### `audiotrove/filters/vad.py` — **MOD**

**What changes in `SileroVADFilter`:**

#### 4a. GPU device parameter

```python
class SileroVADFilter(GPUFilter):   # was: AudioFilter
    name = "silero_vad"

    def __init__(
        self,
        min_speech_ratio: float = 0.3,
        threshold: float = 0.5,
        window_size_samples: int = 512,
        device: str = "cpu",   # NEW
    ):
        ...
        self._device = torch.device(device)
```

#### 4b. GPU model placement

```python
@property
def model(self):
    if self._model is None:
        ...existing load logic...
        self._model = self._model.to(self._device)  # NEW: move to GPU
    return self._model
```

#### 4c. GPU inference path in `filter()`

```python
def filter(self, doc: AudioDocument) -> bool:
    # If doc already has a gpu_tensor, use it directly (skip host→device copy)
    if doc.gpu_tensor is not None and doc.gpu_tensor.device == self._device:
        audio_t = doc.gpu_tensor
    else:
        audio_t = torch.from_numpy(doc.audio).to(self._device)

    with _SILERO_INFERENCE_LOCK:
        timestamps = get_speech_timestamps(
            audio_t, model, sampling_rate=sr,
            threshold=self.threshold,
            window_size_samples=self.window_size,
        )

    # Store gpu_tensor on doc for downstream GPU blocks to reuse
    doc.gpu_tensor = audio_t
    ...
```

#### 4d. `to()` method

```python
def to(self, device: torch.device) -> "SileroVADFilter":
    self._device = device
    if self._model is not None:
        self._model = self._model.to(device)
    return self
```

Same changes apply identically to `VADSegmenter`.

---

### `audiotrove/filters/snr.py` — **MOD**

**What changes:**

Replace numpy operations with torch equivalents when `doc.gpu_tensor` is available,
keeping the numpy path as fallback.

```python
class SNRFilter(GPUFilter):   # was: AudioFilter
    def __init__(self, min_snr_db: float = 15.0, device: str = "cpu"):
        self.min_snr_db = min_snr_db
        self._device = torch.device(device)

    def _compute_snr_gpu(self, doc: AudioDocument) -> float:
        """Torch-based SNR on the gpu_tensor already present on doc."""
        import torch
        audio_t = doc.gpu_tensor   # already on device
        timestamps = doc.metadata.get("vad_speech_timestamps")
        if not timestamps:
            return self._energy_fallback_snr(doc.audio)   # CPU fallback

        n = audio_t.shape[0]
        mask = torch.zeros(n, dtype=torch.bool, device=self._device)
        for ts in timestamps:
            mask[ts["start"]:ts["end"]] = True

        speech = audio_t[mask]
        noise = audio_t[~mask]

        if noise.numel() < doc.sample_rate * 0.1:
            return 40.0

        sp = speech.pow(2).mean() if speech.numel() > 0 else torch.tensor(0.0)
        np_ = noise.pow(2).mean() if noise.numel() > 0 else torch.tensor(0.0)
        if np_.item() == 0.0:
            return 40.0
        return float(10 * torch.log10(sp / (np_ + 1e-10)).cpu())

    def filter(self, doc: AudioDocument) -> bool:
        if doc.gpu_tensor is not None:
            snr_db = self._compute_snr_gpu(doc)
        else:
            snr_db = self._compute_snr(doc)   # existing CPU path, unchanged
        doc.metadata["snr_db"] = round(snr_db, 2)
        return snr_db >= self.min_snr_db
```

---

### `audiotrove/transformers/silence_trim.py` — **MOD**

**What changes:**

```python
class SilenceTrimmingTransformer(GPUTransformer):   # was: AudioTransformer
    def __init__(self, padding_ms: int = 150, device: str = "cpu"):
        self.padding_ms = padding_ms
        self._device = torch.device(device)

    def transform(self, doc: AudioDocument) -> AudioDocument:
        # If gpu_tensor present: perform trim on GPU, update doc.audio from result
        if doc.gpu_tensor is not None:
            trimmed_t = self._trim_gpu(doc)
            doc.gpu_tensor = trimmed_t
            doc.audio = trimmed_t.cpu().numpy()
        else:
            # existing numpy path unchanged
            ...
        return doc

    def _trim_gpu(self, doc: AudioDocument) -> "torch.Tensor":
        import torch
        timestamps = doc.metadata.get("vad_speech_timestamps", [])
        if not timestamps:
            return doc.gpu_tensor
        padding = int(self.padding_ms * doc.sample_rate / 1000)
        start = max(0, timestamps[0]["start"] - padding)
        end = min(doc.gpu_tensor.shape[0], timestamps[-1]["end"] + padding)
        trimmed = doc.gpu_tensor[start:end]
        # shift timestamps to trimmed origin
        shift = timestamps[0]["start"] - padding
        doc.metadata["vad_speech_timestamps"] = [
            {"start": max(0, t["start"] - shift), "end": max(0, t["end"] - shift)}
            for t in timestamps
        ]
        doc.duration_seconds = float(trimmed.shape[0]) / doc.sample_rate
        return trimmed
```

---

### `audiotrove/transformers/whisper_transcribe.py` — **MOD**

**What changes:**

Add `faster-whisper` (CTranslate2 backend) as the preferred GPU transcription path.
Fall back to `openai-whisper` on CPU when faster-whisper is not installed.

```python
class WhisperTranscriber(GPUTransformer):
    name = "whisper_transcriber"

    def __init__(self, model_name: str = "base", device: str = "cpu",
                 compute_type: str = "auto"):
        self.model_name = model_name
        self._device_str = device
        self._compute_type = compute_type
        self._model = None
        self._backend = None  # "faster_whisper" | "openai_whisper"

    @property
    def model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                ct = "float16" if self._device_str == "cuda" else "int8"
                if self._compute_type != "auto":
                    ct = self._compute_type
                self._model = WhisperModel(
                    self.model_name, device=self._device_str, compute_type=ct
                )
                self._backend = "faster_whisper"
            except ImportError:
                # Fall back to openai-whisper
                import whisper
                self._model = whisper.load_model(self.model_name)
                self._backend = "openai_whisper"
        return self._model

    def transform(self, doc: AudioDocument) -> AudioDocument:
        import numpy as np
        audio = doc.audio.astype(np.float32)
        if self._backend == "faster_whisper":
            segments, _ = self.model.transcribe(audio, beam_size=5)
            text = " ".join(s.text for s in segments).strip()
        else:
            import whisper
            result = whisper.transcribe(self.model, audio, fp16=False)
            text = result.get("text", "").strip()
        doc.metadata["transcription"] = text
        doc.metadata["transcription_backend"] = self._backend
        return doc

    def to(self, device: torch.device) -> "WhisperTranscriber":
        self._device_str = str(device)
        self._model = None   # force reload on next access
        return self
```

**Installer note:** `faster-whisper` requires `pip install audiotrove[transcribe-gpu]` (new extra).

---

### `audiotrove/transformers/diarize.py` — **MOD**

**What changes:**

`SpeakerDiarizationTransformer` already uses pyannote. Add `device` parameter:

```python
class SpeakerDiarizationTransformer(GPUTransformer):
    def __init__(self, hf_token, min_speakers=None, max_speakers=None, device="cpu"):
        self._device = device
        ...

    @property
    def pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline
            import torch
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=self._hf_token
            )
            self._pipeline.to(torch.device(self._device))  # NEW: GPU placement
        return self._pipeline
```

---

### `audiotrove/io/readers.py` — **MOD**

**What changes:** Try Rust decoder first; fall back to torchaudio/soundfile.

```python
def _try_rust_batch_decode(paths: list[str], target_sr: int):
    try:
        from audiotrove_core import decode_audio_batch
        return decode_audio_batch(paths, target_sr)
    except ImportError:
        return None

class LocalAudioReader:
    def __init__(self, patterns, ...):
        # Try Rust file discovery first
        try:
            from audiotrove_core import discover_files
            self._use_rust_glob = True
        except ImportError:
            self._use_rust_glob = False
        ...

    def __iter__(self):
        if self._use_rust_glob:
            paths = discover_files(self.patterns, self.extensions)
        else:
            paths = self._python_glob()   # existing fsspec path

        # Optionally batch-decode via Rust
        # (Used only for sequential mode to avoid serialisation overhead in parallel mode)
        for path in paths:
            yield self._read_single(path)   # existing logic per file
```

Batch decode path is wired into `LocalExecutor._run_sequential` only, since the parallel
executor already gives each worker its own decode work.

---

## 5. Inference Layer

**Purpose:** Compete with audio.cpp's inference surface from Python, with the same GPU backends.
Users who curate with AudioTrove can immediately run inference without switching tools.

---

### `audiotrove/inference/base.py` — **NEW**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class InferenceResult:
    audio: np.ndarray | None          # for TTS / VC / source sep
    text: str | None                  # for ASR / alignment
    sample_rate: int
    metadata: dict

class InferenceSession(ABC):
    """Base class for all AudioTrove inference sessions."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights onto device."""

    @abstractmethod
    def run(self, **kwargs) -> InferenceResult:
        """Execute one inference request."""

    def unload(self) -> None:
        """Release GPU memory."""

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.unload()
```

---

### `audiotrove/inference/tts.py` — **NEW**

**Scope:** TTS inference sessions. Each is a thin wrapper that:
1. Locates the model (local path or downloads via HuggingFace Hub)
2. Loads onto the requested device
3. Exposes `run(text, voice_ref_path=None) -> InferenceResult`

Supported families (one class per family, all inherit `InferenceSession`):

| Class | Backed by | Notes |
|---|---|---|
| `F5TTSSession` | f5-tts library | Voice cloning via ref audio |
| `StyleTTS2Session` | StyleTTS2 | Zero-shot voice cloning |
| `PiperSession` | piper-tts | Fast CPU/GPU synthesis |
| `ChatterboxSession` | chatterbox-tts | Multilingual clone |
| `MatchaTTSSession` | matcha-tts | Flow-matching TTS |

Each class pattern:

```python
class F5TTSSession(InferenceSession):
    def __init__(self, model_path: str, device: str = "auto", voice_ref: str | None = None):
        ...

    def load(self) -> None:
        from f5_tts.api import F5TTS
        self._model = F5TTS(model_type="F5-TTS", ckpt_file=self.model_path,
                            device=self.device)

    def run(self, text: str, voice_ref: str | None = None, **kwargs) -> InferenceResult:
        ref = voice_ref or self.voice_ref
        wav, sr, _ = self._model.infer(ref_file=ref, ref_text="", gen_text=text)
        return InferenceResult(audio=wav, text=None, sample_rate=sr, metadata={})
```

Factory function for CLI use:

```python
def get_tts_session(family: str, **kwargs) -> InferenceSession:
    registry = {
        "f5tts": F5TTSSession,
        "styletts2": StyleTTS2Session,
        "piper": PiperSession,
        "chatterbox": ChatterboxSession,
        "matcha": MatchaTTSSession,
    }
    cls = registry.get(family)
    if cls is None:
        raise ValueError(f"Unknown TTS family: {family}. Choose from {list(registry)}")
    return cls(**kwargs)
```

---

### `audiotrove/inference/asr.py` — **NEW**

**Scope:** ASR inference sessions.

| Class | Backed by | Notes |
|---|---|---|
| `FasterWhisperSession` | faster-whisper | CUDA/CPU, streaming |
| `Qwen3ASRSession` | transformers (Qwen3-ASR) | multilingual, 100+ langs |
| `ParakeetSession` | nemo (Parakeet-TDT) | CUDA optimised |

Same pattern as TTS sessions. `run(audio_path: str) -> InferenceResult` where result.text
holds the transcript and result.metadata holds word timestamps if available.

```python
def get_asr_session(family: str = "faster_whisper", **kwargs) -> InferenceSession:
    ...
```

---

### `audiotrove/inference/vc.py` — **NEW**

**Scope:** Voice conversion sessions.

| Class | Backed by |
|---|---|
| `SeedVCSession` | SeedVC |
| `RVCSession` | RVC (v2) |

`run(source_audio_path: str, target_voice_path: str) -> InferenceResult`

---

### `audiotrove/inference/server.py` — **NEW**

**Scope:** Minimal HTTP server to serve loaded inference sessions (competes with `audiocpp_server`).

Uses `aiohttp` (lightweight, no FastAPI dependency). Exposes:

```
GET  /health
GET  /v1/models
POST /v1/audio/speech         { text, voice_ref?, family? }  → audio/wav
POST /v1/audio/transcriptions { audio (multipart) }          → { text }
POST /v1/tasks/run            { task, params }               → { result }
```

Routes map directly to the session registry in `inference/tts.py` and `inference/asr.py`.
Sessions are lazy-loaded on first request and kept warm (same behaviour as audio.cpp server).

```python
class AudioTroveServer:
    def __init__(self, config_path: str):
        self.config = json.load(open(config_path))
        self._sessions: dict[str, InferenceSession] = {}

    def _get_session(self, model_id: str) -> InferenceSession:
        if model_id not in self._sessions:
            spec = next(m for m in self.config["models"] if m["id"] == model_id)
            session = get_tts_session(spec["family"], **spec.get("options", {}))
            session.load()
            self._sessions[model_id] = session
        return self._sessions[model_id]

    async def run(self):
        # aiohttp app setup
        ...
```

---

## 6. Training Layer

### `audiotrove/training/base.py` — **NEW**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    manifest_path: str           # filelist.txt or metadata.csv from TTSManifestExporter
    output_dir: str              # where to write checkpoints and final model
    model_name: str              # base model to fine-tune
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    device: str = "auto"         # "auto" | "cuda" | "mps" | "cpu"
    num_gpus: int = 1            # DDP when > 1
    mixed_precision: bool = True # fp16 training on CUDA
    resume_from: str | None = None

class BaseTrainer(ABC):
    def __init__(self, config: TrainingConfig):
        self.config = config

    @abstractmethod
    def validate_manifest(self) -> None:
        """Raise if the manifest is missing required columns."""

    @abstractmethod
    def train(self) -> dict:
        """Run training. Returns metrics dict."""

    @abstractmethod
    def export(self, output_path: str) -> str:
        """Export final model to output_path. Returns path."""
```

---

### `audiotrove/training/f5tts.py` — **NEW**

**Scope:** Fine-tune F5-TTS on an AudioTrove-produced `filelist.txt`.

Key responsibilities:
1. `validate_manifest()` — check that filelist.txt exists, has ≥ 10 clips, all WAVs readable
2. `train()` — calls `f5_tts.train.train(...)` with the right config dict
3. `export()` — copies the best checkpoint to output_path

```python
class F5TTSTrainer(BaseTrainer):

    def validate_manifest(self) -> None:
        path = Path(self.config.manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        if len(lines) < 10:
            raise ValueError(f"F5-TTS needs ≥ 10 clips; found {len(lines)}")
        for line in lines:
            wav_path, _, _ = line.split("\t")
            if not Path(wav_path).exists():
                raise FileNotFoundError(f"Missing WAV: {wav_path}")

    def train(self) -> dict:
        # Requires: pip install audiotrove[train-f5tts]
        from f5_tts.train import train as f5_train
        from audiotrove.gpu.device import get_device

        device = get_device(self.config.device)
        cfg = {
            "dataset_path": str(Path(self.config.manifest_path).parent),
            "output_dir": self.config.output_dir,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "device": str(device),
            "mixed_precision": self.config.mixed_precision and device.type == "cuda",
        }
        if self.config.num_gpus > 1:
            cfg["num_gpus"] = self.config.num_gpus   # triggers DDP inside f5_train
        return f5_train(**cfg)

    def export(self, output_path: str) -> str:
        import shutil
        best = max(
            Path(self.config.output_dir).glob("**/*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        shutil.copy(best, output_path)
        return output_path
```

---

### `audiotrove/training/styletss2.py` — **NEW**

Same pattern as `F5TTSTrainer`. Wraps StyleTTS2's `train.py` script via `subprocess` with
GPU flags derived from `TrainingConfig.device` and `num_gpus`. Converts AudioTrove's
`metadata.csv` (LJSpeech format) into StyleTTS2's expected data layout before launching.

---

### `audiotrove/training/piper.py` — **NEW**

Wraps `piper-train` CLI. Converts `filelist.txt` to Piper's JSON manifest format, then
calls `python -m piper_train ...` as a subprocess with the right GPU env vars.

---

### `audiotrove/training/matcha.py` — **NEW**

Wraps Matcha-TTS training. Converts manifest to Matcha-TTS format, invokes training with
Hydra config override for device and batch size.

---

### `audiotrove/training/gpu.py` — **NEW**

**Scope:** DDP setup helpers used by trainers when `num_gpus > 1`.

```python
def launch_ddp(train_fn, num_gpus: int, **kwargs):
    """Launch train_fn across num_gpus using torch.multiprocessing.spawn."""
    import torch.multiprocessing as mp
    mp.spawn(train_fn, args=(num_gpus, kwargs), nprocs=num_gpus, join=True)

def setup_ddp(rank: int, world_size: int):
    import torch.distributed as dist
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    import torch.distributed as dist
    dist.destroy_process_group()
```

---

## 7. End-to-End Pipeline

### `audiotrove/pipelines/e2e.py` — **NEW**

**Scope:** `e2e_pipeline()` — the "one command" function that chains curation → transcription
→ training → (optional) inference validation.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class E2EConfig:
    input_path: str
    output_path: str
    # Curation
    min_duration: float = 2.0
    max_duration: float = 15.0
    snr_min: float = 20.0
    extensions: list[str] = None
    workers: int = 1
    # Device
    device: str = "auto"
    # Transcription
    transcribe: bool = True
    whisper_model: str = "base"
    # Training
    train: bool = False
    train_framework: Literal["f5tts", "styletss2", "piper", "matcha"] = "f5tts"
    epochs: int = 100
    batch_size: int = 16
    num_gpus: int = 1
    # Inference validation (optional smoke test after training)
    validate_inference: bool = False
    validate_text: str = "Hello, this is a test of the trained voice."
    validate_voice_ref: str | None = None


def e2e_pipeline(config: E2EConfig) -> dict:
    """
    Full pipeline: curate → transcribe → [train] → [validate].

    Returns a summary dict with keys:
      curate_summary, train_summary, validation_audio_path
    """
    from audiotrove.pipelines.tts import tts_pipeline
    from pathlib import Path

    output = Path(config.output_path)

    # --- Step 1: Curate ---
    curate_summary = tts_pipeline(
        input_path=config.input_path,
        output_path=str(output / "curated"),
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        snr_min=config.snr_min,
        extensions=config.extensions or ["wav"],
        workers=config.workers,
        transcribe=config.transcribe,
        whisper_model=config.whisper_model,
        device=config.device,   # propagated to GPU filters
    )

    if curate_summary["kept"] == 0:
        raise RuntimeError("Curation produced zero clips. Check input audio and filters.")

    result = {"curate_summary": curate_summary, "train_summary": None, "validation_audio_path": None}

    # --- Step 2: Train ---
    if config.train:
        from audiotrove.training import get_trainer
        from audiotrove.training.base import TrainingConfig

        manifest = str(output / "curated" / "filelist.txt")
        train_output = str(output / "model")

        trainer = get_trainer(
            config.train_framework,
            TrainingConfig(
                manifest_path=manifest,
                output_dir=train_output,
                model_name=config.train_framework,
                epochs=config.epochs,
                batch_size=config.batch_size,
                device=config.device,
                num_gpus=config.num_gpus,
            ),
        )
        trainer.validate_manifest()
        train_summary = trainer.train()
        model_path = trainer.export(str(output / "model" / "final.pt"))
        result["train_summary"] = train_summary

        # --- Step 3: Validate (optional smoke test) ---
        if config.validate_inference:
            from audiotrove.inference.tts import get_tts_session
            import soundfile as sf
            session = get_tts_session(
                config.train_framework,
                model_path=model_path,
                device=config.device,
                voice_ref=config.validate_voice_ref,
            )
            with session:
                r = session.run(text=config.validate_text)
            val_path = str(output / "validation.wav")
            sf.write(val_path, r.audio, r.sample_rate)
            result["validation_audio_path"] = val_path

    return result
```

---

### `audiotrove/pipelines/tts.py` — **MOD**

**What changes:** Add `device: str = "cpu"` parameter, propagate to all GPU-aware blocks.

```python
def tts_pipeline(
    ...
    device: str = "cpu",   # NEW
) -> dict:
    ...
    pipeline = [
        *([VADSegmenter(device=device)] if segment else []),
        SileroVADFilter(min_speech_ratio=0.1, device=device),    # MOD: +device
        SilenceTrimmingTransformer(padding_ms=padding_ms, device=device),  # MOD
        SNRFilter(min_snr_db=snr_min, device=device),             # MOD
        DurationBucketFilter(min_duration, max_duration),         # unchanged
    ]
    ...
    executor = LocalExecutor(
        pipeline=pipeline,
        checkpoint_path=...,
        num_workers=workers,
        device=device,   # MOD: new param
    )
```

---

## 8. CLI Overhaul

### `audiotrove/cli/main.py` — **MOD**

**Three new commands added:**

#### 8a. `audiotrove train`

```
audiotrove train MANIFEST_PATH OUTPUT_PATH
  --framework [f5tts|styletss2|piper|matcha]  default: f5tts
  --epochs INTEGER                             default: 100
  --batch-size INTEGER                         default: 16
  --device [auto|cuda|mps|cpu]                 default: auto
  --num-gpus INTEGER                           default: 1
  --resume-from PATH
```

Maps directly to `BaseTrainer.train()` + `BaseTrainer.export()`.
Displays a Rich progress table with epoch, loss, and ETA.

#### 8b. `audiotrove infer`

```
audiotrove infer
  --task [tts|asr|vc]         required
  --family TEXT               e.g. f5tts, faster_whisper, seed_vc
  --model PATH                local model path
  --text TEXT                 TTS input
  --audio PATH                ASR / VC source audio
  --voice-ref PATH            TTS / VC reference voice
  --device [auto|cuda|mps|cpu]
  --out PATH                  output WAV (TTS/VC) or prints transcript (ASR)
```

Maps to `get_tts_session` / `get_asr_session` / `get_vc_session`.

#### 8c. `audiotrove run` (the Product Hunt command)

```
audiotrove run INPUT_PATH OUTPUT_PATH
  [all curate flags]
  --train / --no-train          default: --train
  --framework TEXT              default: f5tts
  --epochs INTEGER
  --batch-size INTEGER
  --num-gpus INTEGER
  --device [auto|cuda|mps|cpu]
  --validate / --no-validate    run a quick TTS smoke test after training
  --validate-text TEXT
  --validate-voice-ref PATH
```

Wraps `e2e_pipeline()`. Prints a Rich multi-step progress display:

```
AudioTrove end-to-end pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [1/3] Curating audio...          ✓  2123 clips kept, 580 filtered (32.1 min)
  [2/3] Training F5-TTS (GPU)...   ✓  100 epochs, best loss 0.041 (14.3 min)
  [3/3] Validating...              ✓  output/validation.wav
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total: 46.7 minutes  |  Model: output/model/final.pt
```

#### 8d. `audiotrove serve` (new)

```
audiotrove serve CONFIG_PATH
  --host TEXT    default: 127.0.0.1
  --port INTEGER default: 8080
```

Starts `AudioTroveServer`. Config JSON follows the same schema as `audiocpp_server`:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "models": [
    {
      "id": "my-voice",
      "task": "tts",
      "family": "f5tts",
      "model_path": "./model/final.pt",
      "options": { "device": "cuda" }
    }
  ]
}
```

#### 8e. Existing `curate` command — MOD

Add `--device` flag:

```
  --device [auto|cuda|mps|cpu]   default: cpu
```

Passes through to `tts_pipeline(device=device)`.

---

## 9. `pyproject.toml` and `Cargo.toml`

### `pyproject.toml` — **MOD**

#### New optional-dependencies extras

```toml
[project.optional-dependencies]
# existing extras unchanged
enhance = ["deepfilternet"]
s3 = ["s3fs>=2023.1.0"]
gcs = ["gcsfs>=2023.1.0"]
diarize = ["pyannote.audio>=3.1.0"]
embed-dedup = ["faiss-cpu>=1.7.4", "speechbrain>=1.0.0"]
transcribe = ["openai-whisper>=20231117"]

# NEW extras
gpu = [
    "torch>=2.0.0",          # already a core dep but listed for clarity
]
transcribe-gpu = [
    "faster-whisper>=1.0.0", # CTranslate2-based, CUDA-accelerated
]
train-f5tts = [
    "f5-tts>=1.0.0",
]
train-styletss2 = [
    "styletts2>=0.1.0",
]
train-piper = [
    "piper-train>=1.0.0",
]
train-matcha = [
    "matcha-tts>=0.2.0",
]
infer = [
    "faster-whisper>=1.0.0",
    "f5-tts>=1.0.0",
    "aiohttp>=3.9.0",
]
rust = [
    # maturin builds the Rust extension; listed here as reminder
    # actual build: `pip install maturin && maturin develop`
]
all = [
    "audiotrove[gpu,transcribe-gpu,train-f5tts,infer,diarize,enhance]",
]
dev = [
    "pytest",
    "pytest-cov",
    "ruff==0.15.22",
    "mypy",
    "click[testing]",
    "scipy>=1.10.0",
    "torchcodec",
    "maturin>=1.5",   # NEW: for building Rust extension in dev
]
```

#### New build-system entry for maturin (optional, for Rust extension)

Add a `[tool.maturin]` section:

```toml
[tool.maturin]
python-source = "."
manifest-path = "audiotrove-core/Cargo.toml"
features = ["pyo3/extension-module"]
```

Note: The pure-Python package continues to install and function without the Rust extension.
The Rust extension is an optional performance layer. When not installed, the existing Python
paths are used transparently.

---

## 10. Tests

### `tests/test_rust_checkpoint.py` — **NEW**

```
pytest tests/test_rust_checkpoint.py
```

Tests:
- `CheckpointStore` creates a DB at the given path
- `is_processed` returns False for unknown doc_ids
- `mark_processed` records a doc_id; subsequent `is_processed` returns True
- `mark_batch` records all doc_ids in a list; all return True
- Concurrent writes from 4 threads do not corrupt the DB (thread-safety test)

Skip entire module if `audiotrove_core` import fails (Rust not built).

---

### `tests/test_gpu_vad.py` — **NEW**

Tests:
- `SileroVADFilter(device="cpu")` behaves identically to current test suite (regression guard)
- `SileroVADFilter(device="cuda")` works when `torch.cuda.is_available()` (skip otherwise)
- `doc.gpu_tensor` is populated after `filter()` on GPU path
- `SNRFilter(device="cuda")` reuses `doc.gpu_tensor` without additional host→device copy

---

### `tests/test_inference_tts.py` — **NEW**

Tests:
- `F5TTSSession` can be instantiated without error (import guard, skip if f5-tts not installed)
- `get_tts_session("f5tts", ...)` returns `F5TTSSession`
- `get_tts_session("unknown")` raises `ValueError`
- `InferenceResult` fields are correctly typed

---

### `tests/test_inference_asr.py` — **NEW**

Tests:
- `FasterWhisperSession` transcribes `tests/fixtures/speech_clean.wav` (skip if not installed)
- Result text is a non-empty string
- `get_asr_session` factory dispatches correctly

---

### `tests/test_training_f5tts.py` — **NEW**

Tests:
- `F5TTSTrainer.validate_manifest()` raises `FileNotFoundError` for missing manifest
- `F5TTSTrainer.validate_manifest()` raises `ValueError` for manifest with < 10 clips
- `F5TTSTrainer.validate_manifest()` passes for a valid temp manifest fixture
- Full `train()` call is integration-tested only in CI with `[train-f5tts]` extra installed

---

### `tests/test_e2e_pipeline.py` — **NEW**

Tests:
- `e2e_pipeline(config)` with `train=False` produces the same output as `tts_pipeline`
  (regression test against existing behaviour)
- `e2e_pipeline(config)` with `train=True` is marked `@pytest.mark.integration`
  and skipped unless `AT_INTEGRATION=1` env var is set
- `E2EConfig` dataclass validates that `train_framework` is one of the supported values

---

## Priority Order

Implement in this sequence to get compounding value at each step:

1. **`pyproject.toml` extras** — unblocks all subsequent installation paths
2. **`audiotrove/gpu/device.py`** — used by everything GPU-aware
3. **`base.py` GPUFilter/GPUTransformer** — needed by filters/transformers
4. **`document.py` gpu_tensor field** — needed by GPU pipeline chaining
5. **`filters/vad.py` GPU path** — highest ROI; VAD is the first and most-called filter
6. **`filters/snr.py` + `transformers/silence_trim.py` GPU paths** — complete GPU curation chain
7. **`executor/local.py` batch checkpoint** — fixes the parallel scaling ceiling
8. **`transformers/whisper_transcribe.py` faster-whisper** — GPU transcription
9. **Rust extension** (`glob.rs`, `checkpoint.rs`, `decode.rs`) — parallelism floor lift
10. **`inference/` layer** — TTS/ASR sessions and server
11. **`training/` layer** — F5-TTS first, others follow
12. **`pipelines/e2e.py` + `cli/main.py` `run` command** — Product Hunt moment