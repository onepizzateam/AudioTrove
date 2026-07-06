"""
Performance benchmark for Phase 0 components.
Creates synthetic audio files and measures throughput.
"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from audiotrove.document import AudioDocument
from audiotrove.executor.local import LocalExecutor
from audiotrove.filters.snr import SNRFilter
from audiotrove.filters.vad import SileroVADFilter
from audiotrove.io.readers import LocalAudioReader
from audiotrove.io.writers import JSONLWriter
from audiotrove.utils.hashing import make_doc_id


def create_synthetic_audio_files(num_files: int, output_dir: Path, duration_s: float = 5.0):
    """Create synthetic audio files for benchmarking."""
    import torchaudio
    
    sr = 16000
    for i in range(num_files):
        # Create simple sine wave
        t = np.arange(int(duration_s * sr)) / sr
        waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        waveform = np.expand_dims(waveform, 0)  # Add channel dimension
        
        waveform_tensor = __import__('torch').from_numpy(waveform)
        audio_path = output_dir / f"audio_{i:04d}.wav"
        torchaudio.save(str(audio_path), waveform_tensor, sr)


def benchmark_reader(input_dir: Path, num_files: int):
    """Benchmark LocalAudioReader throughput."""
    start = time.time()
    reader = LocalAudioReader(str(input_dir / "*.wav"), target_sample_rate=16000)
    count = 0
    for doc in reader:
        if doc is not None:
            count += 1
        if count >= num_files:
            break
    elapsed = time.time() - start
    throughput = count / elapsed if elapsed > 0 else 0
    return throughput, count, elapsed


def benchmark_vad_filter(num_samples: int = 100):
    """Benchmark SileroVADFilter throughput."""
    vad = SileroVADFilter(min_speech_ratio=0.3)
    sr = 16000
    
    start = time.time()
    for i in range(num_samples):
        # Create synthetic audio
        t = np.arange(16000) / sr
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        doc = AudioDocument(
            audio=audio,
            sample_rate=sr,
            source_path=f"test_{i}.wav",
            duration_seconds=1.0,
            doc_id=make_doc_id(f"test_{i}.wav"),
        )
        vad.filter(doc)
    elapsed = time.time() - start
    throughput = num_samples / elapsed if elapsed > 0 else 0
    return throughput, num_samples, elapsed


def benchmark_snr_filter(num_samples: int = 100):
    """Benchmark SNRFilter throughput."""
    snr = SNRFilter(min_snr_db=15.0)
    sr = 16000
    
    start = time.time()
    for i in range(num_samples):
        # Create synthetic audio
        t = np.arange(16000) / sr
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        doc = AudioDocument(
            audio=audio,
            sample_rate=sr,
            source_path=f"test_{i}.wav",
            duration_seconds=1.0,
            doc_id=make_doc_id(f"test_{i}.wav"),
        )
        snr.filter(doc)
    elapsed = time.time() - start
    throughput = num_samples / elapsed if elapsed > 0 else 0
    return throughput, num_samples, elapsed


def benchmark_end_to_end(input_dir: Path, output_dir: Path, num_files: int):
    """Benchmark full pipeline (reader + VAD + SNR + writer)."""
    pipeline = [
        SileroVADFilter(min_speech_ratio=0.1),
        SNRFilter(min_snr_db=0.0),
    ]
    executor = LocalExecutor(
        pipeline=pipeline,
        num_workers=1,
        checkpoint_path=None,
    )
    
    reader = LocalAudioReader(
        path_pattern=str(input_dir / "*.wav"),
        min_duration_seconds=0.5,
    )
    
    manifest_path = output_dir / "manifest.jsonl"
    writer = JSONLWriter(output_path=str(manifest_path))
    
    start = time.time()
    stats = executor.run(reader, writer)
    elapsed = time.time() - start
    
    throughput = stats.get('kept', 0) / elapsed if elapsed > 0 else 0
    return throughput, stats, elapsed


if __name__ == '__main__':
    import sys
    
    # Configuration
    num_test_files = 100
    print(f"Phase 0 Performance Benchmark")
    print(f"=" * 60)
    print(f"Test files: {num_test_files}")
    print(f"Audio duration: 5 seconds per file")
    print(f"Sample rate: 16000 Hz")
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create synthetic audio files
        print("Generating synthetic audio files...")
        try:
            create_synthetic_audio_files(num_test_files, input_dir, duration_s=5.0)
            print(f"  ✓ Created {num_test_files} audio files")
        except ImportError as e:
            print(f"  ✗ Cannot create audio files (torchaudio needed): {e}")
            print("  Skipping file I/O benchmarks")
            sys.exit(1)
        
        # Benchmark reader
        print("\n1. LocalAudioReader Benchmark")
        print("-" * 60)
        throughput, count, elapsed = benchmark_reader(input_dir, num_test_files)
        print(f"  Files read: {count}/{num_test_files}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} files/sec")
        print(f"  Target: > 500 files/sec ✓" if throughput > 500 else f"  Target: > 500 files/sec ✗ (got {throughput:.1f})")
        
        # Benchmark VAD filter
        print("\n2. SileroVADFilter Benchmark")
        print("-" * 60)
        vad_throughput, vad_count, vad_elapsed = benchmark_vad_filter(num_samples=50)
        print(f"  Clips processed: {vad_count}")
        print(f"  Time: {vad_elapsed:.2f}s")
        print(f"  Throughput: {vad_throughput:.1f} clips/sec")
        print(f"  Target: > 10 clips/sec ✓" if vad_throughput > 10 else f"  Target: > 10 clips/sec ✗ (got {vad_throughput:.1f})")
        
        # Benchmark SNR filter
        print("\n3. SNRFilter Benchmark")
        print("-" * 60)
        snr_throughput, snr_count, snr_elapsed = benchmark_snr_filter(num_samples=100)
        print(f"  Clips processed: {snr_count}")
        print(f"  Time: {snr_elapsed:.2f}s")
        print(f"  Throughput: {snr_throughput:.1f} clips/sec")
        print(f"  Target: > 200 clips/sec ✓" if snr_throughput > 200 else f"  Target: > 200 clips/sec ✗ (got {snr_throughput:.1f})")
        
        # Benchmark end-to-end pipeline
        print("\n4. End-to-End Pipeline Benchmark")
        print("-" * 60)
        try:
            e2e_throughput, stats, e2e_elapsed = benchmark_end_to_end(input_dir, output_dir, num_test_files)
            print(f"  Files processed: {stats.get('total', 0)}")
            print(f"  Files kept: {stats.get('kept', 0)}")
            print(f"  Time: {e2e_elapsed:.2f}s")
            print(f"  Throughput: {e2e_throughput:.1f} clips/sec")
            print(f"  Target: > 8 clips/sec ✓" if e2e_throughput > 8 else f"  Target: > 8 clips/sec ✗ (got {e2e_throughput:.1f})")
            
            # Phase 0 gate: 100 files (5s each) should complete < 3 minutes
            theoretical_time = (num_test_files * 5.0) / e2e_throughput
            print(f"\n  Phase 0 Gate: 100 files (5s each) at {e2e_throughput:.1f} clips/sec = {theoretical_time:.1f}s")
            print(f"  Gate (< 180s): {'✓ PASS' if theoretical_time < 180 else '✗ FAIL'}")
        except Exception as e:
            print(f"  ✗ Error running end-to-end benchmark: {e}")
        
        # Save results
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_test_files": num_test_files,
            "reader": {
                "throughput_files_per_sec": throughput,
                "files_processed": count,
                "elapsed_seconds": elapsed,
            },
            "vad": {
                "throughput_clips_per_sec": vad_throughput,
                "clips_processed": vad_count,
                "elapsed_seconds": vad_elapsed,
            },
            "snr": {
                "throughput_clips_per_sec": snr_throughput,
                "clips_processed": snr_count,
                "elapsed_seconds": snr_elapsed,
            },
        }
        
        results_path = Path("benchmarks") / "phase0_baseline.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
