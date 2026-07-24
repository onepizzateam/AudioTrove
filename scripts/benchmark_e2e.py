"""
End-to-end benchmark on real audio fixtures.
Outputs real throughput numbers for README.
Run: python scripts/benchmark_e2e.py
"""

import time
import json
import tempfile
from pathlib import Path
import numpy as np
import argparse


def _make_generator_from_dir(fixtures_dir: Path, ext: str = "wav"):
    import wave as _wave
    from audiotrove.document import AudioDocument
    from audiotrove.utils.hashing import make_doc_id

    def gen():
        # recursive glob to support nested corpus dirs
        for fpath in sorted(fixtures_dir.rglob(f"*.{ext}")):
            try:
                with _wave.open(str(fpath), "rb") as wf:
                    sr = wf.getframerate()
                    nch = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
            except Exception:
                # Skip non-WAV or corrupt files
                print(f"Skipping invalid WAV: {fpath}")
                continue
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            if nch > 1:
                audio = audio.reshape(-1, nch).mean(axis=1)
            duration = float(len(audio)) / float(sr)
            yield AudioDocument(
                audio=audio,
                sample_rate=sr,
                source_path=str(fpath),
                duration_seconds=duration,
                doc_id=make_doc_id(str(fpath)),
            )

    return gen


def run():
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        default="tests/fixtures",
        help="Directory of audio files to benchmark against",
    )
    parser.add_argument("--ext", default="wav", help="Audio file extension (wav or flac)")
    args = parser.parse_args()

    CORPUS_DIR = Path(args.corpus_dir)
    EXT = args.ext

    fixtures_dir = CORPUS_DIR
    # look for files recursively
    wav_files = list(fixtures_dir.rglob(f"*.{EXT}"))

    if not wav_files:
        print(
            f"ERROR: No fixtures found in {fixtures_dir} with extension {EXT}. Run: python scripts/generate_fixtures.py or provide --corpus-dir"
        )
        sys.exit(1)

    print("AudioTrove End-to-End Benchmark")
    print("=" * 50)
    print(f"Fixture files: {len(wav_files)}")
    print()

    from audiotrove.io.readers import LocalAudioReader
    from audiotrove.io.writers import JSONLWriter
    from audiotrove.filters.vad import SileroVADFilter
    from audiotrove.filters.snr import SNRFilter
    from audiotrove.executor.local import LocalExecutor
    import platform
    import os

    results = {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "fixture_count": len(wav_files),
        "runs": {},
    }

    def bench_run(label, workers, with_segmentation=False):
        from audiotrove.filters.vad import VADSegmenter

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pipeline = [
                SileroVADFilter(min_speech_ratio=0.1),
                SNRFilter(min_snr_db=0.0),
            ]
            if with_segmentation:
                pipeline.insert(1, VADSegmenter())

            executor = LocalExecutor(
                pipeline=pipeline,
                checkpoint_path=str(tmp / "ckpt.db"),
                num_workers=workers,
            )

            # Try LocalAudioReader; if it fails at runtime, fallback to generator
            try:
                # try LocalAudioReader with recursive glob pattern
                reader = LocalAudioReader(
                    [str(fixtures_dir / f"**/*.{EXT}")], min_duration_seconds=0.0
                )
                writer = JSONLWriter(str(tmp / "manifest.jsonl"))
                start = time.perf_counter()
                stats = executor.run(reader, writer)
                elapsed = time.perf_counter() - start
            except Exception:
                gen = _make_generator_from_dir(fixtures_dir, ext=EXT)
                writer = JSONLWriter(str(tmp / "manifest.jsonl"))
                start = time.perf_counter()
                stats = executor.run(gen(), writer)
                elapsed = time.perf_counter() - start

            throughput = stats["processed"] / elapsed if elapsed > 0 else 0
            total_audio_s = 0.0
            if (tmp / "manifest.jsonl").exists():
                total_audio_s = sum(
                    float(json.loads(line)["duration_seconds"])
                    for line in (tmp / "manifest.jsonl").read_text().splitlines()
                    if line.strip()
                )

            rtf = total_audio_s / elapsed if elapsed > 0 else 0  # real-time factor

            print(f"  [{label}]")
            print(f"    Processed: {stats['processed']} files in {elapsed:.2f}s")
            print(f"    Kept: {stats['kept']} | Filtered: {stats['skipped']}")
            print(f"    Throughput: {throughput:.1f} files/sec")
            print(f"    Real-time factor: {rtf:.1f}x (audio hours / wall hours)")
            print()

            return {
                "elapsed_s": round(elapsed, 3),
                "files_processed": stats["processed"],
                "files_kept": stats["kept"],
                "throughput_files_per_sec": round(throughput, 2),
                "real_time_factor": round(rtf, 2),
            }

    print("1. Sequential (--workers 1)")
    results["runs"]["sequential"] = bench_run("VAD+SNR, 1 worker", workers=1)

    print("2. Parallel (--workers 4)")
    results["runs"]["parallel_4"] = bench_run("VAD+SNR, 4 workers", workers=4)

    print("3. With segmentation (--segment, 1 worker)")
    results["runs"]["segmentation"] = bench_run(
        "VAD+SNR+Segment, 1 worker", workers=1, with_segmentation=True
    )

    # Checkpoint resume test
    print("4. Checkpoint resume test")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = str(tmp / "ckpt.db")
        pipeline = [SileroVADFilter(min_speech_ratio=0.1), SNRFilter(min_snr_db=0.0)]
        executor = LocalExecutor(pipeline=pipeline, checkpoint_path=ckpt, num_workers=1)

        gen = _make_generator_from_dir(fixtures_dir)
        writer = JSONLWriter(str(tmp / "manifest.jsonl"))
        stats1 = executor.run(gen(), writer)

        # Second run — should skip all already-processed
        writer2 = JSONLWriter(str(tmp / "manifest2.jsonl"))
        executor2 = LocalExecutor(pipeline=pipeline, checkpoint_path=ckpt, num_workers=1)
        start = time.perf_counter()
        stats2 = executor2.run(_make_generator_from_dir(fixtures_dir)(), writer2)
        elapsed2 = time.perf_counter() - start

        print(f"  First run processed: {stats1['processed']} files")
        print(
            f"  Second run (resume): {stats2['processed']} processed, {stats2['skipped']} skipped in {elapsed2:.3f}s"
        )
        assert stats2["processed"] == 0 or stats2["skipped"] >= stats1["processed"], (
            "Checkpoint should skip all previously processed files"
        )
        print(f"  ✓ Checkpoint correctly skipped {stats2['skipped']} already-processed files")
        print()
        results["checkpoint_resume"] = {
            "first_run_processed": stats1["processed"],
            "second_run_skipped": stats2["skipped"],
            "second_run_elapsed_s": round(elapsed2, 3),
        }

    # Save results
    out_path = Path("benchmarks/e2e_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {out_path}")
    return results


if __name__ == "__main__":
    run()
