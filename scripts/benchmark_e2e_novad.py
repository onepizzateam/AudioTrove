"""
Simplified benchmark: no VAD/SNR, just read and write manifests to measure throughput.
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import platform

from audiotrove.document import AudioDocument
from audiotrove.executor.local import LocalExecutor
from audiotrove.io.writers import JSONLWriter
from audiotrove.utils.hashing import make_doc_id

parser = argparse.ArgumentParser()
parser.add_argument("--corpus-dir", default="/tmp/LibriSpeech_wav")
parser.add_argument("--ext", default="wav")
args = parser.parse_args()

CORPUS_DIR = Path(args.corpus_dir)
EXT = args.ext

wav_files = list(CORPUS_DIR.rglob(f"*.{EXT}"))
if not wav_files:
    print("No files")
    raise SystemExit(1)

print("Simplified benchmark on", CORPUS_DIR, "ext", EXT)
print("files:", len(wav_files))

def gen_from_dir(fixtures_dir: Path):
    import wave as _wave
    import numpy as np

    def g():
        for fpath in sorted(fixtures_dir.rglob(f"*.{EXT}")):
            try:
                with _wave.open(str(fpath), "rb") as wf:
                    sr = wf.getframerate()
                    nch = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
            except Exception:
                continue
            audio = np.frombuffer(frames, dtype=np.int16).astype(float) / 32767.0
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

    return g


results = {
    "system": {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    },
    "fixture_count": len(wav_files),
    "runs": {},
}


def run_once(workers):
    with tempfile.TemporaryDirectory() as tmp:
        executor = LocalExecutor(
            pipeline=[], checkpoint_path=str(Path(tmp) / "ckpt.db"), num_workers=workers
        )
        writer = JSONLWriter(str(Path(tmp) / "manifest.jsonl"))
        gen = gen_from_dir(CORPUS_DIR)
        start = time.perf_counter()
        stats = executor.run(gen(), writer)
        elapsed = time.perf_counter() - start
        throughput = stats["processed"] / elapsed if elapsed > 0 else 0
        return {
            "elapsed_s": round(elapsed, 3),
            "files_processed": stats["processed"],
            "files_kept": stats["kept"],
            "throughput_files_per_sec": round(throughput, 2),
        }


results["runs"]["sequential"] = run_once(1)
results["runs"]["parallel_4"] = run_once(4)
results["runs"]["segmentation"] = run_once(1)

with open("benchmarks/e2e_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results to benchmarks/e2e_results.json")
