"""Quick check to verify VAD backend used by workers.
Generates a manifest with `vad_backend` metadata for inspection.
"""

import argparse
from pathlib import Path
import itertools


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--ext", default="wav")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from audiotrove.filters.vad import SileroVADFilter
    from audiotrove.filters.snr import SNRFilter
    from audiotrove.executor.local import LocalExecutor
    from audiotrove.io.writers import JSONLWriter
    from audiotrove.io.readers import LocalAudioReader

    corpus = Path(args.corpus_dir)
    reader = LocalAudioReader([str(corpus / f"**/*.{args.ext}")], min_duration_seconds=0.0)

    pipeline = [SileroVADFilter(min_speech_ratio=0.1), SNRFilter(min_snr_db=0.0)]
    executor = LocalExecutor(
        pipeline=pipeline,
        checkpoint_path=str(Path("benchmarks") / "vad_check_ckpt.db"),
        num_workers=args.workers,
    )

    out = Path("benchmarks")
    out.mkdir(exist_ok=True)
    writer = JSONLWriter(str(out / "vad_check_manifest.jsonl"))

    # Limit reader generator
    gen = reader

    def limited():
        for i, doc in enumerate(reader):
            if i >= args.limit:
                break
            yield doc

    stats = executor.run(limited(), writer)
    print("stats", stats)


if __name__ == "__main__":
    run()
