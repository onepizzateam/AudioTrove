"""Run a demo `audiotrove curate` invocation using local fixtures and capture output.
Usage: python scripts/run_demo_capture.py
Writes `packaging/curate_demo_output.txt` with the captured stdout.
"""

import sys
from pathlib import Path
import json

OUT = Path("packaging")
OUT.mkdir(exist_ok=True)
OUT_FILE = OUT / "curate_demo_output.txt"


def _make_generator_from_dir(fixtures_dir: Path):
    import wave as _wave
    from audiotrove.document import AudioDocument
    from audiotrove.utils.hashing import make_doc_id
    import numpy as np

    def gen():
        for fpath in sorted(fixtures_dir.glob("*.wav")):
            try:
                with _wave.open(str(fpath), "rb") as wf:
                    sr = wf.getframerate()
                    nch = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
            except Exception:
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


def main():
    from click.testing import CliRunner
    from audiotrove.cli import main as cli
    from audiotrove.io import readers as readers_mod

    fixtures = Path("tests/fixtures")
    if not fixtures.exists():
        print("No fixtures found; run scripts/generate_fixtures.py first")
        sys.exit(1)

    # Replace LocalAudioReader with a simple generator-based reader that uses wave
    class FakeReader:
        def __init__(self, patterns, min_duration_seconds=0.0, max_duration_seconds=None):
            self._gen = _make_generator_from_dir(fixtures)()

        def __iter__(self):
            return self._gen

    readers_mod.LocalAudioReader = FakeReader

    runner = CliRunner()
    out_dir = Path("demo_out")
    if out_dir.exists():
        # remove old manifest if present
        try:
            for f in out_dir.glob("*"):
                f.unlink()
        except Exception:
            pass
    args = [
        "curate",
        str(fixtures),
        str(out_dir),
        "--workers",
        "1",
        "--snr-min",
        "0",
        "--vad-threshold",
        "0.0",
    ]
    result = runner.invoke(cli.cli, args)

    OUT_FILE.write_text(result.output, encoding="utf-8")
    print(result.output)
    print("\nSaved demo output to", OUT_FILE)


if __name__ == "__main__":
    main()
