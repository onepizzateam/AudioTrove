import sys
from pathlib import Path

base = Path(r"C:\tmp\LibriSpeech_wav\dev-clean")
if not base.exists():
    print("missing base dir", base)
    sys.exit(2)

wav = next(base.rglob("*.wav"), None)
if wav is None:
    print("no wav found under", base)
    sys.exit(2)

p = wav
print("path", p)

try:
    import torchaudio

    try:
        w, sr = torchaudio.load(str(p))
        print("torchaudio.load ok", w.shape, sr)
    except Exception as e:
        print("torchaudio.load failed", type(e).__name__, e)
        # Try explicit backends
        try:
            from torchaudio.backend import sox_io_backend

            w, sr = sox_io_backend.load(str(p))
            print("sox_io_backend ok", w.shape, sr)
        except Exception as e2:
            print("sox_io_backend failed", type(e2).__name__, e2)
        try:
            from torchaudio.backend import soundfile_backend

            w, sr = soundfile_backend.load(str(p))
            print("soundfile_backend ok", w.shape, sr)
        except Exception as e3:
            print("soundfile_backend failed", type(e3).__name__, e3)
except Exception as e:
    print("import torchaudio failed", type(e).__name__, e)

# Also test soundfile directly
try:
    import soundfile as sf

    w, sr = sf.read(str(p), dtype="float32")
    print("soundfile.read ok", (w.shape if hasattr(w, "shape") else "scalar"), sr)
except Exception as e:
    print("soundfile.read failed", type(e).__name__, e)
