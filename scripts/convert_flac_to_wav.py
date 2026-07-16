import soundfile as sf, os
from pathlib import Path

src = Path("/tmp/LibriSpeech")
dst = Path("/tmp/LibriSpeech_wav")
dst.mkdir(parents=True, exist_ok=True)
count = 0
for root, _, files in os.walk(src):
    for f in files:
        if f.lower().endswith(".flac"):
            srcp = Path(root) / f
            rel = srcp.relative_to(src)
            outp = dst / rel
            outp.parent.mkdir(parents=True, exist_ok=True)
            try:
                data, sr = sf.read(str(srcp))
                sf.write(str(outp.with_suffix(".wav")), data, sr)
                count += 1
                if count % 100 == 0:
                    print("converted", count)
            except Exception as e:
                print("skip", srcp, e)
print("done", count)
