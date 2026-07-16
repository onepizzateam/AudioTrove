from datasets import load_dataset
import soundfile as sf, os, numpy as np

os.makedirs("/tmp/cv_bench", exist_ok=True)
ds = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation")
for i, row in enumerate(ds):
    if i >= 200:
        break
    arr = np.array(row["audio"]["array"], dtype=np.float32)
    sf.write(f"/tmp/cv_bench/clip_{i:04d}.wav", arr, row["audio"]["sampling_rate"])
print("Done", i + 1)
