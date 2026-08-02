import json

import soundfile as sf

violations = []
with open("benchmarks/tts_out_w1/filelist.txt") as filelist:
    for line in filelist:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        path, claimed = parts[0], float(parts[1])
        actual = sf.info(path).duration
        if abs(actual - claimed) > 0.05:
            violations.append(f"MISMATCH {path}: claimed={claimed:.3f} actual={actual:.3f}")
        if actual < 2.0 or actual > 15.0:
            violations.append(f"BOUNDS {path}: {actual:.3f}s")

with open("benchmarks/run_w1_sanity.json", "w") as result:
    json.dump(violations, result)
print(f"violations: {len(violations)}")
for violation in violations[:20]:
    print(violation)
