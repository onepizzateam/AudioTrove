import gc
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from audiotrove.executor.local import LocalExecutor  # noqa: E402
from audiotrove.exporters.tts_manifest import TTSManifestExporter  # noqa: E402
from audiotrove.filters.duration import DurationBucketFilter  # noqa: E402
from audiotrove.filters.snr import SNRFilter  # noqa: E402
from audiotrove.filters.vad import SileroVADFilter  # noqa: E402
from audiotrove.io.readers import LocalAudioReader  # noqa: E402
from audiotrove.transformers.silence_trim import SilenceTrimmingTransformer  # noqa: E402

files = sorted(str(p) for p in Path("benchmarks/LibriSpeech/dev-clean").rglob("*.flac"))
out = "benchmarks/tts_out_w1"
t0 = time.time()
kept = filtered = 0
for start in range(0, len(files), 200):
    exporter = TTSManifestExporter(out, export_format=["ljspeech", "f5tts"])
    pipeline = [SileroVADFilter(min_speech_ratio=0.1), SilenceTrimmingTransformer(padding_ms=150), SNRFilter(min_snr_db=20.0), DurationBucketFilter(2.0, 15.0)]
    stats = LocalExecutor(pipeline, checkpoint_path=f"{out}/checkpoint.db", num_workers=1).run(LocalAudioReader(files[start:start + 200], min_duration_seconds=0.0, max_duration_seconds=None), exporter)
    kept += stats["kept"]
    filtered += stats["skipped"]
    del pipeline, exporter
    gc.collect()
result = {"kept": kept, "filtered": filtered, "total_duration_seconds": 0.0, "output_files": [f"{out}/metadata.csv", f"{out}/filelist.txt"], "wall_time_s": round(time.time() - t0, 2)}
with open("benchmarks/run_w1_result.json", "w") as f:
    json.dump(result, f)
print(json.dumps(result, indent=2))
