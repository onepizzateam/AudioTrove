import json
from pathlib import Path

import soundfile as sf

corpus = list(Path("benchmarks/LibriSpeech/dev-clean").rglob("*.flac"))
total_input = sum(sf.info(path).duration for path in corpus)
kept_duration = sum(sf.info(path).duration for path in Path("benchmarks/tts_out_w1").rglob("*.wav"))
print(f"corpus: {len(corpus)} files input={total_input:.1f}s ({total_input / 3600:.2f}h)")
print(f"kept audio: {kept_duration:.1f}s ({kept_duration / 3600:.2f}h)")
for workers in (1, 4):
    result = json.load(open(f"benchmarks/run_w{workers}_result.json"))
    wall = result["wall_time_s"]
    print(
        f"workers={workers} kept={result['kept']} filtered={result['filtered']} "
        f"wall={wall}s RTFx={total_input / wall:.1f} "
        f"clips/sec={(result['kept'] + result['filtered']) / wall:.2f}"
    )
