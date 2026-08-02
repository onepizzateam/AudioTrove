import subprocess
import time
from pathlib import Path


log = Path("sherlock_run.log")
command = [
    "audiotrove",
    "--verbose",
    "curate",
    r"C:\Users\palak_uge27\sherlock",
    "./sherlock-curated",
    "--tts",
    "--segment",
    "--extensions",
    "mp3",
    "--workers",
    "4",
    "--tts-min-duration",
    "2",
    "--tts-max-duration",
    "15",
    "--tts-snr-min",
    "20",
]
t0 = time.time()
with log.open("w", encoding="utf-8") as output:
    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, text=True)
    output.write(f"\nWALL_TIME_SECONDS={time.time() - t0:.2f}\n")
raise SystemExit(result.returncode)
