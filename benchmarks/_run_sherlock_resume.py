"""Measure an immediate checkpoint resume of the full Sherlock benchmark."""

import subprocess
import time
from pathlib import Path


command = [
    "audiotrove",
    "--verbose",
    "curate",
    r"C:\Users\palak_uge27\sherlock\adventuresherlockholmes_01_doyle.mp3",
    r".\sherlock-curated-full",
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

started = time.perf_counter()
completed = subprocess.run(command, text=True, capture_output=True, check=False)
wall_time = time.perf_counter() - started
Path("sherlock_resume_run.log").write_text(
    completed.stdout + completed.stderr + f"\nWALL_TIME_SECONDS={wall_time:.2f}\n",
    encoding="utf-8",
)
raise SystemExit(completed.returncode)
