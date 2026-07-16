"""Pre-cache Silero VAD model via torch.hub, retrying on HTTP 429.
Prints 'model cached' when done.
"""

import time
import sys

RETRY_DELAY = 300  # 5 minutes
MAX_RETRIES = 20

for attempt in range(1, MAX_RETRIES + 1):
    try:
        import torch

        torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=True)
        print("model cached")
        sys.exit(0)
    except Exception as e:
        msg = str(e)
        if "HTTP Error 429" in msg or "429" in msg:
            print(
                f"Attempt {attempt}: HTTP 429 encountered, sleeping {RETRY_DELAY}s and retrying..."
            )
            time.sleep(RETRY_DELAY)
            continue
        # Other errors: print and exit non-zero so user can inspect
        print("error", type(e).__name__, msg)
        sys.exit(2)

print("failed to cache model after retries")
sys.exit(1)
