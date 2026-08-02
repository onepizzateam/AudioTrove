from pathlib import Path

import requests


BASE = "https://archive.org/download/adventures_sherlockholmes_1007_librivox"
OUTPUT = Path(r"C:\Users\palak_uge27\sherlock")
OUTPUT.mkdir(parents=True, exist_ok=True)

for chapter in range(1, 13):
    name = f"adventuresherlockholmes_{chapter:02d}_doyle.mp3"
    target = OUTPUT / name
    if target.exists() and target.stat().st_size > 0:
        print(f"exists {name}", flush=True)
        continue
    url = f"{BASE}/{name}"
    print(f"downloading {name}", flush=True)
    with requests.get(url, stream=True, timeout=(60, 120)) as response:
        response.raise_for_status()
        with target.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    print(f"complete {name} {target.stat().st_size}", flush=True)
