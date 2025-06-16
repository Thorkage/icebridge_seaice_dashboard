#!/usr/bin/env python3
import os, requests

DATA_DIR   = "/tmp/icebridge"
ZENODO_API = "https://zenodo.org/api/records/15672481"

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    print("⏬ Fetching Zenodo metadata…")
    r = requests.get(ZENODO_API)
    r.raise_for_status()
    meta = r.json()

    for file_meta in meta.get("files", []):
        fname = file_meta["key"]
        links = file_meta.get("links", {})
        url   = links.get("download") or links.get("self")
        if not url:
            print(f"⚠️  No link for {fname}, skipping")
            continue

        local = os.path.join(DATA_DIR, fname)
        if os.path.exists(local):
            continue

        print(f"  ↓ {fname}")
        with requests.get(url, stream=True) as resp:
            resp.raise_for_status()
            with open(local, "wb") as f:
                for chunk in resp.iter_content(1<<20):
                    f.write(chunk)
    print("✅ Download complete.")