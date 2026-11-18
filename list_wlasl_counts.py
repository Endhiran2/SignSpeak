"""
list_wlasl_counts.py

Utility to help pick better TARGET_GLOSSES for prepare_wlasl_data.py.

Rather than guessing which everyday words WLASL has good coverage for, this
scans WLASL_v0.3.json and checks, for every word (gloss), how many of its
listed video instances actually exist as downloaded .mp4 files in
wlasl_raw/videos/ (accounting for the missing/dead-link videos in any given
mirror of the dataset). It then prints the words with the most *actually
usable* samples, sorted highest first.

Words near the top of this list are much safer choices for TARGET_GLOSSES
than picking common English words and hoping WLASL happens to have good
coverage for them - some ordinary words (like "hello") are surprisingly
rare in WLASL, while other, less obvious words have far more examples.

Usage:
    python list_wlasl_counts.py            # prints top 100 by default
    python list_wlasl_counts.py 200        # prints top 200 instead
"""

import json
import os
import sys

WLASL_JSON = "wlasl_raw/WLASL_v0.3.json"
VIDEOS_DIR = "wlasl_raw/videos"


def main():
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    if not os.path.exists(WLASL_JSON):
        raise FileNotFoundError(f"Could not find {WLASL_JSON} - download/unzip WLASL first.")

    with open(WLASL_JSON, "r") as f:
        entries = json.load(f)

    counts = []
    for entry in entries:
        gloss = entry["gloss"]
        instances = entry["instances"]
        available = sum(
            1 for inst in instances
            if os.path.exists(os.path.join(VIDEOS_DIR, f"{inst['video_id']}.mp4"))
        )
        counts.append((gloss, available, len(instances)))

    counts.sort(key=lambda x: x[1], reverse=True)

    print(f"{'GLOSS':<20} {'AVAILABLE':>10} {'TOTAL LISTED':>14}")
    print("-" * 46)
    for gloss, available, total in counts[:top_n]:
        print(f"{gloss:<20} {available:>10} {total:>14}")

    print(f"\nShowing top {top_n} of {len(counts)} total words.")
    print("Tip: copy whichever gloss names you want into TARGET_GLOSSES in")
    print("prepare_wlasl_data.py (lowercase, matching exactly as shown above).")


if __name__ == "__main__":
    main()
