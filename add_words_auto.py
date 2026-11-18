"""
add_words_auto.py

Adds new words to your vocabulary from WLASL with no manual word-picking.

It ranks every WLASL word by how many video clips you actually have on disk,
skips words you already have in sequence_data/, and processes the top N
into landmark sequences (same format as prepare_wlasl_data.py).

Usage (run from your project folder, with the venv active):
    python add_words_auto.py          # adds 20 new words
    python add_words_auto.py 15       # adds 15 new words

Needs prepare_wlasl_data.py in the same folder (it reuses its processing code).
"""

import json
import os
import sys

import numpy as np

import prepare_wlasl_data as p

NUM_NEW_WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MIN_AVAILABLE = 8  # only consider words with at least this many usable clips


def main():
    if not os.path.exists(p.WLASL_JSON):
        raise FileNotFoundError(f"Could not find {p.WLASL_JSON}")

    with open(p.WLASL_JSON, "r") as f:
        entries = json.load(f)

    existing = set()
    if os.path.isdir(p.OUTPUT_DIR):
        existing = {
            d for d in os.listdir(p.OUTPUT_DIR)
            if os.path.isdir(os.path.join(p.OUTPUT_DIR, d))
        }

    candidates = []
    for entry in entries:
        gloss = entry["gloss"]
        label = gloss.upper().replace(" ", "_")
        if label in existing:
            continue
        usable = [
            inst for inst in entry["instances"]
            if os.path.exists(os.path.join(p.VIDEOS_DIR, f"{inst['video_id']}.mp4"))
        ]
        if len(usable) >= MIN_AVAILABLE:
            candidates.append((gloss, label, usable))

    candidates.sort(key=lambda c: len(c[2]), reverse=True)
    chosen = candidates[:NUM_NEW_WORDS]

    if not chosen:
        print("No new words found with enough usable clips. Try lowering MIN_AVAILABLE.")
        return

    print(f"Adding {len(chosen)} new words:")
    for gloss, label, usable in chosen:
        print(f"  {label}: {len(usable)} clips available")
    print()

    hands = p.mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    for gloss, label, usable in chosen:
        out_dir = os.path.join(p.OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        saved = 0
        for inst in usable:
            video_path = os.path.join(p.VIDEOS_DIR, f"{inst['video_id']}.mp4")
            seq = p.process_video(
                video_path,
                inst.get("bbox"),
                inst.get("frame_start", 0),
                inst.get("frame_end", -1),
                hands,
            )
            if seq is None:
                continue
            np.save(os.path.join(out_dir, f"sample_{saved:04d}.npy"), seq)
            saved += 1
        print(f"{label}: saved {saved}/{len(usable)} sequences")

    print("\nDone. Now run: python train_sequence_model.py")


if __name__ == "__main__":
    main()
