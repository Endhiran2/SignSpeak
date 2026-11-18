"""
prepare_wlasl_data.py

Converts videos from the WLASL (Word-Level ASL) dataset into the same
landmark-sequence format produced by collect_sequence_data.py, so you can
train the word/phrase LSTM model (train_sequence_model.py) on WLASL's
vocabulary instead of recording every sign yourself.

===========================================================================
SETUP (do this first, before running this script)
===========================================================================
1. Create a free Kaggle account, then create an API token:
   https://www.kaggle.com/settings -> "Create New Token"
   This downloads kaggle.json. Place it at:
     Linux/Mac:   ~/.kaggle/kaggle.json
     Windows:     C:\\Users\\<you>\\.kaggle\\kaggle.json

2. pip install kaggle

3. Download the pre-processed WLASL dataset (videos already downloaded +
   metadata JSON, ~4.8GB). This uses a community-hosted mirror that avoids
   WLASL's well-known dead-link problem (many original source URLs -
   YouTube/other ASL sites - have gone offline over the years):

     kaggle datasets download -d risangbaskoro/wlasl-processed
     unzip wlasl-processed.zip -d wlasl_raw

   You should end up with:
     wlasl_raw/WLASL_v0.3.json
     wlasl_raw/videos/<video_id>.mp4   (thousands of files)

===========================================================================
WHAT THIS SCRIPT DOES
===========================================================================
- Reads WLASL_v0.3.json (maps each word/"gloss" to a list of video clips,
  each with a bounding box and frame range for where the sign occurs).
- For each word you list in TARGET_GLOSSES below:
    - crops each of its video clips to the labeled bounding box + frame
      range
    - runs MediaPipe over the cropped clip to extract two-hand landmarks
      per frame (same 126-feature format used elsewhere in this project)
    - resamples the sequence to a fixed SEQUENCE_LENGTH (handles WLASL
      clips having different lengths/frame rates than each other)
    - saves it into sequence_data/<WORD>/sample_XXXX.npy
- This is the exact same folder structure collect_sequence_data.py
  produces, so train_sequence_model.py needs no changes, and you can
  freely mix WLASL-derived samples with your own recorded samples in the
  same sequence_data/ folder.

Start with a modest word list (the default below has ~15) to verify the
pipeline works and check accuracy before scaling up to more words - WLASL
has ~2,000 words total, but each additional word means more processing
time and a harder classification problem.

IMPORTANT: WLASL contains no "IDLE"/no-sign clips - it's only positive
examples of words being signed. Without an IDLE class, the real-time app
will try to classify every hand movement as one of your target words. Run
collect_sequence_data.py afterward to record a handful of your own IDLE
clips (hands resting, casual movement) - they'll combine automatically
with the WLASL data since both use the same sequence_data/ folder format.
"""

import json
import os

import cv2
import mediapipe as mp
import numpy as np

WLASL_JSON = "wlasl_raw/WLASL_v0.3.json"
VIDEOS_DIR = "wlasl_raw/videos"
OUTPUT_DIR = "sequence_data"
SEQUENCE_LENGTH = 30

# Edit this to the words you want to train on. Must match WLASL's "gloss"
# field (case-insensitive here, but check WLASL_v0.3.json for exact
# spelling/wording if a word you expect doesn't show up).
TARGET_GLOSSES = [
    "hello", "thanks", "please", "sorry", "yes", "no",
    "help", "want", "eat", "drink", "book", "computer",
    "friend", "school", "again",
]

mp_hands = mp.solutions.hands


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords = coords - wrist
    max_val = np.max(np.linalg.norm(coords, axis=1))
    if max_val > 0:
        coords = coords / max_val
    return coords.flatten()


def extract_two_hand_features(results):
    features = {"Left": np.zeros(63, dtype=np.float32), "Right": np.zeros(63, dtype=np.float32)}
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            features[label] = normalize_landmarks(hand_landmarks).astype(np.float32)
    return np.concatenate([features["Left"], features["Right"]])


def crop_frame(frame, bbox):
    if not bbox or bbox == [0, 0, 0, 0]:
        return frame
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def process_video(video_path, bbox, frame_start, frame_end, hands):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, (frame_start or 0) - 1)
    end = frame_end if frame_end and frame_end > 0 else total_frames

    raw_sequence = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= start:
            cropped = crop_frame(frame, bbox)
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            raw_sequence.append(extract_two_hand_features(result))
        frame_idx += 1
        if end > 0 and frame_idx >= end:
            break

    cap.release()

    if len(raw_sequence) == 0:
        return None

    raw_sequence = np.array(raw_sequence, dtype=np.float32)

    # Resample to a fixed length so every sample matches the LSTM's
    # expected input shape, regardless of the clip's original length/fps.
    indices = np.linspace(0, len(raw_sequence) - 1, SEQUENCE_LENGTH).astype(int)
    return raw_sequence[indices]


def main():
    if not os.path.exists(WLASL_JSON):
        raise FileNotFoundError(
            f"Could not find {WLASL_JSON}. See the setup instructions in "
            f"this script's docstring for how to download WLASL first."
        )

    with open(WLASL_JSON, "r") as f:
        entries = json.load(f)

    gloss_to_entry = {entry["gloss"].lower(): entry for entry in entries}

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for gloss in TARGET_GLOSSES:
        gloss = gloss.lower()
        if gloss not in gloss_to_entry:
            print(f"[skip] '{gloss}' not found in WLASL_v0.3.json - check spelling")
            continue

        label = gloss.upper().replace(" ", "_")
        out_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        instances = gloss_to_entry[gloss]["instances"]
        saved = 0

        for instance in instances:
            video_id = instance["video_id"]
            video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue  # missing/dead-link video in this mirror, skip

            sequence = process_video(
                video_path,
                instance.get("bbox"),
                instance.get("frame_start", 0),
                instance.get("frame_end", -1),
                hands,
            )
            if sequence is None:
                continue

            out_path = os.path.join(out_dir, f"sample_{saved:04d}.npy")
            np.save(out_path, sequence)
            saved += 1

        print(f"'{gloss}': saved {saved}/{len(instances)} sequences -> {out_dir}")

    print("\nDone. Next steps:")
    print("  1. (Recommended) Run collect_sequence_data.py to record a few")
    print("     'IDLE' clips of yourself, so the model can tell signing")
    print("     apart from pauses/resting hands.")
    print("  2. Run train_sequence_model.py to train on everything in")
    print("     sequence_data/.")


if __name__ == "__main__":
    main()
