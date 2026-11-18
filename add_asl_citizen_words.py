"""
add_asl_citizen_words.py

Adds many new words (and extra clips for existing words) from ASL Citizen:
~84,000 videos of 2,731 ASL signs, ~30 clips per sign, recorded by 50+
different signers on ordinary cameras - much closer to your webcam setup
than WLASL, and with enough clips per word to actually train on.

SETUP (one time, from your project folder, venv active):
    wget https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip
    unzip -q ASL_Citizen.zip -d asl_citizen_raw
  (check you have enough free disk first:  df -h .)

USAGE:
    python add_asl_citizen_words.py --dry-run     # shows which words exist, no processing
    python add_asl_citizen_words.py               # processes all found words (long run)
    python add_asl_citizen_words.py 40            # only the first 40 words of the list
    python add_asl_citizen_words.py --words DOG,CAT,HOUSE   # your own words

It stores each video's full per-frame landmarks in asl_citizen_features/, then
TRIMS every clip to the frames where hands are actually visible (dropping the
signer raising/lowering their hands) and resamples to 30 frames. Use
    python add_asl_citizen_words.py --rebuild
to redo only the trimming/resampling from the stored landmarks (fast, no
video processing).

It is resumable: if you stop it (Ctrl+C) and run it again, finished words
are skipped. Samples are saved as sequence_data/<WORD>/ac_XXXX.npy next to
your existing samples, so everything mixes in train_sequence_model.py.
"""

import argparse
import glob
import multiprocessing as mp_proc
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

RAW_DIR = "asl_citizen_raw"
OUTPUT_DIR = "sequence_data"
FEATURES_DIR = "asl_citizen_features"   # full-length landmarks per video
SEQUENCE_LENGTH = 30
STRIDE = 2                               # process every 2nd frame (2x faster)
MIN_ACTIVE_FRAMES = 6
MAX_CLIPS_PER_WORD = 30

# Everyday words useful for building sentences. Words ASL Citizen doesn't
# have under this exact spelling are reported and skipped.
WORD_LIST = """
ME YOU HE SHE WE THEY MY YOUR NAME WHAT WHERE WHEN WHY HOW WHO WHICH
GO COME WALK RUN SIT STAND STOP START FINISH WAIT LIKE LOVE NEED WANT HAVE
KNOW THINK UNDERSTAND LEARN TEACH READ WRITE WORK PLAY SLEEP WAKE LIVE
BUY SELL GIVE TAKE MEET SEE LOOK HEAR TALK TELL ASK ANSWER CALL
EAT DRINK WATER FOOD MILK COFFEE BREAD APPLE PIZZA
HOME HOUSE SCHOOL STORE HOSPITAL CHURCH CITY BATHROOM KITCHEN ROOM
MOTHER FATHER SISTER BROTHER FAMILY BABY BOY GIRL WOMAN MAN FRIEND TEACHER DOCTOR STUDENT
GOOD BAD HAPPY SAD ANGRY TIRED SICK HUNGRY BIG SMALL HOT COLD NEW OLD
TODAY TOMORROW YESTERDAY NOW MORNING NIGHT WEEK MONTH YEAR TIME
MORE FINISH ALL SAME DIFFERENT AGAIN ALWAYS NEVER MAYBE
THANKYOU WELCOME EXCUSE FINE LISTEN SHOW WEATHER
CAR BUS TRAIN PHONE COMPUTER BOOK MONEY CLOTHES SHOE DOG CAT
ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN
RED BLUE GREEN YELLOW BLACK WHITE
HELP PLEASE SORRY HELLO GOODBYE YES NO
""".split()


def find_metadata(raw_dir):
    csvs = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    frames = []
    for path in csvs:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        vcol = next((cols[c] for c in cols if "video" in c), None)
        gcol = next((cols[c] for c in cols if "gloss" in c), None)
        if vcol and gcol:
            frames.append(pd.DataFrame({"video": df[vcol], "gloss": df[gcol]}))
    if not frames:
        raise FileNotFoundError(
            f"No metadata CSV with video + gloss columns found under '{raw_dir}'. "
            "Did you unzip ASL_Citizen.zip into that folder?"
        )
    meta = pd.concat(frames).drop_duplicates()
    meta["gloss"] = meta["gloss"].astype(str).str.upper().str.strip()
    return meta


def index_videos(raw_dir):
    return {os.path.basename(p): p
            for p in glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)}


_worker = {}


def _init_worker():
    import prepare_wlasl_data as p
    _worker["p"] = p
    _worker["hands"] = p.mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )


def _extract_raw(path):
    """Full-length per-frame landmarks (T, 126) for one video, untrimmed."""
    p, hands = _worker["p"], _worker["hands"]
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        feats, i = [], 0
        while True:
            if not cap.grab():
                break
            if i % STRIDE == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feats.append(p.extract_two_hand_features(hands.process(rgb)))
            i += 1
        cap.release()
        return np.array(feats, dtype=np.float32) if feats else None
    except Exception:
        return None


def trim_and_resample(raw):
    """Keep only the span from the first to the last frame where a hand is
    visible (plus a small margin), then resample to SEQUENCE_LENGTH."""
    present = np.any(raw != 0, axis=1)
    idx = np.where(present)[0]
    if len(idx) < MIN_ACTIVE_FRAMES:
        return None
    start, end = max(0, idx[0] - 2), min(len(raw) - 1, idx[-1] + 2)
    seg = raw[start:end + 1]
    pick = np.linspace(0, len(seg) - 1, SEQUENCE_LENGTH).astype(int)
    return seg[pick]


def build_word(word):
    """(Re)build sequence_data/<WORD>/ac_*.npy from stored raw landmarks."""
    label = word.replace(" ", "_")
    raw_dir = os.path.join(FEATURES_DIR, label)
    out_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "ac_*.npy")):
        os.remove(f)
    saved = 0
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.npy"))):
        seq = trim_and_resample(np.load(f))
        if seq is not None:
            np.save(os.path.join(out_dir, f"ac_{saved:04d}.npy"), seq)
            saved += 1
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("count", nargs="?", type=int, default=None)
    ap.add_argument("--words", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()

    meta = find_metadata(RAW_DIR)
    videos = index_videos(RAW_DIR)
    print(f"Found {len(meta)} labeled clips, {len(videos)} video files, "
          f"{meta['gloss'].nunique()} distinct signs.")

    wanted = [w.strip().upper() for w in args.words.split(",")] if args.words else list(dict.fromkeys(WORD_LIST))
    available, missing = [], []
    for w in wanted:
        # ASL Citizen often names variants WORD1, WORD2, ... (different ways
        # of signing the same word). Merge the exact gloss and all variants.
        mask = (meta["gloss"] == w) | meta["gloss"].str.fullmatch(re.escape(w) + r"\d+")
        clips = []
        for v in meta.loc[mask, "video"]:
            path = videos.get(v) or videos.get(os.path.basename(str(v)))
            if path:
                clips.append(path)
        (available if clips else missing).append((w, clips))
    if args.count:
        available = available[:args.count]

    print(f"\nWords found: {len(available)}")
    print(" ", ", ".join(f"{w}({len(c)})" for w, c in available))
    if missing:
        print(f"\nNot found under this spelling ({len(missing)}): "
              + ", ".join(w for w, _ in missing))
        print("(check spelling with:  grep -i <word> on the CSV, or pass --words)")
    if args.dry_run:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.rebuild:
        for n, (word, _) in enumerate(available, 1):
            if os.path.isdir(os.path.join(FEATURES_DIR, word)):
                print(f"[{n}/{len(available)}] {word}: rebuilt {build_word(word)} sequences")
        print("\nDone. Now run: python train_sequence_model.py")
        return

    ctx = mp_proc.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for n, (word, clips) in enumerate(available, 1):
            raw_dir = os.path.join(FEATURES_DIR, word.replace(" ", "_"))
            os.makedirs(raw_dir, exist_ok=True)
            target = min(MAX_CLIPS_PER_WORD, len(clips))
            have = len(glob.glob(os.path.join(raw_dir, "*.npy")))
            if have < target:
                for f in glob.glob(os.path.join(raw_dir, "*.npy")):
                    os.remove(f)
                raws = pool.map(_extract_raw, clips[:MAX_CLIPS_PER_WORD])
                k = 0
                for r in raws:
                    if r is not None:
                        np.save(os.path.join(raw_dir, f"{k:04d}.npy"), r)
                        k += 1
            saved = build_word(word)
            print(f"[{n}/{len(available)}] {word}: {saved}/{target} usable clips")
            sys.stdout.flush()

    print("\nDone. Now run: python train_sequence_model.py")


if __name__ == "__main__":
    main()
