"""
extract_asl_citizen_holistic.py   (v2 pipeline, step 1)

Turns ASL Citizen videos into body-aware landmark sequences (hands + pose,
see holistic_features.py). Needs asl_citizen_raw/ from the earlier setup.
No recording needed.

USAGE (venv active, project folder):
    python extract_asl_citizen_holistic.py --dry-run      # preview words + splits
    python extract_asl_citizen_holistic.py --pilot        # 40 words first (quicker test)
    python extract_asl_citizen_holistic.py                # all words (long)
    python extract_asl_citizen_holistic.py --words BREAKFAST,LUNCH
    python extract_asl_citizen_holistic.py --rebuild      # redo trimming only (fast)

Resumable: finished words are skipped if you stop and rerun.
Output: holistic_data/<split>/<WORD>/NNNN.npy, where <split> (train / val /
test) comes from ASL Citizen's own CSVs. Those splits are by SIGNER, so test
results measure people the model has never seen.
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

import holistic_features as hf

RAW_DIR = "asl_citizen_raw"
FEATURES_DIR = "holistic_raw"      # full-length per-video features
OUT_DIR = "holistic_data"          # trimmed 30-frame sequences
STRIDE = 2                         # use every 2nd frame

PILOT = """HEAR LISTEN THREE SEVEN NINE SCHOOL CITY NO YES AGAIN WANT EAT DRINK HELLO HELP
PLEASE SORRY FRIEND BOOK COMPUTER WATER HOME GO COME GOOD BAD MORE FINISH NAME WHAT WHERE WHO
LIKE LOVE KNOW MOTHER FATHER TODAY TOMORROW FAMILY""".split()

FULL = """
ME YOU HE SHE WE THEY MY YOUR NAME WHAT WHERE WHEN WHY HOW WHO WHICH
GO COME WALK RUN SIT STAND STOP START FINISH WAIT LIKE LOVE NEED WANT HAVE
KNOW THINK UNDERSTAND LEARN TEACH READ WRITE WORK PLAY SLEEP WAKE LIVE
BUY SELL GIVE TAKE MEET SEE LOOK HEAR LISTEN TALK TELL ASK ANSWER CALL SHOW
EAT DRINK WATER FOOD MILK COFFEE BREAD APPLE PIZZA BREAKFAST LUNCH DINNER
HOME HOUSE SCHOOL STORE HOSPITAL CHURCH CITY BATHROOM KITCHEN ROOM
MOTHER FATHER SISTER BROTHER FAMILY BABY BOY GIRL WOMAN MAN FRIEND TEACHER DOCTOR STUDENT
GOOD BAD HAPPY SAD ANGRY TIRED SICK HUNGRY BIG SMALL HOT COLD NEW OLD
TODAY TOMORROW YESTERDAY NOW MORNING NIGHT WEEK MONTH YEAR TIME WEATHER
MORE FINISH ALL SAME DIFFERENT AGAIN ALWAYS NEVER MAYBE
THANKYOU WELCOME EXCUSE FINE
CAR BUS TRAIN PHONE COMPUTER BOOK MONEY CLOTHES SHOE DOG CAT
ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN
RED BLUE GREEN YELLOW BLACK WHITE
HELP PLEASE SORRY HELLO GOODBYE YES NO
""".split()


def load_metadata(raw_dir):
    frames = []
    for path in glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        v = next((cols[c] for c in cols if "video" in c), None)
        g = next((cols[c] for c in cols if "gloss" in c), None)
        if not (v and g):
            continue
        stem = os.path.basename(path).lower()
        split = next((s for s in ("train", "val", "test") if s in stem), None)
        if split is None:
            continue
        frames.append(pd.DataFrame({"video": df[v], "gloss": df[g], "split": split}))
    if not frames:
        raise FileNotFoundError(
            f"No train/val/test CSVs with video+gloss columns under '{raw_dir}'.")
    meta = pd.concat(frames).drop_duplicates(subset=["video"])
    meta["gloss"] = meta["gloss"].astype(str).str.upper().str.strip()
    return meta


_worker = {}


def _init_worker():
    import mediapipe as mp
    _worker["holistic"] = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=0, smooth_landmarks=True,
        enable_segmentation=False, refine_face_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def _extract(path):
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
                res = _worker["holistic"].process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                feats.append(hf.extract_frame_features(res))
            i += 1
        cap.release()
        return np.array(feats, dtype=np.float32) if feats else None
    except Exception:
        return None


def build_split_word(split, word):
    raw_dir = os.path.join(FEATURES_DIR, split, word)
    out_dir = os.path.join(OUT_DIR, split, word)
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.npy")):
        os.remove(f)
    k = 0
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.npy"))):
        seq = hf.trim_and_resample(np.load(f))
        if seq is not None:
            np.save(os.path.join(out_dir, f"{k:04d}.npy"), seq)
            k += 1
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()

    meta = load_metadata(RAW_DIR)
    videos = {os.path.basename(p): p
              for p in glob.glob(os.path.join(RAW_DIR, "**", "*.mp4"), recursive=True)}
    print(f"{len(meta)} labeled clips | splits: "
          + ", ".join(f"{s}={n}" for s, n in meta["split"].value_counts().items()))

    if args.words:
        wanted = [w.strip().upper() for w in args.words.split(",")]
    else:
        wanted = list(dict.fromkeys(PILOT if args.pilot else FULL))

    jobs, missing = [], []
    for w in wanted:
        mask = (meta["gloss"] == w) | meta["gloss"].str.fullmatch(re.escape(w) + r"\d+")
        rows = []
        for v, sp in zip(meta.loc[mask, "video"], meta.loc[mask, "split"]):
            path = videos.get(v) or videos.get(os.path.basename(str(v)))
            if path:
                rows.append((path, sp))
        (jobs if rows else missing).append((w, rows))

    print(f"Words found: {len(jobs)}  | missing: {', '.join(w for w, _ in missing) or 'none'}")
    if args.dry_run:
        for w, rows in jobs[:10]:
            print(f"  {w}: {len(rows)} clips")
        return

    if args.rebuild:
        for n, (w, rows) in enumerate(jobs, 1):
            for sp in ("train", "val", "test"):
                if os.path.isdir(os.path.join(FEATURES_DIR, sp, w)):
                    build_split_word(sp, w)
            print(f"[{n}/{len(jobs)}] {w}: rebuilt")
        return

    os.makedirs(os.path.join(FEATURES_DIR, "_done"), exist_ok=True)
    ctx = mp_proc.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for n, (w, rows) in enumerate(jobs, 1):
            marker = os.path.join(FEATURES_DIR, "_done", w)
            if os.path.exists(marker):
                print(f"[{n}/{len(jobs)}] {w}: already done")
                continue
            raws = pool.map(_extract, [p for p, _ in rows])
            counters = {}
            for (path, sp), raw in zip(rows, raws):
                if raw is None:
                    continue
                d = os.path.join(FEATURES_DIR, sp, w)
                os.makedirs(d, exist_ok=True)
                k = counters.get(sp, 0)
                np.save(os.path.join(d, f"{k:04d}.npy"), raw)
                counters[sp] = k + 1
            kept = sum(build_split_word(sp, w) for sp in counters)
            open(marker, "w").close()
            print(f"[{n}/{len(jobs)}] {w}: {kept}/{len(rows)} usable clips")
            sys.stdout.flush()
    print("\nDone. Next: python train_holistic_model.py")


if __name__ == "__main__":
    main()
