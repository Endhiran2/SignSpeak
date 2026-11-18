"""
kaggle_convert.py  -  step 2: turn the downloaded Kaggle landmark files into
the SAME 232-feature format our model trains on (uses holistic_features.py,
so the numbers are computed identically).

Clips go into holistic_data/train/<WORD>/kg_*.npy  (TRAIN ONLY, never val/test,
so the ASL Citizen test score stays comparable).

The Kaggle landmarks may be mirrored relative to ours, so the script checks
both orientations with your current model and keeps the better one.

Run: python kaggle_convert.py                 (auto-detect mirroring)
     python kaggle_convert.py --mirror yes|no (force)
"""
import argparse, glob, os, re, sys
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
import holistic_features as hf

RAW = "kaggle_raw"
OUT = os.path.join("holistic_data", "train")
OFF = {"face": 0, "left_hand": 468, "pose": 489, "right_hand": 522}
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14)]


def load_array(path):
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frames = np.sort(df["frame"].unique())
    pos = {f: i for i, f in enumerate(frames)}
    arr = np.full((len(frames), 543, 3), np.nan, np.float32)
    fi = df["frame"].map(pos).to_numpy()
    li = df["type"].map(OFF).to_numpy() + df["landmark_index"].to_numpy()
    arr[fi, li] = df[["x", "y", "z"]].to_numpy(np.float32)
    return arr


def variant(arr, mode):
    """no | yes (flip x + swap hands/pose pairs) | swap (swap only) | flipx (flip x only)"""
    if mode == "no":
        return arr
    m = arr.copy()
    if mode in ("yes", "flipx"):
        m[:, :, 0] = 1.0 - m[:, :, 0]
    if mode in ("yes", "swap"):
        lh, rh = m[:, 468:489].copy(), m[:, 522:543].copy()
        m[:, 468:489], m[:, 522:543] = rh, lh
        pose = m[:, 489:522].copy()
        for a_, b_ in POSE_PAIRS:
            pose[:, [a_, b_]] = pose[:, [b_, a_]]
        m[:, 489:522] = pose
    return m


def _lm(block):
    if np.isnan(block).any():
        return None
    return NS(landmark=[NS(x=float(p[0]), y=float(p[1]), z=float(p[2]), visibility=1.0) for p in block])


def features(arr):
    seq = []
    for f in arr:
        res = NS(pose_landmarks=_lm(f[489:522]), left_hand_landmarks=_lm(f[468:489]),
                 right_hand_landmarks=_lm(f[522:543]))
        seq.append(hf.extract_frame_features(res))
    return hf.trim_and_resample(np.array(seq, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", choices=["auto", "no", "yes", "swap", "flipx"], default="auto")
    a = ap.parse_args()
    sel = pd.read_csv(os.path.join(RAW, "_selection.csv"))
    sel["file"] = sel["path"].map(os.path.basename)
    sel = sel[sel["file"].map(lambda f: os.path.exists(os.path.join(RAW, f)))].reset_index(drop=True)
    print(f"{len(sel)} clips downloaded")

    arrays = {}
    def get(i):
        if i not in arrays:
            arrays[i] = load_array(os.path.join(RAW, sel.loc[i, "file"]))
        return arrays[i]

    mode = a.mirror
    if mode == "auto":
        try:
            import pickle, tensorflow as tf
            model = tf.keras.models.load_model("sign_holistic_model.keras")
            enc = pickle.load(open("holistic_label_encoder.pkl", "rb"))
            known = set(enc.classes_)
            sub = sel[sel["word"].isin(known)].groupby("word").head(4).reset_index(drop=True)
            print(f"Checking orientation on {len(sub)} clips with the current model ...")
            score = {}
            for name in ("no", "yes", "swap", "flipx"):
                X, y = [], []
                for _, r in sub.iterrows():
                    s = features(variant(load_array(os.path.join(RAW, r["file"])), name))
                    if s is not None:
                        X.append(hf.add_motion_features(s)); y.append(r["word"])
                p = enc.inverse_transform(np.argmax(model.predict(np.array(X), verbose=0), axis=1))
                score[name] = float(np.mean(p == np.array(y)))
                print(f"  mirror={name}: zero-shot accuracy {score[name]:.2%}")
            mode = max(score, key=score.get)
            if score[mode] < 0.25:
                print("WARNING: all four orientations score low; the Kaggle format may not match ours. "
                      "Treat the merge with caution and compare test accuracy before using it.")
        except Exception as e:
            print("Could not auto-check orientation (", e, ") - using mirror=no")
            mode = "no"
    print("Using mirror =", mode)

    n = skipped = 0
    for i, r in sel.iterrows():
        arr = get(i)
        arr = variant(arr, mode)
        s = features(arr)
        arrays.pop(i, None)
        if s is None:
            skipped += 1; continue
        d = os.path.join(OUT, r["word"]); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "kg_" + os.path.splitext(r["file"])[0] + ".npy"), s)
        n += 1
        if n % 500 == 0:
            print(f"  {n} converted")
    print(f"Converted {n} clips ({skipped} skipped: too few hand frames) into {OUT}/")
    print("Next: back up your model, then  python train_holistic_model.py")


if __name__ == "__main__":
    main()
