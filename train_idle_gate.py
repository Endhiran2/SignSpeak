"""
train_idle_gate.py   -  teaches a small "is this a sign?" filter.

It learns from
   not signs : idle_data/   (record_idle.py)
   signs     : user_data/   (your saved sign clips from drill_mode / c / k)
             + a sample of ASL Citizen clips (holistic_data/train)
and saves idle_gate.keras and idle_gate.json. translator_holistic.py then
ignores any movement the gate says is not a sign, BEFORE the word model runs.

Run:  python train_idle_gate.py
      python train_idle_gate.py --dataset-pos 300
"""
import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import holistic_features as hf
from train_holistic_model import augment

rng = np.random.default_rng(3)


def load(pattern):
    X = [np.load(f) for f in sorted(glob.glob(pattern, recursive=True))]
    X = [x for x in X if x.shape == (hf.SEQUENCE_LENGTH, hf.FRAME_FEATURES)]
    return np.array(X, np.float32) if X else np.zeros((0, hf.SEQUENCE_LENGTH, hf.FRAME_FEATURES), np.float32)


def prep(A):
    return np.array([hf.add_motion_features(a) for a in A], np.float32)


def build():
    m = models.Sequential([
        layers.Input(shape=(hf.SEQUENCE_LENGTH, hf.MODEL_FEATURES)),
        layers.GaussianNoise(0.02),
        layers.TimeDistributed(layers.Dense(64, activation="relu")),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.GRU(48)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def split(X, frac=0.2):
    idx = rng.permutation(len(X)); k = max(1, int(len(X) * frac)) if len(X) >= 5 else 0
    return X[idx[k:]], X[idx[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-pos", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()

    neg = load("idle_data/**/*.npy")
    pos_user = load("user_data/**/*.npy")
    allp = load("holistic_data/train/**/*.npy")
    pos_ds = allp[rng.permutation(len(allp))[: a.dataset_pos]] if len(allp) else allp
    print(f"idle clips (not signs): {len(neg)} | your sign clips: {len(pos_user)} | dataset sign clips: {len(pos_ds)}")
    if len(neg) < 20 or len(pos_user) < 20:
        raise SystemExit("Need at least 20 idle clips (python record_idle.py) and 20 of your sign clips.")

    ntr, nte = split(neg); utr, ute = split(pos_user); dtr, dte = split(pos_ds)

    def aug(A, copies):
        out = [A] + [np.array([augment(s) for s in A], np.float32) for _ in range(copies)]
        return np.concatenate(out)

    # weight the classes so webcam clips dominate on both sides (avoids "webcam vs dataset" shortcuts)
    Xn = aug(ntr, 9)
    Xp = np.concatenate([aug(utr, max(1, int(len(Xn) / max(len(utr), 1)) - 1)), aug(dtr, 1)])
    X = prep(np.concatenate([Xn, Xp])); y = np.array([0] * len(Xn) + [1] * len(Xp), np.float32)
    perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
    print(f"training on {len(X)} sequences ({len(Xn)} idle, {len(Xp)} sign)")

    m = build()
    m.fit(X, y, epochs=a.epochs, batch_size=32, validation_split=0.1, verbose=2,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)])

    p_neg = m.predict(prep(nte), verbose=0)[:, 0] if len(nte) else np.array([])
    p_usr = m.predict(prep(ute), verbose=0)[:, 0] if len(ute) else np.array([])
    p_ds = m.predict(prep(dte), verbose=0)[:, 0] if len(dte) else np.array([])
    # choose the threshold: block >= 95% of held-out idle clips, keep as many of your signs as possible
    scores = []
    for t in np.arange(0.1, 0.95, 0.05):
        blocked = float(np.mean(p_neg < t)) if len(p_neg) else 1.0
        kept = float(np.mean(p_usr >= t)) if len(p_usr) else 1.0
        scores.append((round(float(t), 2), (blocked >= 0.95, round(kept, 3))))
    top = max(sc for _, sc in scores)
    tied = [t for t, sc in scores if sc == top]
    thr = float(np.median(tied))          # middle of the best range, not an edge
    print("\nHeld-out results at threshold %.2f:" % thr)
    if len(p_neg): print(f"  idle clips correctly ignored : {np.mean(p_neg < thr):6.1%}  ({len(p_neg)} clips)")
    if len(p_usr): print(f"  your signs correctly kept    : {np.mean(p_usr >= thr):6.1%}  ({len(p_usr)} clips)")
    if len(p_ds):  print(f"  dataset signs correctly kept : {np.mean(p_ds >= thr):6.1%}  ({len(p_ds)} clips)")
    m.save("idle_gate.keras")
    json.dump({"threshold": thr}, open("idle_gate.json", "w"))
    print("\nSaved idle_gate.keras and idle_gate.json. translator_holistic.py will use them automatically.")


if __name__ == "__main__":
    main()
