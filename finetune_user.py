"""
finetune_user.py   -  teach the model YOUR signing from the clips you saved
with the 'c' / 'k' keys in translator_holistic.py (they live in user_data/).

It starts from the ORIGINAL model (sign_holistic_model.keras) every time, so
you can re-run it whenever you have saved more clips. Your clips are
augmented and mixed with a sample of the original training data so the model
does not forget the other words. The result is saved as
sign_holistic_model_user.keras, which translator_holistic.py then uses
automatically (start it with --base to compare against the original).

It prints accuracy on (a) a few of YOUR clips held out, and (b) the ASL Citizen
test set, before and after, so you can see whether it helped and whether it
hurt anything. Takes a few minutes on CPU.

Run:  python finetune_user.py
      python finetune_user.py --epochs 12 --user-copies 10
"""
import argparse
import glob
import os
import pickle

import numpy as np
import tensorflow as tf

import holistic_features as hf
from train_holistic_model import augment, load_split

MODEL_FILE = "sign_holistic_model.keras"
ENCODER_FILE = "holistic_label_encoder.pkl"
USER_MODEL_FILE = "sign_holistic_model_user.keras"
USER_DIR = "user_data"


def load_user(classes):
    X, y, skipped = [], [], set()
    for wdir in sorted(glob.glob(os.path.join(USER_DIR, "*"))):
        w = os.path.basename(wdir)
        if w not in classes:
            skipped.add(w)
            continue
        for f in sorted(glob.glob(os.path.join(wdir, "*.npy"))):
            a = np.load(f)
            if a.shape == (hf.SEQUENCE_LENGTH, hf.FRAME_FEATURES):
                X.append(a); y.append(classes.index(w))
    return np.array(X, np.float32), np.array(y), skipped


def prep(A):
    return np.array([hf.add_motion_features(a) for a in A], dtype=np.float32)


def acc(model, X, y):
    if len(X) == 0:
        return float("nan")
    return float(np.mean(np.argmax(model.predict(prep(X), verbose=0), axis=1) == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--user-copies", type=int, default=8, help="augmented copies per clip of yours")
    ap.add_argument("--replay", type=int, default=3000, help="original clips mixed in")
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()

    model = tf.keras.models.load_model(MODEL_FILE)
    classes = list(pickle.load(open(ENCODER_FILE, "rb")).classes_)
    n = len(classes)

    Xu, yu, skipped = load_user(classes)
    if skipped:
        print("Ignored folders not in the vocabulary:", ", ".join(sorted(skipped)))
    if len(Xu) < 5:
        raise SystemExit(f"Only {len(Xu)} clips in {USER_DIR}/ - save more with 'c' / 'k' first.")
    counts = {classes[i]: int(np.sum(yu == i)) for i in np.unique(yu)}
    print(f"{len(Xu)} clips of yours across {len(counts)} words:",
          ", ".join(f"{w} {c}" for w, c in sorted(counts.items(), key=lambda t: -t[1])))

    # hold out ~1 in 5 of your clips (only if you have enough) to measure the effect honestly
    rng = np.random.default_rng(1)
    hold = np.zeros(len(Xu), bool)
    if len(Xu) >= 25:
        hold[rng.permutation(len(Xu))[: len(Xu) // 5]] = True
    Xh, yh = Xu[hold], yu[hold]
    Xt, yt = Xu[~hold], yu[~hold]
    print(f"Training on {len(Xt)} of your clips, holding out {len(Xh)} to test.")

    Xte, yte_s = load_split("test")
    yte = np.array([classes.index(w) for w in yte_s]) if len(Xte) else np.array([])
    before_test, before_user = acc(model, Xte, yte), acc(model, Xh, yh)

    # training mix: your clips (+augmented copies) and a replay sample of the original data
    Xa, ya = [Xt], [yt]
    for _ in range(a.user_copies):
        Xa.append(np.array([augment(s) for s in Xt], np.float32)); ya.append(yt)
    Xo, yo_s = load_split("train")
    if len(Xo):
        pick = rng.permutation(len(Xo))[: a.replay]
        Xo, yo = Xo[pick], np.array([classes.index(w) for w in yo_s[pick]])
        Xa += [Xo, np.array([augment(s) for s in Xo], np.float32)]; ya += [yo, yo]
    X, y = prep(np.concatenate(Xa)), np.concatenate(ya)
    perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
    print(f"Fine-tuning on {len(X)} sequences for {a.epochs} epochs ...")

    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])
    model.fit(X, tf.keras.utils.to_categorical(y, n), epochs=a.epochs, batch_size=32, verbose=2)

    after_test, after_user = acc(model, Xte, yte), acc(model, Xh, yh)
    print("\n                         before    after")
    if len(Xh):
        print(f"  your held-out clips   {before_user:7.1%}  {after_user:7.1%}   ({len(Xh)} clips)")
    if len(Xte):
        print(f"  ASL Citizen test      {before_test:7.1%}  {after_test:7.1%}   ({len(Xte)} clips)")
        if after_test < before_test - 0.03:
            print("  WARNING: general accuracy dropped by more than 3 points - try --replay 5000 or fewer epochs.")
    model.save(USER_MODEL_FILE)
    print(f"\nSaved {USER_MODEL_FILE}. translator_holistic.py will use it automatically "
          f"(run it with --base to compare with the original).")


if __name__ == "__main__":
    main()
