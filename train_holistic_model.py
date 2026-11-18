"""
train_holistic_model.py   (v2 pipeline, step 2)

Trains on holistic_data/ (made by extract_asl_citizen_holistic.py).

HONEST EVALUATION: ASL Citizen's train / val / test splits are by SIGNER.
Training uses only 'train' (+ augmentation), early stopping uses 'val', and
the reported accuracy is on 'test' - people the model never saw. That is the
number that predicts how it behaves on you.

Outputs: sign_holistic_model.keras, holistic_label_encoder.pkl,
         holistic_curves.png, holistic_results.xlsx
"""

import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

import holistic_features as hf

DATA_DIR = "holistic_data"
MODEL_FILE = "sign_holistic_model.keras"
ENCODER_FILE = "holistic_label_encoder.pkl"
CURVES_FILE = "holistic_curves.png"
EXCEL_FILE = "holistic_results.xlsx"

AUG_COPIES = 3
rng = np.random.default_rng(0)


# ------------------------------------------------------------ augmentation
def augment(seq):
    """seq: (30, 232). Returns a randomly perturbed copy."""
    s = seq.copy()
    T = len(s)

    # mirror (left-handed signers / mirrored videos): flip x, swap hands
    if rng.random() < 0.3:
        pose = s[:, hf.POSE].reshape(T, 11, 2).copy()
        pose[:, :, 0] *= -1
        pose = pose[:, hf.POSE_SWAP]
        ls, rs = s[:, hf.L_SHAPE].reshape(T, 21, 3).copy(), s[:, hf.R_SHAPE].reshape(T, 21, 3).copy()
        ll, rl = s[:, hf.L_LOC].reshape(T, 21, 2).copy(), s[:, hf.R_LOC].reshape(T, 21, 2).copy()
        for a in (ls, rs):
            a[:, :, 0] *= -1
        for a in (ll, rl):
            a[:, :, 0] *= -1
        s[:, hf.POSE] = pose.reshape(T, 22)
        s[:, hf.L_SHAPE], s[:, hf.R_SHAPE] = rs.reshape(T, 63), ls.reshape(T, 63)
        s[:, hf.L_LOC], s[:, hf.R_LOC] = rl.reshape(T, 42), ll.reshape(T, 42)

    # body-relative location: small global shift/scale (different framing/body size)
    shift = rng.normal(0, 0.08, size=2).astype(np.float32)
    scale = rng.uniform(0.9, 1.1)
    pp = np.any(s[:, hf.POSE] != 0, axis=1)
    s[pp, hf.POSE] = (s[pp][:, hf.POSE].reshape(-1, 11, 2) * scale + shift).reshape(-1, 22)
    for shape_b, loc_b in ((hf.L_SHAPE, hf.L_LOC), (hf.R_SHAPE, hf.R_LOC)):
        present = np.any(s[:, shape_b] != 0, axis=1)
        if not present.any():
            continue
        # hand location
        loc = s[present][:, loc_b].reshape(-1, 21, 2) * scale + shift
        s[np.where(present)[0][:, None], np.arange(loc_b.start, loc_b.stop)] = loc.reshape(-1, 42)
        # hand shape: rotation + scale + noise
        pts = s[present][:, shape_b].reshape(-1, 21, 3)
        th = np.deg2rad(rng.uniform(-15, 15))
        c, sn = np.cos(th), np.sin(th)
        x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
        pts[:, :, 0], pts[:, :, 1] = c * x - sn * y, sn * x + c * y
        pts = pts * rng.uniform(0.9, 1.1) + rng.normal(0, 0.01, pts.shape)
        s[np.where(present)[0][:, None], np.arange(shape_b.start, shape_b.stop)] = pts.reshape(-1, 63)

    # speed change
    f = rng.uniform(0.8, 1.25)
    idx = np.clip(np.round(np.arange(T) * f), 0, T - 1).astype(int)
    return s[idx].astype(np.float32)


# ----------------------------------------------------------------- loading
def load_split(split):
    X, y = [], []
    for wdir in sorted(glob.glob(os.path.join(DATA_DIR, split, "*"))):
        label = os.path.basename(wdir)
        for f in sorted(glob.glob(os.path.join(wdir, "*.npy"))):
            X.append(np.load(f))
            y.append(label)
    return (np.array(X, dtype=np.float32) if X else np.zeros((0, 30, hf.FRAME_FEATURES), np.float32),
            np.array(y))


def build_model(num_classes):
    m = models.Sequential([
        layers.Input(shape=(hf.SEQUENCE_LENGTH, hf.MODEL_FEATURES)),
        layers.GaussianNoise(0.02),
        layers.TimeDistributed(layers.Dense(192, activation="relu")),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.4),
        layers.LSTM(128),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")])
    return m


def main():
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    if len(Xtr) == 0:
        raise SystemExit("No data in holistic_data/train. Run extract_asl_citizen_holistic.py first.")

    signer_split = len(Xva) > 0 and len(Xte) > 0
    if not signer_split:
        print("WARNING: no val/test splits found - falling back to a RANDOM split "
              "(this is NOT signer-independent and will look better than reality).")
        X = Xtr
        y = ytr
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.12, stratify=ytr, random_state=42)

    enc = LabelEncoder().fit(np.concatenate([ytr, yva, yte]))
    classes = list(enc.classes_)
    n = len(classes)
    print(f"{n} words | real clips: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)} "
          f"({'signer-independent splits' if signer_split else 'random split'})")

    Xaug, yaug = [], []
    for s, l in zip(Xtr, enc.transform(ytr)):
        for _ in range(AUG_COPIES):
            Xaug.append(augment(s))
            yaug.append(l)
    Xtrain = np.concatenate([Xtr, np.array(Xaug, dtype=np.float32)])
    ytrain = np.concatenate([enc.transform(ytr), np.array(yaug)])
    print(f"Training sequences: {len(Xtrain)} ({len(Xtr)} real + {len(Xaug)} augmented)")

    prep = lambda A: np.array([hf.add_motion_features(a) for a in A], dtype=np.float32)
    Xtrain, Xva_f, Xte_f = prep(Xtrain), prep(Xva), prep(Xte)
    cat = lambda v: tf.keras.utils.to_categorical(v, num_classes=n)
    ytrain_c, yva_c, yte_c = cat(ytrain), cat(enc.transform(yva)), cat(enc.transform(yte))

    model = build_model(n)
    model.summary()
    hist = model.fit(
        Xtrain, ytrain_c, validation_data=(Xva_f, yva_c), epochs=120, batch_size=32, verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ])

    loss, acc, top5 = model.evaluate(Xte_f, yte_c, verbose=0)
    kind = "signers the model never saw" if signer_split else "random split"
    print(f"\nTEST accuracy ({kind}): {acc:.4f} | top-5: {top5:.4f} | loss {loss:.4f}")

    model.save(MODEL_FILE)
    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(enc, f)

    pred = np.argmax(model.predict(Xte_f, verbose=0), axis=1)
    truth = enc.transform(yte)
    rep = classification_report(truth, pred, labels=range(n), target_names=classes,
                                output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose()
    words = rep_df.iloc[:n].sort_values("f1-score")
    print("\nWeakest 15 words (f1):")
    print(words.head(15)[["precision", "recall", "f1-score", "support"]].round(2).to_string())
    print("\nStrongest 10 words:")
    print(words.tail(10)[["precision", "recall", "f1-score", "support"]].round(2).to_string())

    cm = confusion_matrix(truth, pred, labels=range(n))
    pairs = [(classes[i], classes[j], int(cm[i, j])) for i in range(n) for j in range(n)
             if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda p: -p[2])
    print("\nMost common mix-ups (true -> predicted, count):")
    for a, b, c in pairs[:12]:
        print(f"  {a} -> {b}: {c}")

    h = pd.DataFrame(hist.history)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["accuracy"], label="train"); ax[0].plot(h["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(CURVES_FILE, dpi=150)

    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as w:
        pd.DataFrame({"Metric": ["Test accuracy", "Top-5 accuracy", "Words", "Train clips (real)",
                                 "Val clips", "Test clips", "Evaluation"],
                      "Value": [round(acc, 4), round(top5, 4), n, len(Xtr), len(Xva), len(Xte), kind]}
                     ).to_excel(w, sheet_name="Summary", index=False)
        rep_df.to_excel(w, sheet_name="Per-Word Metrics")
        pd.DataFrame(pairs, columns=["True", "Predicted", "Count"]).to_excel(w, sheet_name="Mix-ups", index=False)
        pd.DataFrame(cm, index=classes, columns=classes).to_excel(w, sheet_name="Confusion Matrix")
        h.to_excel(w, sheet_name="Training History", index=False)
    print(f"\nSaved {MODEL_FILE}, {ENCODER_FILE}, {CURVES_FILE}, {EXCEL_FILE}")


if __name__ == "__main__":
    main()
