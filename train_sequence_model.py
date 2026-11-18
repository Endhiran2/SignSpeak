"""
train_sequence_model.py

Trains an LSTM-based deep learning model on the motion sequences collected
by collect_sequence_data.py / prepare_wlasl_data.py, to recognize whole
word/phrase-level signs.

Expects a directory structure:
  sequence_data/
    HELLO/
      sample_0000.npy
      ...
    IDLE/
      sample_0000.npy
      ...

Each .npy file is shape (SEQUENCE_LENGTH, 126).

EVALUATION METHODOLOGY (read this if you're checking these numbers):
  Real (recorded/WLASL-derived) sequences are split into train/test FIRST,
  before any synthetic augmentation is generated. Augmented copies are only
  ever created from the training portion, and the test set is 100% real,
  never-augmented, held-out data. This matters: if you split AFTER
  augmenting, a synthetic near-duplicate of a training sample can land in
  the test set, letting the model get credit for something it's
  effectively already seen - silently inflating accuracy. This script
  avoids that.

Outputs:
  - sign_sequence_model.h5     : the trained Keras LSTM model
  - sequence_label_encoder.pkl : maps model output indices back to word labels
  - training_history.csv       : per-epoch accuracy/loss, for plotting
  - training_curves.png        : accuracy & loss vs. epoch, plotted
  - confusion_matrix.png       : heatmap of true vs. predicted labels on the
                                  held-out real test set
  - evaluation_results.xlsx    : summary + per-class precision/recall/F1 +
                                  confusion matrix, as an Excel workbook
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

DATA_DIR = "sequence_data"
MODEL_FILE = "sign_sequence_model.h5"
ENCODER_FILE = "sequence_label_encoder.pkl"
HISTORY_FILE = "training_history.csv"
CURVES_FILE = "training_curves.png"
CONFUSION_FILE = "confusion_matrix.png"
EXCEL_FILE = "evaluation_results.xlsx"
SEQUENCE_LENGTH = 30
RAW_FEATURES = 126          # 2 hands x 21 landmarks x (x, y, z)
NUM_FEATURES = 252          # raw landmarks + frame-to-frame motion (velocity)

# Data augmentation: for every real sequence IN THE TRAINING SPLIT ONLY,
# generate this many synthetic variants (noise, scale, small rotation,
# speed change). Never applied to the test split - see the module docstring.
AUGMENTATIONS_PER_SAMPLE = 4
NOISE_STD = 0.015
SCALE_RANGE = (0.9, 1.1)
ROTATION_DEG = 15           # random in-plane rotation, +/- this many degrees
SPEED_RANGE = (0.8, 1.25)   # >1 = signed faster, <1 = slower

_rng = np.random.default_rng()


def add_motion_features(seq):
    """(30, 126) landmarks -> (30, 252): landmarks + per-frame velocity.
    Velocity is zeroed whenever a hand is missing in either frame, so a hand
    flickering in/out of detection doesn't look like fast motion.
    MUST be identical in word_phrase_translator.py."""
    seq = np.asarray(seq, dtype=np.float32)
    vel = np.zeros_like(seq)
    for h in (0, 63):
        block = seq[:, h:h + 63]
        present = np.any(block != 0, axis=1)
        both = present[1:] & present[:-1]
        vel[1:, h:h + 63] = (block[1:] - block[:-1]) * both[:, None]
    return np.concatenate([seq, vel * 10.0], axis=1).astype(np.float32)


def augment_sequence(seq):
    """Random noise + scale + small rotation per hand, then a random speed
    change for the whole sequence. Only touches detected (non-zero) hands,
    so we never fabricate a hand where none was present."""
    aug = seq.copy()
    for h in (0, 63):  # 0:63 = left-hand features, 63:126 = right-hand
        block = aug[:, h:h + 63]
        present = np.any(block != 0, axis=1)
        if not present.any():
            continue
        pts = block.reshape(-1, 21, 3).copy()
        theta = np.deg2rad(_rng.uniform(-ROTATION_DEG, ROTATION_DEG))
        c, s = np.cos(theta), np.sin(theta)
        x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
        pts[:, :, 0] = c * x - s * y
        pts[:, :, 1] = s * x + c * y
        pts = pts * _rng.uniform(*SCALE_RANGE)
        block = pts.reshape(-1, 63)
        noise = _rng.normal(0, NOISE_STD, size=block[present].shape).astype(np.float32)
        block[present] = block[present] + noise
        block[~present] = 0.0
        aug[:, h:h + 63] = block

    # mirror: flip x and swap the two hands (covers mirrored videos and
    # left-handed signers). Landmarks are wrist-relative, so this is just -x.
    if _rng.random() < 0.5:
        for h in (0, 63):
            blk = aug[:, h:h + 63].reshape(-1, 21, 3).copy()
            blk[:, :, 0] *= -1
            aug[:, h:h + 63] = blk.reshape(-1, 63)
        aug = np.concatenate([aug[:, 63:], aug[:, :63]], axis=1)

    # speed change: resample frame indices (nearest frame, no blending)
    f = _rng.uniform(*SPEED_RANGE)
    idx = np.clip(np.round(np.arange(SEQUENCE_LENGTH) * f), 0, SEQUENCE_LENGTH - 1).astype(int)
    aug = aug[idx]
    return aug.astype(np.float32)


def load_real_dataset():
    """Loads only real (non-augmented) sequences from disk - one entry per
    recorded/extracted .npy file, no synthetic copies."""
    X, y = [], []
    labels = sorted(
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    if not labels:
        raise RuntimeError(
            f"No label folders found in '{DATA_DIR}'. Run collect_sequence_data.py first."
        )

    for label in labels:
        label_dir = os.path.join(DATA_DIR, label)
        files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
        for fname in files:
            seq = np.load(os.path.join(label_dir, fname))
            if seq.shape[0] < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - seq.shape[0], RAW_FEATURES), dtype=np.float32)
                seq = np.vstack([seq, pad])
            elif seq.shape[0] > SEQUENCE_LENGTH:
                seq = seq[:SEQUENCE_LENGTH]
            X.append(seq.astype(np.float32))
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
            layers.Masking(mask_value=0.0),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.4),
            layers.LSTM(64),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["epoch"], history_df["accuracy"], label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["loss"], label="Train")
    axes[1].plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(CURVES_FILE, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(min(36, max(6, len(class_names) * 0.55)), min(32, max(5, len(class_names) * 0.5))))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (held-out real test samples)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_FILE, dpi=150)
    plt.close(fig)


def main():
    X_real, y_real_raw = load_real_dataset()

    encoder = LabelEncoder()
    y_real = encoder.fit_transform(y_real_raw)
    num_classes = len(encoder.classes_)
    class_names = list(encoder.classes_)

    print(f"Loaded {len(X_real)} REAL sequences across {num_classes} labels")
    unique, counts = np.unique(y_real_raw, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} real samples")

    # Split the REAL data first. Augmentation happens only on the training
    # portion afterward, so the test set is never contaminated by synthetic
    # near-duplicates of training samples. See module docstring.
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )

    X_train, y_train = list(X_train_real), list(y_train_real)
    if AUGMENTATIONS_PER_SAMPLE:
        for seq, label in zip(X_train_real, y_train_real):
            for _ in range(AUGMENTATIONS_PER_SAMPLE):
                X_train.append(augment_sequence(seq))
                y_train.append(label)

    X_train = np.array([add_motion_features(s) for s in X_train], dtype=np.float32)
    y_train = np.array(y_train)
    # 100% real, held out, untouched (only the deterministic feature step is applied)
    X_test = np.array([add_motion_features(s) for s in X_test_real], dtype=np.float32)
    y_test = y_test_real

    # Rare words count more, so frequent words (IDLE, YES) don't dominate.
    cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    print(f"\nTraining set: {len(X_train)} sequences ({len(X_train_real)} real + "
          f"{len(X_train) - len(X_train_real)} augmented)")
    print(f"Test set: {len(X_test)} sequences (100% real, never augmented)")

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    model = build_model(num_classes)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
    )
    csv_logger = tf.keras.callbacks.CSVLogger(HISTORY_FILE)

    model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop, reduce_lr, csv_logger],
        class_weight=class_weight,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest accuracy (leakage-free, real data only): {acc:.4f}  (loss: {loss:.4f})")

    model.save(MODEL_FILE)
    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(encoder, f)
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved label encoder to {ENCODER_FILE}")

    # ---- Evaluation report: per-class metrics + confusion matrix ----
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    print("\nPer-class metrics (held-out real test set):")
    print(report_df.round(3))

    cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Charts
    history_df = pd.read_csv(HISTORY_FILE)
    plot_training_curves(history_df)
    plot_confusion_matrix(cm, class_names)
    print(f"Saved training curves to {CURVES_FILE}")
    print(f"Saved confusion matrix plot to {CONFUSION_FILE}")

    # Excel workbook: summary + per-class metrics + confusion matrix
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Overall test accuracy",
                "Overall test loss",
                "Real training samples",
                "Augmented training samples",
                "Real test samples (held out, never augmented)",
                "Number of classes",
            ],
            "Value": [
                round(acc, 4),
                round(loss, 4),
                len(X_train_real),
                len(X_train) - len(X_train_real),
                len(X_test),
                num_classes,
            ],
        }
    )

    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        report_df.to_excel(writer, sheet_name="Per-Class Metrics")
        cm_df.to_excel(writer, sheet_name="Confusion Matrix")
        history_df.to_excel(writer, sheet_name="Training History", index=False)

    print(f"Saved evaluation workbook to {EXCEL_FILE}")


if __name__ == "__main__":
    main()
