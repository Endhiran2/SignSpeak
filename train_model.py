"""
train_model.py

Trains a feedforward neural network (deep learning classifier) on the
hand-landmark data collected by collect_data.py, to recognize ASL
fingerspelling letters/gestures.

Run this after you've collected a reasonable amount of data
(landmark_data.csv) with collect_data.py.

Outputs:
  - sign_model.h5     : the trained Keras model
  - label_encoder.pkl : maps model output indices back to letter/label strings
"""

import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

DATA_FILE = "landmark_data.csv"
MODEL_FILE = "sign_model.h5"
ENCODER_FILE = "label_encoder.pkl"


def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} labels")
    print(df["label"].value_counts())

    X = df.drop("label", axis=1).values.astype(np.float32)
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    model = models.Sequential(
        [
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  (loss: {loss:.4f})")

    model.save(MODEL_FILE)
    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(encoder, f)

    print(f"\nSaved model to {MODEL_FILE}")
    print(f"Saved label encoder to {ENCODER_FILE}")


if __name__ == "__main__":
    main()
