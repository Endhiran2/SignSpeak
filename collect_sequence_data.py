"""
collect_sequence_data.py

Collects short motion sequences of two-hand landmarks for training a
word/phrase-level sign recognizer. Each sample is a fixed-length clip
(SEQUENCE_LENGTH frames) of 126 features per frame (21 landmarks x 3 coords
x 2 hands, zero-filled if a hand isn't visible).

Controls:
  'l' -> start typing a new label, right in the camera window (e.g.
         "HELLO", "THANK_YOU", "IDLE" for a no-sign/resting baseline
         class). Press Enter to confirm, Esc to cancel, Backspace to
         correct a typo.
  'r' -> start recording one sample of the current label (after a short
         countdown so you can get into position)
  'q' -> quit

Tips:
  - Record at least 20-30 samples per word, varying your speed/position
    slightly each time.
  - Definitely record an "IDLE" class: hands resting, fidgeting, or doing
    nothing sign-related. Without it, the real-time app will try to
    classify every random hand movement as a word.
  - Keep each sign's motion within roughly 1 second (SEQUENCE_LENGTH at
    ~30fps) - if your signs are slower/faster, adjust SEQUENCE_LENGTH
    and FPS_ESTIMATE below and re-collect.
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = "sequence_data"
SEQUENCE_LENGTH = 30  # frames per recorded sample
COUNTDOWN_SECONDS = 3


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords = coords - wrist
    max_val = np.max(np.linalg.norm(coords, axis=1))
    if max_val > 0:
        coords = coords / max_val
    return coords.flatten()  # (63,)


def extract_two_hand_features(results):
    """Returns a fixed-size 126-length vector: 63 for left hand + 63 for
    right hand, zero-filled for any hand not currently detected."""
    features = {"Left": np.zeros(63, dtype=np.float32), "Right": np.zeros(63, dtype=np.float32)}
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            features[label] = normalize_landmarks(hand_landmarks).astype(np.float32)
    return np.concatenate([features["Left"], features["Right"]])  # (126,)


def next_sample_index(label_dir):
    existing = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
    return len(existing)


def record_sequence(cap, hands, label):
    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    # Countdown
    countdown_start = time.time()
    while time.time() - countdown_start < COUNTDOWN_SECONDS:
        ret, frame = cap.read()
        if not ret:
            return
        display = cv2.flip(frame, 1)
        remaining = COUNTDOWN_SECONDS - (time.time() - countdown_start)
        cv2.putText(display, f"Get ready: {remaining:.1f}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(display, f"Label: {label}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Sequence Data Collection", display)
        cv2.waitKey(1)

    sequence = []
    while len(sequence) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            return

        # IMPORTANT: landmarks are extracted from the RAW (unflipped) frame,
        # to match the orientation of the WLASL dataset videos (which are
        # not mirrored). We only flip a copy for on-screen display, so the
        # feature data saved here stays consistent with prepare_wlasl_data.py
        # and with word_phrase_translator.py's live inference.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        features = extract_two_hand_features(result)
        sequence.append(features)

        display = cv2.flip(frame, 1)
        cv2.putText(display, f"RECORDING  {len(sequence)}/{SEQUENCE_LENGTH}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display, f"Label: {label}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Sequence Data Collection", display)
        cv2.waitKey(1)

    sequence = np.stack(sequence)  # (SEQUENCE_LENGTH, 126)
    idx = next_sample_index(label_dir)
    path = os.path.join(label_dir, f"sample_{idx:04d}.npy")
    np.save(path, sequence)
    print(f"Saved {path}  shape={sequence.shape}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
    )

    current_label = None
    typing_mode = False
    typed_buffer = ""
    print(__doc__)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = cv2.flip(frame, 1)

        if typing_mode:
            cv2.putText(display, f"Type label, Enter=confirm Esc=cancel: {typed_buffer}_",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display, f"Current label: {current_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "l: set label | r: record | q: quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Sequence Data Collection", display)
        key = cv2.waitKey(1) & 0xFF

        if typing_mode:
            # Typing happens entirely within this loop (no blocking input()
            # call), so the camera feed keeps running the whole time and
            # the window manager never sees the app as "unresponsive."
            if key in (13, 10):  # Enter
                label = typed_buffer.strip().upper()
                if label:
                    current_label = label
                    print(f"Label set to: {current_label}")
                typing_mode = False
                typed_buffer = ""
            elif key == 27:  # Esc cancels without changing the label
                typing_mode = False
                typed_buffer = ""
            elif key == 8:  # Backspace
                typed_buffer = typed_buffer[:-1]
            elif key != 255 and 32 <= key <= 126:
                ch = chr(key)
                if ch.isalnum() or ch in (" ", "_", "-"):
                    typed_buffer += ch
            continue

        if key == ord("q"):
            break
        elif key == ord("l"):
            typing_mode = True
            typed_buffer = ""
        elif key == ord("r"):
            if current_label:
                record_sequence(cap, hands, current_label)
            else:
                print("Set a label first by pressing 'l'.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
