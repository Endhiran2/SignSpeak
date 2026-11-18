"""
collect_data.py

Collects hand-landmark training data via webcam for the ASL fingerspelling
translator. For each frame, MediaPipe extracts 21 (x, y, z) landmarks per
detected hand, which are normalized (relative to wrist, scaled by hand size)
and saved to a CSV along with the label you assign.

Controls:
  - Letter keys A-Y (except J)  -> set current label to that letter
  - SPACE bar                   -> set current label to "SPACE"
  - BACKSPACE                   -> set current label to "DEL"
  - 'n'                         -> set current label to "NOTHING" (no sign)
  - 'c'                         -> capture one sample under the current label
  - 's'                         -> save all captured samples to CSV
  - ESC                         -> quit (auto-saves on exit)

Tip: For a good model, aim for at least 150-300 samples per label, captured
across slightly different hand angles, positions, and lighting.
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_FILE = "landmark_data.csv"
VALID_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")  # J and Z need motion, excluded
SPECIAL_LABELS = {"SPACE", "DEL", "NOTHING"}


def normalize_landmarks(hand_landmarks):
    """Translate landmarks relative to the wrist and scale by hand size,
    so the model is invariant to hand position and distance from camera."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords = coords - wrist
    max_val = np.max(np.linalg.norm(coords, axis=1))
    if max_val > 0:
        coords = coords / max_val
    return coords.flatten()  # shape (63,)


def save_samples(samples):
    if not samples:
        return
    columns = [f"f{i}" for i in range(63)] + ["label"]
    df = pd.DataFrame(samples, columns=columns)
    if os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(samples)} samples to {DATA_FILE}")
    samples.clear()


def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )

    current_label = None
    samples = []

    print(__doc__)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(
            frame,
            f"Label: {current_label}   Buffered: {len(samples)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(" "):
            current_label = "SPACE"
        elif key == 8:  # Backspace
            current_label = "DEL"
        elif key == ord("n"):
            current_label = "NOTHING"
        elif key == ord("c"):
            if current_label and result.multi_hand_landmarks:
                feats = normalize_landmarks(result.multi_hand_landmarks[0])
                samples.append(list(feats) + [current_label])
                print(f"Captured sample for '{current_label}', buffered: {len(samples)}")
            elif not result.multi_hand_landmarks:
                print("No hand detected - nothing captured.")
        elif key == ord("s"):
            save_samples(samples)
        else:
            ch = chr(key).upper() if 0 <= key < 256 else ""
            if ch in VALID_LETTERS:
                current_label = ch

    save_samples(samples)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
