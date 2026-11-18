"""
sign_language_translator.py

Real-time ASL fingerspelling translator.

Pipeline:
  webcam frame -> MediaPipe hand landmark detection -> normalize landmarks
  -> trained Keras model predicts letter -> stability/confidence check
  -> confirmed letters build up a sentence -> optional text-to-speech.

Run collect_data.py and train_model.py first to produce sign_model.h5 and
label_encoder.pkl before running this script.

Controls:
  'q' -> quit
  'x' -> clear the current sentence
  't' -> speak the current sentence out loud (requires pyttsx3)
"""

import pickle
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

MODEL_FILE = "sign_model.h5"
ENCODER_FILE = "label_encoder.pkl"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords = coords - wrist
    max_val = np.max(np.linalg.norm(coords, axis=1))
    if max_val > 0:
        coords = coords / max_val
    return coords.flatten()


class SignTranslator:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_FILE)
        with open(ENCODER_FILE, "rb") as f:
            self.encoder = pickle.load(f)

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        self.sentence = ""
        self.recent_predictions = deque(maxlen=15)
        self.last_confirm_time = 0.0
        self.confirm_cooldown = 1.0  # seconds between confirmed letters
        self.confidence_threshold = 0.85
        self.stability_count = 10  # frames needed agreeing before confirming

        self.tts_engine = pyttsx3.init() if TTS_AVAILABLE else None

    def predict(self, features):
        features = features.reshape(1, -1)
        probs = self.model.predict(features, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self.encoder.inverse_transform([idx])[0]
        return label, confidence

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        predicted_label, confidence = None, 0.0

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = normalize_landmarks(hand_landmarks)
            predicted_label, confidence = self.predict(features)
            self.recent_predictions.append(predicted_label)

            if confidence >= self.confidence_threshold:
                now = time.time()
                is_stable = self.recent_predictions.count(predicted_label) >= self.stability_count
                cooldown_elapsed = (now - self.last_confirm_time) > self.confirm_cooldown
                if is_stable and cooldown_elapsed:
                    self._confirm_letter(predicted_label)
                    self.last_confirm_time = now
                    self.recent_predictions.clear()
        else:
            self.recent_predictions.clear()

        self._draw_overlay(frame, predicted_label, confidence)
        return frame

    def _confirm_letter(self, label):
        if label == "SPACE":
            self.sentence += " "
        elif label == "DEL":
            self.sentence = self.sentence[:-1]
        elif label == "NOTHING":
            pass
        else:
            self.sentence += label

    def speak(self):
        if self.tts_engine and self.sentence.strip():
            self.tts_engine.say(self.sentence)
            self.tts_engine.runAndWait()
        elif not self.tts_engine:
            print("pyttsx3 not installed - text-to-speech unavailable.")

    def _draw_overlay(self, frame, label, confidence):
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Sentence: {self.sentence}",
            (10, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        if label:
            cv2.putText(
                frame,
                f"Detected: {label} ({confidence:.2f})",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        cv2.putText(
            frame,
            "q: quit   x: clear   t: speak",
            (w - 260, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )


def main():
    translator = SignTranslator()
    cap = cv2.VideoCapture(0)

    print("Controls: 'q' quit | 'x' clear sentence | 't' speak sentence")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame = translator.process_frame(frame)
        cv2.imshow("Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("x"):
            translator.sentence = ""
        elif key == ord("t"):
            translator.speak()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
