"""
word_phrase_translator.py

Real-time word/phrase-level ASL translator. Unlike sign_language_translator.py
(which classifies one static hand shape per frame), this maintains a rolling
window of the last SEQUENCE_LENGTH frames of two-hand landmarks and feeds
that whole window into an LSTM model, so it can recognize signs that involve
motion (most real ASL words).

Run collect_sequence_data.py and train_sequence_model.py first to produce
sign_sequence_model.h5 and sequence_label_encoder.pkl.

How recognition works:
  1. Landmarks are extracted every frame and pushed into a rolling buffer.
  2. Once the buffer is full, it's fed to the LSTM every frame.
  3. When the model is confident AND agrees on the same word for several
     consecutive frames AND the cooldown has elapsed, the word is confirmed
     and appended to the phrase.
  4. The buffer is then cleared, so you get a brief pause before signing the
     next word - this gives a clean boundary between signs (real continuous
     sign language would need proper segmentation, which is a harder
     research problem - see README).

Sentence formatting (optional, requires an Anthropic API key):
  The recognized words are in ASL sign order, not English grammar (e.g.
  "GO ART SCHOOL" rather than "I go to art school"). Pressing 'f' sends the
  raw recognized word sequence to the Claude API, which turns it into a
  natural English sentence. This step needs internet access and an
  ANTHROPIC_API_KEY environment variable - see the README for setup. If no
  key is configured, everything else in this script still works; you just
  won't be able to use 'f'.

Controls:
  'q' -> quit
  'x' -> clear the current phrase
  't' -> speak the current phrase aloud (requires pyttsx3)
  'f' -> finalize: send the recognized words to Claude to form a natural
         English sentence (requires ANTHROPIC_API_KEY)
"""

import os
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

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

MODEL_FILE = "sign_sequence_model.h5"
ENCODER_FILE = "sequence_label_encoder.pkl"
SEQUENCE_LENGTH = 30
RAW_FEATURES = 126
NUM_FEATURES = 252   # raw landmarks + velocity (must match train_sequence_model.py)
DEBUG = True  # prints top-3 predicted labels + confidence to the terminal
              # every time the model runs, for troubleshooting

CLAUDE_MODEL = "claude-sonnet-5"
SENTENCE_SYSTEM_PROMPT = (
    "You convert a sequence of ASL (American Sign Language) gloss words into "
    "a single, natural, fluent English sentence. The words are given in ASL "
    "sign order, which often differs from English word order (e.g. 'STORE I "
    "GO' means 'I'm going to the store'). Assume a first-person subject "
    "unless the words clearly indicate otherwise. Respond with ONLY the "
    "resulting English sentence - no explanation, no quotes, nothing else."
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils



def add_motion_features(seq):
    """(30, 126) landmarks -> (30, 252). MUST be identical to the function of
    the same name in train_sequence_model.py."""
    seq = np.asarray(seq, dtype=np.float32)
    vel = np.zeros_like(seq)
    for h in (0, 63):
        block = seq[:, h:h + 63]
        present = np.any(block != 0, axis=1)
        both = present[1:] & present[:-1]
        vel[1:, h:h + 63] = (block[1:] - block[:-1]) * both[:, None]
    return np.concatenate([seq, vel * 10.0], axis=1).astype(np.float32)


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords = coords - wrist
    max_val = np.max(np.linalg.norm(coords, axis=1))
    if max_val > 0:
        coords = coords / max_val
    return coords.flatten()


def extract_two_hand_features(results):
    features = {"Left": np.zeros(63, dtype=np.float32), "Right": np.zeros(63, dtype=np.float32)}
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            features[label] = normalize_landmarks(hand_landmarks).astype(np.float32)
    return np.concatenate([features["Left"], features["Right"]])


class WordPhraseTranslator:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_FILE)
        with open(ENCODER_FILE, "rb") as f:
            self.encoder = pickle.load(f)

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )

        self.buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.recent_predictions = deque(maxlen=8)
        self.phrase = ""
        self.last_confirm_time = 0.0
        self.motion_threshold = 0.05      # below this, hands are treated as still (force IDLE)
        self.presence_threshold = 0.7     # below this fraction of frames with a hand visible, force IDLE
        self.confirm_cooldown = 1.5       # pause required between confirmed words (no gate now, so a bit longer)
        self.confidence_threshold = 0.7
        self.min_margin = 0.25            # top-1 must beat top-2 by this much
        self.stability_count = 5          # consecutive-ish matching frames needed to confirm

        self.tts_engine = pyttsx3.init() if TTS_AVAILABLE else None

        self.claude_client = None
        if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            self.claude_client = Anthropic()

        self.last_sentence = ""

    def _motion_energy(self):
        """Average frame-to-frame movement across the buffered window. Near
        zero means your hands have been essentially still (nothing being
        signed), regardless of what the classifier itself might guess."""
        arr = np.array(self.buffer, dtype=np.float32)
        diffs = np.diff(arr, axis=0)
        return float(np.mean(np.linalg.norm(diffs, axis=1)))

    def _hand_presence_ratio(self):
        """Fraction of buffered frames where at least one hand was actually
        detected. A low ratio means the hand was flickering in and out of
        detection (occlusion, motion blur, moving out of frame) rather than
        being reliably tracked through a real sign - that flicker itself can
        create artificial "motion" that fools the energy check, so we treat
        unreliable windows as unreliable rather than classifying them."""
        arr = np.array(self.buffer, dtype=np.float32)
        present = np.any(arr != 0, axis=1)
        return float(np.mean(present))

    def predict(self):
        seq = add_motion_features(np.array(self.buffer, dtype=np.float32))[np.newaxis]
        probs = self.model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self.encoder.inverse_transform([idx])[0]

        # If the top two guesses are too close, the model is unsure: treat as IDLE.
        top2 = np.sort(probs)[-2:]
        if (top2[1] - top2[0]) < self.min_margin:
            label = "IDLE"

        if DEBUG:
            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = [(self.encoder.inverse_transform([i])[0], float(probs[i])) for i in top3_idx]
            print("Top 3:", ", ".join(f"{lbl}={p:.2f}" for lbl, p in top3))

        return label, confidence

    def process_frame(self, frame):
        """frame: the RAW camera frame (not yet flipped). Landmarks are
        extracted here, in the same (unmirrored) orientation as
        prepare_wlasl_data.py uses for WLASL videos, so live predictions
        are geometrically consistent with how the model was trained. The
        hand skeleton is drawn directly onto this raw frame; the caller is
        responsible for flipping the frame afterward for display."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        features = extract_two_hand_features(result)
        self.buffer.append(features)

        predicted_label, confidence = None, 0.0

        if len(self.buffer) == SEQUENCE_LENGTH:
            motion_energy = self._motion_energy()
            presence_ratio = self._hand_presence_ratio()

            if motion_energy < self.motion_threshold or presence_ratio < self.presence_threshold:
                # Either hands are essentially still, or detection was too
                # unreliable/flickery across this window to trust - in both
                # cases, override to IDLE rather than let the classifier
                # guess on bad input.
                predicted_label, confidence = "IDLE", 1.0
                if DEBUG:
                    print(f"(motion={motion_energy:.4f} presence={presence_ratio:.2f} - forcing IDLE)")
            else:
                predicted_label, confidence = self.predict()
                if DEBUG:
                    print(f"(motion={motion_energy:.4f} presence={presence_ratio:.2f})")

            self.recent_predictions.append(predicted_label)

            if predicted_label != "IDLE" and confidence >= self.confidence_threshold:
                now = time.time()
                is_stable = self.recent_predictions.count(predicted_label) >= self.stability_count
                cooldown_elapsed = (now - self.last_confirm_time) > self.confirm_cooldown
                if is_stable and cooldown_elapsed:
                    self._confirm_word(predicted_label)
                    self.last_confirm_time = now
                    self.buffer.clear()
                    self.recent_predictions.clear()

        return predicted_label, confidence

    def reset(self):
        """Full reset for starting a new sentence: clears the phrase, the
        sentence, and the internal buffers so nothing from the last sign
        leaks into the next one."""
        self.phrase = ""
        self.last_sentence = ""
        self.buffer.clear()
        self.recent_predictions.clear()
        self.last_confirm_time = time.time()  # brief pause before the next word

    def _confirm_word(self, label):
        if label == "IDLE":
            return  # baseline/no-sign class, don't add to phrase
        if self.phrase:
            self.phrase += " "
        self.phrase += label

    def speak(self):
        if self.tts_engine and self.phrase.strip():
            self.tts_engine.say(self.phrase)
            self.tts_engine.runAndWait()
        elif not self.tts_engine:
            print("pyttsx3 not installed - text-to-speech unavailable.")

    def finalize_sentence(self):
        """Sends the recognized word sequence (in ASL sign order) to Claude
        to produce a natural English sentence. Requires ANTHROPIC_API_KEY to
        be set in the environment and the 'anthropic' package installed."""
        if not self.phrase.strip():
            print("Nothing to finalize yet - sign a few words first.")
            return

        if not ANTHROPIC_AVAILABLE:
            print("The 'anthropic' package isn't installed - run: pip install anthropic")
            return

        if not self.claude_client:
            print("ANTHROPIC_API_KEY isn't set - see the README for setup instructions.")
            return

        print(f"Sending to Claude: \"{self.phrase}\" ...")
        try:
            response = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=100,
                system=SENTENCE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self.phrase}],
            )
            sentence = response.content[0].text.strip()
            self.last_sentence = sentence
            print(f"Sentence: {sentence}")
        except Exception as e:
            print(f"Error calling Claude API: {e}")

    def _draw_overlay(self, frame, label, confidence):
        h, w, _ = frame.shape
        bar_height = 130 if self.last_sentence else 100
        cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)

        # Trim the phrase from the front (keep the most recent words) so
        # the text never overflows past the right edge of the window.
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        max_text_width = w - 20
        display_phrase = self.phrase
        prefix = ""
        while display_phrase:
            text = f"Phrase: {prefix}{display_phrase}"
            (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
            if text_w <= max_text_width:
                break
            space_idx = display_phrase.find(" ")
            if space_idx == -1:
                display_phrase = ""
                break
            display_phrase = display_phrase[space_idx + 1:]
            prefix = "... "
        cv2.putText(frame, f"Phrase: {prefix}{display_phrase}", (10, h - bar_height + 40),
                    font, scale, (255, 255, 255), thickness)

        if self.last_sentence:
            sent_font, sent_scale, sent_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            display_sentence = self.last_sentence
            prefix2 = ""
            while display_sentence:
                text = f"Sentence: {prefix2}{display_sentence}"
                (text_w, _), _ = cv2.getTextSize(text, sent_font, sent_scale, sent_thickness)
                if text_w <= max_text_width:
                    break
                space_idx = display_sentence.find(" ")
                if space_idx == -1:
                    display_sentence = ""
                    break
                display_sentence = display_sentence[space_idx + 1:]
                prefix2 = "... "
            cv2.putText(frame, f"Sentence: {prefix2}{display_sentence}", (10, h - bar_height + 75),
                        sent_font, sent_scale, (0, 220, 255), sent_thickness)

        label_y = h - 20
        if label:
            cv2.putText(frame, f"Detected: {label} ({confidence:.2f})", (10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        buf_pct = int(100 * len(self.buffer) / SEQUENCE_LENGTH)
        cv2.putText(frame, f"Buffer: {buf_pct}%", (w - 160, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


BUTTONS = [("RESET (x)", "reset"), ("SPEAK (t)", "speak"), ("SENTENCE (f)", "sentence")]
_button_rects = {}
_pending_action = {"name": None}


def draw_buttons(frame):
    """Clickable buttons along the top-left of the window."""
    x = 10
    for text, action in BUTTONS:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x2, y1, y2 = x + tw + 20, 10, 10 + th + 16
        _button_rects[action] = (x, y1, x2, y2)
        cv2.rectangle(frame, (x, y1), (x2, y2), (60, 60, 60), -1)
        cv2.rectangle(frame, (x, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(frame, text, (x + 10, y2 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        x = x2 + 10


def on_mouse(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for action, (x1, y1, x2, y2) in _button_rects.items():
            if x1 <= mx <= x2 and y1 <= my <= y2:
                _pending_action["name"] = action


WINDOW = "Word/Phrase Sign Translator"


def main():
    translator = WordPhraseTranslator()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("Click the buttons, or press: 'q' quit | 'x' reset | 't' speak | 'f' sentence")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam.")
            break

        # Landmarks are extracted from the raw frame (process_frame draws
        # the hand skeleton onto it in place). We flip only afterward, for
        # a natural mirror-like display, then draw the text overlay on the
        # flipped copy so it reads correctly on screen.
        predicted_label, confidence = translator.process_frame(frame)
        display = cv2.flip(frame, 1)
        translator._draw_overlay(display, predicted_label, confidence)
        draw_buttons(display)
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        action = _pending_action["name"]
        _pending_action["name"] = None
        if key == ord("q"):
            break
        elif key == ord("x") or action == "reset":
            translator.reset()
        elif key == ord("t") or action == "speak":
            translator.speak()
        elif key == ord("f") or action == "sentence":
            translator.finalize_sentence()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
