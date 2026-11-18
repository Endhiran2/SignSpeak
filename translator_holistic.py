"""
translator_holistic.py   (v2 pipeline, step 3: the live app)

Real-time translator using hands + body pose (see holistic_features.py), so
the model knows WHERE your hands are, not just their shape. Needs the files
made by train_holistic_model.py: sign_holistic_model.keras and
holistic_label_encoder.pkl, plus holistic_features.py in the same folder.

HOW IT WORKS: one prediction per sign, not one per frame.
  1. Raise your hands and sign. Frames are collected while hands are visible.
  2. The sign ends when your hands drop out of view (or hold still for a moment).
  3. The collected sign is trimmed and resampled to 30 frames - exactly how
     the training clips were made - and the model predicts ONCE.
  4. The word is accepted only if confidence >= confidence_threshold AND its
     top guess beats its second guess by min_margin.
This does not depend on camera speed, so lag no longer ruins the window. If it
accepts wrong words, raise those two numbers; if it ignores real signs, lower
them (DEBUG prints what it saw after every sign).

Controls: 'q' quit | 'x' reset | 't' speak | 'f' finalize into a sentence
(or click the on-screen buttons). 'f' needs ANTHROPIC_API_KEY.

TEACH IT YOUR SIGNING (live correction):
  'c' (or WRONG? button) - the last sign was misread: type the right word in
      the window, press Enter. The clip is saved to user_data/<WORD>/.
      Esc cancels, Backspace deletes.
  'k' (or RIGHT button)  - the last guess was correct: saves it as an example too.
Then run  python finetune_user.py  to teach the model from those clips.
If sign_holistic_model_user.keras exists it is used automatically;
start with  --base  to use the original model instead.
"""

import os
import pickle
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

import holistic_features as hf

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

MODEL_FILE = "sign_holistic_model.keras"
ENCODER_FILE = "holistic_label_encoder.pkl"
USER_MODEL_FILE = "sign_holistic_model_user.keras"
USER_DIR = "user_data"
GATE_FILE = "idle_gate.keras"
GATE_JSON = "idle_gate.json"
DEBUG = True   # prints one line per sign

CLAUDE_MODEL = "claude-sonnet-5"
SENTENCE_SYSTEM_PROMPT = (
    "You convert a sequence of ASL (American Sign Language) gloss words into "
    "a single, natural, fluent English sentence. The words are given in ASL "
    "sign order, which often differs from English word order (e.g. 'STORE I "
    "GO' means 'I'm going to the store'). Assume a first-person subject "
    "unless the words clearly indicate otherwise. Respond with ONLY the "
    "resulting English sentence - no explanation, no quotes, nothing else."
)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class WordPhraseTranslator:
    def __init__(self):
        use_user = os.path.exists(USER_MODEL_FILE) and "--base" not in sys.argv
        model_path = USER_MODEL_FILE if use_user else MODEL_FILE
        print(f"Loading model: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.gate, self.gate_threshold = None, 0.5
        if os.path.exists(GATE_FILE) and "--no-gate" not in sys.argv:
            try:
                import json
                self.gate = tf.keras.models.load_model(GATE_FILE)
                self.gate_threshold = float(json.load(open(GATE_JSON))["threshold"])
                print(f"Idle filter on (threshold {self.gate_threshold:.2f}); start with --no-gate to turn it off")
            except Exception as e:
                print("Could not load idle filter:", e)
        with open(ENCODER_FILE, "rb") as f:
            self.encoder = pickle.load(f)

        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,          # same as extraction; fastest
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.segment = []          # frames of the sign in progress
        self.gap = 0               # consecutive frames with no hand
        self.still = 0             # consecutive frames of (almost) no hand motion
        self.armed = True          # False after a still-hold until you move again
        self.phrase = ""
        self.last_sentence = ""
        self.last_result = (None, 0.0)
        self.last_seq = None       # (30,232) of the last sign that was predicted
        self.last_label = None
        self.sugg = None
        self.sugg_items = []       # [(key_label, text, action, payload)] drawn as chips
        self.last_alts = []
        self.last_top = []
        self.pending = False       # last sign was too uncertain to add: offer candidates instead
        self.typing = None         # None, or the text typed so far while correcting
        self.status = ("", 0.0)    # (message, time shown)
        self.fps = 0.0
        self._t_prev = None

        # ---- tunables ----
        self.confidence_threshold = 0.50
        self.min_margin = 0.15
        self.max_gap = 4           # frames a hand may flicker out mid-sign
        self.min_active = 8        # shortest sign worth predicting (frames)
        self.max_segment = 90      # force a prediction if hands stay up this long
        self.still_motion = 0.03   # per-frame hand motion below this = holding still
        self.move_motion = 0.08    # motion above this re-arms after a still-hold
        self.still_frames = 8

        self.tts_engine = pyttsx3.init() if TTS_AVAILABLE else None
        self.claude_client = None
        if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            self.claude_client = Anthropic()

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _hand_motion(prev, cur):
        """Largest per-frame movement of either hand's location (shoulder widths)."""
        best = 0.0
        for shape, loc in ((hf.L_SHAPE, hf.L_LOC), (hf.R_SHAPE, hf.R_LOC)):
            if np.any(prev[shape] != 0) and np.any(cur[shape] != 0):
                best = max(best, float(np.linalg.norm(cur[loc] - prev[loc])))
        return best

    def _segment_motion(self, seg):
        if len(seg) < 2:
            return 0.0
        return float(np.mean([self._hand_motion(seg[i - 1], seg[i]) for i in range(1, len(seg))]))

    def predict(self, seq30):
        x = hf.add_motion_features(seq30)[np.newaxis]
        probs = self.model(x, training=False).numpy()[0]
        order = np.argsort(probs)[::-1]
        label = self.encoder.inverse_transform([int(order[0])])[0]
        conf = float(probs[order[0]])
        margin = float(probs[order[0]] - probs[order[1]])
        self.last_alts = [self.encoder.inverse_transform([int(i)])[0] for i in order[1:4]]
        self.last_top = [(self.encoder.inverse_transform([int(i)])[0], float(probs[i])) for i in order[:4]]
        if DEBUG:
            top3 = [(self.encoder.inverse_transform([int(i)])[0], float(probs[i])) for i in order[:3]]
            print("Top 3:", ", ".join(f"{l}={p:.2f}" for l, p in top3), f"| margin={margin:.2f}")
        return label, conf, margin

    def _finish(self, seg):
        """A sign just ended. Predict once and accept or reject it."""
        self.segment, self.gap, self.still = [], 0, 0
        seg = np.array(seg, dtype=np.float32)
        if len(seg) < self.min_active:
            return
        motion = self._segment_motion(seg)
        if motion < self.still_motion * 1.5:      # hands were just resting
            if DEBUG:
                print(f"(hands up but still, motion={motion:.3f} - ignored)")
            return
        seq = hf.trim_and_resample(seg, min_active=self.min_active)
        if seq is None:
            return
        self.last_seq = seq.copy()
        if self.gate is not None:
            g = float(self.gate(hf.add_motion_features(seq)[np.newaxis], training=False).numpy()[0][0])
            if g < self.gate_threshold:
                if DEBUG:
                    print(f"(not a sign: idle filter {g:.2f} < {self.gate_threshold:.2f} - ignored)")
                self.last_label = None
                self._say("ignored (not a sign)")
                return
        label, conf, margin = self.predict(seq)
        self.last_label = label
        if DEBUG:
            print(f"-> {label} conf={conf:.2f} margin={margin:.2f} frames={len(seg)} motion={motion:.3f}")
        self.last_result = (label, conf)
        if conf >= self.confidence_threshold and margin >= self.min_margin:
            self.pending = False
            self._confirm_word(label)
        else:
            self.pending = True
            self._say("not sure - pick the right word (keys 1-4)")

    # ------------------------------------------------------------ main step
    def update(self, f):
        """Feed one frame's (232,) features. Predicts when a sign ends."""
        present = bool(np.any(f[hf.L_SHAPE] != 0) or np.any(f[hf.R_SHAPE] != 0))

        if present:
            if not self.armed:
                prev = self.segment[-1] if self.segment else f
                if self._hand_motion(prev, f) > self.move_motion:
                    self.armed = True
                else:
                    self.segment = [f]
                    return
            self.gap = 0
            if self.segment:
                m = self._hand_motion(self.segment[-1], f)
                self.still = self.still + 1 if m < self.still_motion else 0
            self.segment.append(f)
            if len(self.segment) >= self.max_segment:
                self._finish(self.segment)
            elif self.still >= self.still_frames and len(self.segment) >= self.min_active + self.still_frames:
                self._finish(self.segment[:-self.still_frames])
                self.armed = False
        else:
            if self.segment:
                self.gap += 1
                self.segment.append(f)
                if self.gap > self.max_gap:
                    self._finish(self.segment[:-self.gap])
            elif not self.armed:
                self.gap += 1
                if self.gap >= 3:
                    self.armed, self.gap = True, 0

    def process_frame(self, frame):
        """frame: RAW (unflipped) camera frame; landmarks are drawn onto it."""
        now = time.time()
        if self._t_prev:
            dt = max(now - self._t_prev, 1e-3)
            self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1.0 / dt
        self._t_prev = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        for hl in (results.left_hand_landmarks, results.right_hand_landmarks):
            if hl:
                mp_drawing.draw_landmarks(frame, hl, mp_holistic.HAND_CONNECTIONS)

        if self.typing is None:            # recognition pauses while you type a correction
            self.update(hf.extract_frame_features(results))
        return self.last_result

    # ---------------------------------------------------- live correction
    def _say(self, msg):
        self.status = (msg, time.time())
        print(msg)

    def _words(self):
        return list(self.encoder.classes_)

    def count_user_clips(self):
        n = 0
        if os.path.isdir(USER_DIR):
            for w in os.listdir(USER_DIR):
                n += len([f for f in os.listdir(os.path.join(USER_DIR, w)) if f.endswith(".npy")])
        return n

    def save_clip(self, word):
        if self.last_seq is None:
            self._say("No sign to save yet.")
            return False
        word = word.upper()
        if word not in self._words():
            self._say(f"'{word}' is not in the vocabulary.")
            return False
        d = os.path.join(USER_DIR, word)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, time.strftime("u_%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}.npy")
        np.save(path, self.last_seq.astype(np.float32))
        self._say(f"Saved clip as {word}  ({self.count_user_clips()} total)")
        self.last_seq = None          # one save per sign
        return True

    def keep_last(self):
        if self.last_label is None or self.last_seq is None:
            self._say("Nothing to confirm yet.")
        else:
            self.save_clip(self.last_label)

    def begin_correct(self):
        if self.last_seq is None:
            self._say("Sign something first, then press c.")
        else:
            self.typing = ""

    def handle_typing_key(self, key):
        """Returns True if the key was consumed by the correction prompt."""
        if self.typing is None:
            return False
        if key == 27:                                   # Esc
            self.typing = None
        elif key in (8, 127):                           # Backspace
            self.typing = self.typing[:-1]
        elif key in (13, 10):                           # Enter
            matches = [w for w in self._words() if w.startswith(self.typing)]
            word = self.typing if self.typing in self._words() else (matches[0] if len(matches) == 1 else None)
            if word and self.save_clip(word):
                self.typing = None
            elif not word:
                self._say("Type more letters until one word matches, or Esc to cancel.")
        elif 65 <= key <= 90 or 97 <= key <= 122:
            self.typing += chr(key).upper()
        return True

    def reset(self):
        self.phrase = ""
        self.pending = False
        self.last_sentence = ""
        self.segment, self.gap, self.still = [], 0, 0
        self.armed = True
        self.last_result = (None, 0.0)

    def _confirm_word(self, label):
        if self.phrase:
            self.phrase += " "
        self.phrase += label
        self.alts_for = label

    def _suggester(self):
        if self.sugg is None:
            try:
                from suggest import Suggester
                self.sugg = Suggester(self._words())
            except Exception as e:
                print("Suggestions unavailable:", e)
                self.sugg = False
        return self.sugg or None

    def build_suggestions(self):
        """Chips. Normal: 1-5 next words, 9 finish phrase, 6-8 'did you mean' for the last sign.
        After an uncertain sign (nothing added): 1-4 = the model's top guesses with confidence."""
        items = []
        if self.pending and self.last_top and self.typing is None:
            for i, (w, p) in enumerate(self.last_top[:4]):
                items.append((str(i + 1), f"{w} {p*100:.0f}%", "pick", w))
            self.sugg_items = items
            return items
        sg = self._suggester()
        words = self.phrase.split()
        if sg and self.typing is None:
            comps = sg.complete(words, 1) if words else []
            for w in sg.next(words, 5):
                items.append((str(len(items) + 1), w, "add", w))
            if comps:
                rest = comps[0]
                items.append(("9", "+ " + " ".join(rest), "add", " ".join(rest)))
        if words and len(self.last_top) >= 4 and getattr(self, "alts_for", None) == words[-1]:
            for i, (a, p) in enumerate(self.last_top[1:4]):
                items.append((str(6 + i), f"{a}? {p*100:.0f}%", "swap", a))
        self.sugg_items = items
        return items

    def pick_suggestion(self, idx):
        idx = str(idx)
        for lab, text, act, pay in self.sugg_items:
            if lab != idx:
                continue
            words = self.phrase.split()
            if act in ("swap", "pick"):
                if self.last_seq is not None:
                    self.save_clip(pay)           # your correction becomes training data
                if act == "swap" and words:
                    words[-1] = pay
                    self.phrase = " ".join(words)
                else:
                    self.phrase = (self.phrase + " " + pay).strip()
                self.pending = False
                self.alts_for = None
                self._say(f"{pay} (saved for training)")
            else:
                self.phrase = (self.phrase + " " + pay).strip()
                self.alts_for = None
            return True
        return False

    def _learn_phrase(self):
        sg = self._suggester()
        if sg and self.phrase.strip():
            sg.learn(self.phrase)

    def speak(self):
        self._learn_phrase()
        if self.tts_engine and self.phrase.strip():
            self.tts_engine.say(self.phrase)
            self.tts_engine.runAndWait()
        elif not self.tts_engine:
            print("pyttsx3 not installed - text-to-speech unavailable.")

    def finalize_sentence(self):
        """Sends the recognized word sequence (in ASL sign order) to Claude
        to produce a natural English sentence. Requires ANTHROPIC_API_KEY to
        be set in the environment and the 'anthropic' package installed."""
        self._learn_phrase()
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
        if self.typing is not None:
            matches = [w for w in self._words() if w.startswith(self.typing)][:6]
            cv2.rectangle(frame, (20, 60), (w - 20, 150), (30, 30, 30), -1)
            cv2.rectangle(frame, (20, 60), (w - 20, 150), (0, 200, 255), 2)
            cv2.putText(frame, f"Correct word: {self.typing}_", (35, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "  ".join(matches) if matches else "(no match)", (35, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.putText(frame, "Enter = save   Esc = cancel", (35, 142),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        self.build_suggestions()
        cx, cy = 10, h - bar_height - 40
        self.chip_rects = _chip_rects
        _chip_rects.clear()
        for lab, text, act, pay in self.sugg_items:
            label = f"{lab} {text}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if cx + tw + 16 > w:
                break
            col = (0, 140, 255) if act == "swap" else (90, 200, 90)
            cv2.rectangle(frame, (cx, cy), (cx + tw + 14, cy + th + 12), (40, 40, 40), -1)
            cv2.rectangle(frame, (cx, cy), (cx + tw + 14, cy + th + 12), col, 1)
            cv2.putText(frame, label, (cx + 7, cy + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            self.chip_rects[lab] = (cx, cy, cx + tw + 14, cy + th + 12)
            cx += tw + 24
        msg, t0 = self.status
        if msg and time.time() - t0 < 4:
            cv2.putText(frame, msg, (10, h - bar_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        buf_pct = len(self.segment)
        cv2.putText(frame, f"FPS {self.fps:.0f} | sign frames {buf_pct}", (w - 280, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


BUTTONS = [("RESET (x)", "reset"), ("SPEAK (t)", "speak"), ("SENTENCE (f)", "sentence"),
           ("WRONG? (c)", "correct"), ("RIGHT (k)", "keep")]
_button_rects = {}
_pending_action = {"name": None}
_chip_rects = {}


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
        for lab, (x1, y1, x2, y2) in list(_chip_rects.items()):
            if x1 <= mx <= x2 and y1 <= my <= y2:
                _pending_action["name"] = "chip" + lab


WINDOW = "Word/Phrase Sign Translator"


def main():
    translator = WordPhraseTranslator()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("Click the buttons, or press: 'q' quit | 'x' reset | 't' speak | 'f' sentence")
    print("Suggestions: 1-5 next word, 9 finish phrase, 6-8 swap a misread word; after an unsure sign, 1-4 pick the right one - or click a chip")
    print("Teach it: 'c' = last sign was wrong (type the right word) | 'k' = it was right")

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
        cv2.putText(display, f"FPS: {translator.fps:.0f}", (display.shape[1] - 110, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        draw_buttons(display)
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        action = _pending_action["name"]
        _pending_action["name"] = None
        if translator.typing is not None:
            if key != 255:
                translator.handle_typing_key(key)
            continue
        if key == ord("q"):
            break
        elif ord("1") <= key <= ord("9"):
            translator.pick_suggestion(chr(key))
        elif action and action.startswith("chip"):
            translator.pick_suggestion(action[4:])
        elif key == ord("c") or action == "correct":
            translator.begin_correct()
        elif key == ord("k") or action == "keep":
            translator.keep_last()
        elif key == ord("x") or action == "reset":
            translator.reset()
        elif key == ord("t") or action == "speak":
            translator.speak()
        elif key == ord("f") or action == "sentence":
            translator.finalize_sentence()

    cap.release()
    cv2.destroyAllWindows()
    n = translator.count_user_clips()
    if n:
        print(f"\n{n} of your clips are saved in {USER_DIR}/. Teach the model with:  python finetune_user.py")


if __name__ == "__main__":
    main()
