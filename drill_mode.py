"""
drill_mode.py   -  quickly teach the model YOUR signing, no typing.

The app shows a word, you sign it, and the clip is saved automatically as that
word. Repeat a few times per word, and it moves on. It also tells you whether
the CURRENT model got each take right, so it doubles as a test.

Run (venv active, project folder):
    python drill_mode.py                     # drills the usual weak words
    python drill_mode.py --words GOOD,BAD,GO # your own list
    python drill_mode.py --file mywords.txt  # one word (or several) per line
    python drill_mode.py --all               # every word in the vocabulary
    python drill_mode.py --reps 8            # takes per word (default 5)

How to sign each take: raise your hands, sign, then LOWER your hands (or hold
still) so the app knows the sign ended. The clip is saved when it does.

Keys:  u = undo the last take (bad take, redo it)   n = skip this word
       q = quit (everything saved so far is kept)
If sign_sheet/<WORD>.png exists (python make_sign_sheet.py) it is shown as a
reference at the bottom.

Afterwards:  python finetune_user.py
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

import translator_holistic as T

# words that were weakest in the signer-independent test
WEAK = ("GOOD BAD THANKYOU STOP WRITE GO COME MONEY NEED READ BUY EXCUSE YOU NIGHT HOT TALK "
        "KITCHEN SISTER TODAY BLUE WHAT NAME CALL ANSWER DOCTOR WORK OLD DOG NOW YES SCHOOL "
        "WEEK MOTHER BROTHER").split()
SHEET_DIR = "sign_sheet"
WINDOW = "Drill mode"


class Drill:
    """The drill's bookkeeping (no camera code), so it is easy to reason about."""

    def __init__(self, words, reps):
        self.words, self.reps = list(words), reps
        self.i = 0                        # index of the current word
        self.got = {w: 0 for w in words}  # takes saved
        self.right = {w: 0 for w in words}  # takes the current model got right
        self.saved = []                   # (word, path, was_right) in order

    @property
    def finished(self):
        return self.i >= len(self.words)

    @property
    def word(self):
        return self.words[self.i]

    def record(self, path, predicted):
        w = self.word
        ok = predicted == w
        self.got[w] += 1
        self.right[w] += int(ok)
        self.saved.append((w, path, ok))
        if self.got[w] >= self.reps:
            self.i += 1
        return ok

    def undo(self):
        if not self.saved:
            return None
        w, path, ok = self.saved.pop()
        self.got[w] -= 1
        self.right[w] -= int(ok)
        if self.words.index(w) < self.i:
            self.i = self.words.index(w)
        return path

    def skip(self):
        if not self.finished:
            self.i += 1


def newest(folder, before):
    now = {f for f in os.listdir(folder) if f.endswith(".npy")} if os.path.isdir(folder) else set()
    new = sorted(now - before)
    return os.path.join(folder, new[-1]) if new else None


def listing(folder):
    return {f for f in os.listdir(folder) if f.endswith(".npy")} if os.path.isdir(folder) else set()


def put(frame, text, org, scale, color, th=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words")
    ap.add_argument("--file")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    a, _ = ap.parse_known_args()

    tr = T.WordPhraseTranslator()
    tr.gate = None            # record everything, whatever the idle filter would say
    T.DEBUG = False
    vocab = set(tr._words())
    if a.all:
        words = sorted(vocab)
    elif a.words:
        words = [w.strip().upper() for w in a.words.split(",")]
    elif a.file:
        words = [w.upper() for line in open(a.file) for w in line.replace(",", " ").split()]
    else:
        words = WEAK
    bad = [w for w in words if w not in vocab]
    if bad:
        print("Not in the vocabulary, skipped:", ", ".join(bad))
    words = [w for w in dict.fromkeys(words) if w in vocab]
    if not words:
        raise SystemExit("No valid words to drill.")
    drill = Drill(words, a.reps)
    print(f"Drilling {len(words)} words x {a.reps} takes. Raise hands, sign, lower hands. u=undo n=skip q=quit")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow(WINDOW)
    flash = ("", 0.0, (255, 255, 255))
    t_word = time.time()
    last_i = -1
    ref_cache = {}

    while not drill.finished:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from webcam.")
            break
        if drill.i != last_i:
            last_i, t_word = drill.i, time.time()
            tr.reset()
        w = drill.word
        tr.last_seq, tr.last_label = None, None
        tr.process_frame(frame)
        tr.phrase = ""
        if tr.last_seq is not None:                       # a sign just finished
            folder = os.path.join(T.USER_DIR, w)
            before = listing(folder)
            predicted = tr.last_label
            if tr.save_clip(w):
                path = newest(folder, before)
                right = drill.record(path, predicted)
                flash = (f"Saved. Model heard: {predicted}" + ("  (correct)" if right else "  (wrong)"),
                         time.time(), (0, 255, 0) if right else (0, 165, 255))
                t_word = time.time()

        d = cv2.flip(frame, 1)
        h, wd = d.shape[:2]
        cv2.rectangle(d, (0, 0), (wd, 90), (0, 0, 0), -1)
        if not drill.finished:
            put(d, f"Sign:  {drill.word}", (12, 45), 1.4, (255, 255, 255), 3)
            put(d, f"take {drill.got[drill.word] + 1} of {drill.reps}   |   word {drill.i + 1} of {len(words)}",
                (12, 78), 0.6, (200, 200, 200), 1)
            if drill.word not in ref_cache:
                p = os.path.join(SHEET_DIR, drill.word + ".png")
                img = cv2.imread(p) if os.path.exists(p) else None
                ref_cache[drill.word] = None if img is None else cv2.resize(img, (wd, int(img.shape[0] * wd / img.shape[1])))
            ref = ref_cache[drill.word]
            if ref is not None:
                rh = min(ref.shape[0], h // 3)
                d[h - rh:h, 0:wd] = ref[:rh]
        if time.time() - flash[1] < 2.5:
            put(d, flash[0], (12, 125), 0.8, flash[2], 2)
        elif time.time() - t_word > 10:
            put(d, "Raise hands, sign, then lower hands", (12, 125), 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW, d)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            drill.skip()
        elif key == ord("u"):
            path = drill.undo()
            if path and os.path.exists(path):
                os.remove(path)
                flash = ("Undone - sign it again", time.time(), (0, 255, 255))

    cap.release()
    cv2.destroyAllWindows()

    total = sum(drill.got.values())
    right = sum(drill.right.values())
    print(f"\nSaved {total} clips this session; the CURRENT model got {right} of them right "
          f"({right / total:.0%})." if total else "\nNo clips saved.")
    rows = [(w, drill.right[w], drill.got[w]) for w in words if drill.got[w]]
    if rows:
        print("\nPer word (model right / takes):")
        print("  " + ", ".join(f"{w} {r}/{n}" for w, r, n in sorted(rows, key=lambda t: t[1] / t[2])))
    print(f"\nTotal clips in {T.USER_DIR}/: {tr.count_user_clips()}.  Next:  python finetune_user.py")


if __name__ == "__main__":
    main()
