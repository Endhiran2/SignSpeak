"""
record_idle.py   -  record the movements that are NOT signs (adjusting
headphones or glasses, scratching, drinking...), so the model can learn to
ignore them. Same idea as drill_mode.py, but nothing is a word.

Run (venv active, project folder):
    python record_idle.py            # 10 everyday movements x 6 takes (about 5 minutes)
    python record_idle.py --reps 10
    python record_idle.py --also "tie my hair,hold my phone"   # add your own

For each prompt: do the movement naturally (hands up in view), then lower your
hands. The clip is saved when it ends. Vary it: faster, slower, left hand, right hand.
Keys: u = undo last take | n = next prompt | q = quit
Clips go to idle_data/<movement>/ . Then:  python train_idle_gate.py
"""
import argparse
import os
import re
import time

import cv2
import numpy as np

import translator_holistic as T

PROMPTS = [
    "adjust your headphones",
    "adjust your glasses",
    "scratch your face or nose",
    "touch or fix your hair",
    "rub your eyes or chin",
    "drink from a cup or bottle",
    "pick up and hold your phone",
    "wave or fidget with your hands",
    "stretch your arms",
    "move your hands to rest them in your lap or on the desk",
]
OUT = "idle_data"
WINDOW = "Record idle movements"


def slug(s):
    return re.sub(r"[^a-z]+", "_", s.lower()).strip("_")


def put(frame, text, org, scale, color, th=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--also", default="")
    a, _ = ap.parse_known_args()
    prompts = PROMPTS + [p.strip() for p in a.also.split(",") if p.strip()]

    tr = T.WordPhraseTranslator()
    tr.gate = None            # record everything, whatever the idle filter would say
    T.DEBUG = False
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow(WINDOW)

    i, got, saved = 0, {p: 0 for p in prompts}, []
    flash, t0 = ("", 0.0), time.time()
    while i < len(prompts):
        ok, frame = cap.read()
        if not ok:
            print("Could not read from webcam."); break
        p = prompts[i]
        tr.last_seq, tr.last_label = None, None
        tr.process_frame(frame)
        tr.phrase = ""
        if tr.last_seq is not None:
            d = os.path.join(OUT, slug(p)); os.makedirs(d, exist_ok=True)
            path = os.path.join(d, time.strftime("i_%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}.npy")
            np.save(path, tr.last_seq.astype(np.float32))
            saved.append((p, path)); got[p] += 1
            flash = (f"Saved ({sum(got.values())} total)", time.time()); t0 = time.time()
            if got[p] >= a.reps:
                i += 1; tr.reset()
        d = cv2.flip(frame, 1)
        h, w = d.shape[:2]
        cv2.rectangle(d, (0, 0), (w, 90), (0, 0, 0), -1)
        if i < len(prompts):
            put(d, f"Now: {prompts[i]}", (12, 42), 0.9, (255, 255, 255), 2)
            put(d, f"take {got[prompts[i]] + 1} of {a.reps}  |  prompt {i + 1} of {len(prompts)}  |  lower hands to finish",
                (12, 76), 0.55, (200, 200, 200), 1)
        if time.time() - flash[1] < 2:
            put(d, flash[0], (12, 125), 0.8, (0, 255, 0), 2)
        elif time.time() - t0 > 10:
            put(d, "Do the movement, then lower your hands", (12, 125), 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW, d)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            i += 1; tr.reset()
        elif key == ord("u") and saved:
            pr, path = saved.pop()
            if os.path.exists(path):
                os.remove(path)
            got[pr] -= 1
            i = min(i, prompts.index(pr))
            flash = ("Undone", time.time())
    cap.release(); cv2.destroyAllWindows()
    n = sum(got.values())
    print(f"\nSaved {n} idle clips in {OUT}/.  Next:  python train_idle_gate.py")


if __name__ == "__main__":
    main()
