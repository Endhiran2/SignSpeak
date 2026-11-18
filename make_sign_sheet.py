"""
make_sign_sheet.py  -  builds a visual reference of every word the model knows.

For each word it takes one real ASL Citizen video (the same data the model was
trained on), pulls 6 frames across the sign and puts them in a strip. Output:
  sign_sheet/<WORD>.png    one strip per word
  sign_sheet/index.html    all words on one page (open it in your browser)

Run (venv active, project folder):   python make_sign_sheet.py
Options: --split train|val|test   (which clips to pick from, default train)
         --pick N                 (use the Nth clip of each word, default 0; try 1, 2 ...
                                   if a clip is unclear)
"""
import argparse, glob, html, os, pickle, re
import cv2
import numpy as np
import pandas as pd

RAW_DIR = "asl_citizen_raw"
ENCODER_FILE = "holistic_label_encoder.pkl"
OUT = "sign_sheet"
FRAMES, H = 6, 230


def load_metadata():
    frames = []
    for path in glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        v = next((cols[c] for c in cols if "video" in c), None)
        g = next((cols[c] for c in cols if "gloss" in c), None)
        stem = os.path.basename(path).lower()
        split = next((s for s in ("train", "val", "test") if s in stem), None)
        if v and g and split:
            frames.append(pd.DataFrame({"video": df[v], "gloss": df[g], "split": split}))
    meta = pd.concat(frames).drop_duplicates(subset=["video"])
    meta["gloss"] = meta["gloss"].astype(str).str.upper().str.strip()
    meta["word"] = meta["gloss"].str.replace(r"\d+$", "", regex=True)   # WANT1 -> WANT
    return meta


def strip(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 6:
        return None
    imgs = []
    for i in np.linspace(n * 0.15, n * 0.85, FRAMES).astype(int):   # skip rest poses at the ends
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            h, w = fr.shape[:2]
            imgs.append(cv2.resize(fr, (int(w * H / h), H)))
    cap.release()
    return np.hstack(imgs) if len(imgs) == FRAMES else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--pick", type=int, default=0)
    a = ap.parse_args()

    words = sorted(pickle.load(open(ENCODER_FILE, "rb")).classes_)
    meta = load_metadata()
    videos = {os.path.basename(p): p for p in glob.glob(os.path.join(RAW_DIR, "**", "*.mp4"), recursive=True)}
    os.makedirs(OUT, exist_ok=True)

    done = []
    for w in words:
        rows = meta[(meta["word"] == w) & (meta["split"] == a.split)]
        if rows.empty:
            rows = meta[meta["word"] == w]
        img = None
        for k in range(a.pick, len(rows)):
            p = videos.get(os.path.basename(str(rows.iloc[k]["video"])))
            if p:
                img = strip(p)
                if img is not None:
                    break
        if img is None:
            print("no clip for", w); continue
        cv2.imwrite(os.path.join(OUT, f"{w}.png"), img)
        done.append(w)
        print(f"{len(done)}/{len(words)} {w}")

    cards = "\n".join(
        f'<div class="c" data-w="{html.escape(w)}"><h3>{html.escape(w)} '
        f'<button class="ok" onclick="mark(\'{html.escape(w)}\',1)">&#10003; worked</button>'
        f'<button class="no" onclick="mark(\'{html.escape(w)}\',-1)">&#10007; failed</button></h3>'
        f'<img src="{html.escape(w)}.png"></div>' for w in done)
    page = """<!doctype html><meta charset="utf-8"><title>Sign reference and test sheet</title>
<style>body{font-family:sans-serif;max-width:1100px;margin:20px auto;padding:0 12px}
.c{margin:0 0 18px;padding:6px;border-left:6px solid #ddd} h3{margin:0 0 4px}
img{max-width:100%;border:1px solid #ccc} button{margin-left:8px;cursor:pointer}
.c.pass{border-color:#2e9e4f} .c.fail{border-color:#d33}
#bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccc}</style>
<h1>Words the model knows (__N__)</h1>
<div id="bar"><span id="sum"></span>
<label style="margin-left:16px"><input type="checkbox" id="onlyfail" onchange="render()"> show only failed</label>
<button onclick="if(confirm('Clear all marks?')){localStorage.removeItem('signtest');render()}">clear marks</button></div>
<p>Six frames from one ASL Citizen clip per word, left to right. Sign the word to the model, then mark it.
People sign slightly differently; run make_sign_sheet.py --pick 1 for another clip.</p>
__CARDS__
<script>
function load(){try{return JSON.parse(localStorage.getItem('signtest')||'{}')}catch(e){return {}}}
function mark(w,v){var d=load();d[w]=(d[w]===v)?0:v;try{localStorage.setItem('signtest',JSON.stringify(d))}catch(e){}render()}
function render(){var d=load(),p=0,f=0,only=document.getElementById('onlyfail').checked;
document.querySelectorAll('.c').forEach(function(c){var v=d[c.dataset.w]||0;
c.className='c'+(v==1?' pass':v==-1?' fail':'');if(v==1)p++;if(v==-1)f++;
c.style.display=(only&&v!=-1)?'none':'';});
document.getElementById('sum').textContent='worked '+p+' | failed '+f+' | untested '+(document.querySelectorAll('.c').length-p-f);}
render();
</script>"""
    page = page.replace("__N__", str(len(done))).replace("__CARDS__", cards)
    open(os.path.join(OUT, "index.html"), "w").write(page)
    print(f"\nOpen {OUT}/index.html in your browser.")

if __name__ == "__main__":
    main()
