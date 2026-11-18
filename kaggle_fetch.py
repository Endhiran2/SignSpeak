"""
kaggle_fetch.py  -  step 1 of adding the Kaggle "Google - Isolated Sign Language
Recognition" (asl-signs) clips for the words we share with it.

Downloads ONLY the landmark files for our words (default 60 clips per word,
spread across the 21 signers), not the 40 GB dataset.

Needs: pip install pyarrow     and your kaggle.json already working
Run:   python kaggle_fetch.py                 (uses train.csv, downloads it if missing)
       python kaggle_fetch.py --per-word 100
       python kaggle_fetch.py --info          (which of our words the dataset has)
       python kaggle_fetch.py --dry-run       (just prints what it would download)
"""
import argparse, os, re, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

DATASET = "najaf456ali/asl-100-google-slr"   # public copy of the Google ISLR data (no competition access needed)
OUT = "kaggle_raw"

OURS = set(open("check_kaggle_overlap.py").read().split('"""')[3].split())  # reuse word list


def norm(s):
    return re.sub(r"[^A-Z]", "", s.upper())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-word", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--info", action="store_true", help="only show which of our words this dataset has")
    a = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()

    if not os.path.exists("train.csv"):
        print("Downloading train.csv ...")
        api.dataset_download_file(DATASET, "train.csv", path=".", quiet=False)
        if os.path.exists("train.csv.zip"):
            zipfile.ZipFile("train.csv.zip").extractall("."); os.remove("train.csv.zip")
    df = pd.read_csv("train.csv")
    df["word"] = df["sign"].map(norm)
    df = df[df["word"].isin(OURS)]
    print(f"{df['word'].nunique()} shared words, {len(df)} clips available")
    if a.info:
        cnt = df.groupby("word").size().sort_values(ascending=False)
        print("\nclips per shared word (all signers):")
        print(", ".join(f"{w} {n}" for w, n in cnt.items()))
        print("\nsigners:", df["participant_id"].nunique())
        return

    pick = []
    for w, g in df.groupby("word"):
        # round-robin over signers so every signer is represented
        buckets = [list(x["path"]) for _, x in g.groupby("participant_id")]
        chosen, i = [], 0
        while len(chosen) < min(a.per_word, len(g)):
            for b in buckets:
                if i < len(b) and len(chosen) < a.per_word:
                    chosen.append(b[i])
            i += 1
        pick += [(w, p) for p in chosen]
    print(f"Will download {len(pick)} files into {OUT}/")
    if a.dry_run:
        return
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(pick, columns=["word", "path"]).to_csv(os.path.join(OUT, "_selection.csv"), index=False)

    def get(item):
        w, p = item
        dest = os.path.join(OUT, os.path.basename(p))
        if os.path.exists(dest):
            return True
        err = None
        for attempt in range(8):
            try:
                api.dataset_download_file(DATASET, p, path=OUT, quiet=True)
                z = dest + ".zip"
                if os.path.exists(z):
                    zipfile.ZipFile(z).extractall(OUT); os.remove(z)
                if os.path.exists(dest):
                    return True
            except Exception as e:
                err = e
                import time
                time.sleep(3 * (attempt + 1) if "429" in str(e) else 1)
        print("failed:", p, err)
        return False

    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for ok in ex.map(get, pick):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(pick)}")
    print("Done. Next: python kaggle_convert.py")


if __name__ == "__main__":
    main()
