# Sign Language Translator (ASL Fingerspelling)

A real-time American Sign Language (ASL) fingerspelling recognizer built with
OpenCV (video), MediaPipe (hand landmark detection), and a Keras deep
learning classifier. It reads hand shapes from your webcam and builds up a
sentence, with optional text-to-speech.

## What it does (and doesn't) do

- ✅ Recognizes static ASL alphabet hand shapes (A–Y, excluding J which
  requires motion) plus SPACE, DELETE, and NOTHING gestures you define.
- ✅ Builds sentences in real time and can speak them aloud.
- ❌ Does not translate full, continuous sign language (which involves
  grammar, facial expressions, and motion-based signs) — that's a much
  larger research problem. See "Going further" below for how to extend
  toward that.

## 1. Install dependencies

Requires Python 3.9–3.11 (TensorFlow and MediaPipe compatibility varies by
version — 3.10 is a safe bet).

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If `pyttsx3` fails to install/run on your OS, everything else still works —
text-to-speech is optional.

## 2. Collect training data

```bash
python collect_data.py
```

- Press a letter key (A–Y, skipping J) to set the label you're about to
  demonstrate, hold that hand shape in front of the webcam, then press `c`
  repeatedly to capture samples (vary your hand angle/position slightly
  between captures).
- Press `space` for the "SPACE" gesture, `backspace` for "DEL", `n` for
  "NOTHING" (relaxed hand / no sign).
- Aim for **150–300 samples per label** for decent accuracy.
- Press `s` to save periodically, `ESC` to quit (auto-saves on exit).

This produces `landmark_data.csv`.

## 3. Train the model

```bash
python train_model.py
```

This trains a small feedforward neural network on the landmark data and
saves `sign_model.h5` and `label_encoder.pkl`. Watch the printed test
accuracy — if it's low, collect more/cleaner data (consistent lighting,
varied angles, and being careful about mislabeled captures matters much
more than model architecture at this stage).

## 4. Run the translator

```bash
python sign_language_translator.py
```

- Hold a hand shape steadily; once the model is confident and stable for
  ~10 frames, the letter is added to the sentence.
- `x` clears the sentence, `t` speaks it aloud, `q` quits.

## Tuning tips

- If letters are confirmed too eagerly/erratically: raise
  `confidence_threshold` or `stability_count` in
  `sign_language_translator.py`.
- If it feels sluggish to register signs: lower `stability_count` or
  `confirm_cooldown`.
- If certain letters are frequently confused (e.g. M/N/S, which look
  similar in ASL), collect more targeted data distinguishing them.

## Part 2: Word/phrase-level translation (motion-aware)

The scripts above only recognize static letters. `collect_sequence_data.py`,
`train_sequence_model.py`, and `word_phrase_translator.py` add a second,
independent pipeline that recognizes whole word/phrase signs by modeling
**motion over time** with an LSTM, using both hands.

### How it works

- Each frame, both hands' landmarks are extracted (126 features: 63 per
  hand, zero-filled if a hand isn't visible) and normalized.
- A rolling window of the last 30 frames (~1 second) is fed into an LSTM,
  which classifies the *sequence* as a whole word — this is what lets it
  pick up on motion, not just a fixed hand shape.
- When a word is predicted confidently and consistently for several frames
  in a row, it's confirmed, added to the phrase, and the buffer clears so
  you get a natural pause before the next sign.

### Steps

```bash
python collect_sequence_data.py
```
- Press `l`, type a word/phrase label in the terminal (e.g. `HELLO`,
  `THANK_YOU`), press Enter.
- Press `r` to record one ~1-second clip of yourself performing that sign
  (there's a 3-second countdown first). Repeat 20-30+ times per word,
  varying your speed and position slightly.
- **Important**: also record an `IDLE` label — clips of your hands resting,
  fidgeting, or moving without signing anything. Without this baseline
  class, the app will try to interpret every hand movement as a word.
- Press `q` when done. This produces a `sequence_data/` folder with one
  subfolder per label.

```bash
python train_sequence_model.py
```
Trains the LSTM on your recorded sequences, saves `sign_sequence_model.h5`
and `sequence_label_encoder.pkl`, and prints test accuracy.

```bash
python word_phrase_translator.py
```
Runs the real-time word/phrase translator. Perform a sign, pause briefly,
and the recognized word should appear and get added to the phrase.
Controls: `x` clear phrase, `t` speak phrase, `q` quit.

### Option B: use WLASL instead of recording everything yourself

`prepare_wlasl_data.py` converts videos from the **WLASL** (Word-Level ASL)
dataset into the same landmark-sequence format `collect_sequence_data.py`
produces, so you get a real vocabulary without recording each word.

```bash
pip install kaggle
```
1. Get a free Kaggle API token: https://www.kaggle.com/settings → "Create
   New Token" → downloads `kaggle.json` → place it at `~/.kaggle/kaggle.json`
   (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
2. Download the pre-processed WLASL dataset (videos already downloaded,
   ~4.8GB — this mirror avoids WLASL's original dead-link problem):
   ```bash
   kaggle datasets download -d risangbaskoro/wlasl-processed
   unzip wlasl-processed.zip -d wlasl_raw
   ```
   You should end up with `wlasl_raw/WLASL_v0.3.json` and
   `wlasl_raw/videos/<id>.mp4`.
3. Open `prepare_wlasl_data.py` and edit `TARGET_GLOSSES` to the words you
   want (start with ~15–30 to test the pipeline before scaling up — WLASL
   has ~2,000 words total, but more words = more processing time and a
   harder classification problem).
4. Run it:
   ```bash
   python prepare_wlasl_data.py
   ```
   This populates `sequence_data/<WORD>/sample_XXXX.npy` for each word,
   using each clip's labeled bounding box and frame range, resampled to
   match the model's expected sequence length.
5. **Important**: WLASL only contains positive examples of words being
   signed — no "nothing is happening" clips. Run `collect_sequence_data.py`
   and record a handful of your own `IDLE` clips (hands resting, casual
   movement) so the model can tell signing apart from pauses. These drop
   into the same `sequence_data/` folder and combine automatically.
6. Then train as normal:
   ```bash
   python train_sequence_model.py
   python word_phrase_translator.py
   ```

You can mix both approaches freely — WLASL words plus your own recorded
words all live side by side in `sequence_data/`, since they're saved in
the same format.

### Limitations of this approach (be aware)

- **You have to pause between signs** — this app clears its buffer after
  each confirmed word, so it doesn't handle continuous, flowing signing
  where one word blends into the next without a break. Real continuous
  sign language recognition needs proper temporal segmentation, which is
  a harder research problem.
- **Vocabulary is limited to what you record.** This isn't pretrained on
  general ASL vocabulary — it only knows the exact words/phrases you
  collect training data for. Realistically, expect to hand-record dozens,
  not thousands, of signs before this becomes tedious; see below for
  using an existing dataset instead.
- **Two hands, no face/body.** Some ASL signs also use facial expression,
  shoulder/head movement, or mouth shapes as part of the sign; this system
  only looks at hand landmarks.

## Going further (toward full sign language translation)

- **Pretrained vocabulary instead of self-recording**: train on
  [WLASL](https://dxli94.github.io/WLASL/) (Word-Level ASL, 2,000+ signs)
  or RWTH-PHOENIX (German Sign Language) instead of `collect_sequence_data.py`,
  to get broad vocabulary coverage without recording everything yourself.
- **Continuous signing without pauses**: replace the "clear buffer after
  each word" approach with proper temporal action segmentation/detection
  (e.g. a sliding window with change-point detection, or a
  CTC-style/attention-based sequence-to-sequence model that can output a
  variable-length sequence of words from one continuous video, similar to
  speech recognition).
- **Non-manual markers**: add MediaPipe Face Mesh landmarks as additional
  input features, since facial expression carries real grammatical
  meaning in ASL (e.g. question intonation).
- **Combine both pipelines**: fall back to the letter-by-letter
  fingerspelling model for proper nouns or words outside the word-level
  vocabulary, similar to how human signers fingerspell unfamiliar terms.
- **Text-to-Sign (reverse direction)**: rendering an avatar or video clips
  that sign back translated text, for two-way communication.
