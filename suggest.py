"""Predictive suggestions for signed phrases (the "search bar for the deaf").

Given the words recognised so far, suggest:
  * next(): likely NEXT words      (like Google's autocomplete)
  * complete(): whole phrases that continue what you've signed so far
Phrases live in phrase_bank.txt (one per line, sign order). Edit it freely.
Every phrase you finish with SPEAK / SENTENCE is also learned (user_phrases.txt).
"""
import os
from collections import Counter, defaultdict

BANK = "phrase_bank.txt"
LEARNED = "user_phrases.txt"
START = "<s>"


class Suggester:
    def __init__(self, vocab, bank=BANK, learned=LEARNED):
        self.vocab = set(vocab)
        self.learned_path = learned
        self.phrases = Counter()
        for path, weight in ((bank, 1), (learned, 3)):
            if os.path.exists(path):
                for line in open(path, encoding="utf-8"):
                    line = line.strip().upper()
                    if not line or line.startswith("#"):
                        continue
                    ws = tuple(line.split())
                    if ws and all(w in self.vocab for w in ws):
                        self.phrases[ws] += weight
        self.bigram = defaultdict(Counter)
        self.unigram = Counter()
        for ws, n in self.phrases.items():
            seq = (START,) + ws
            for a, b in zip(seq, seq[1:]):
                self.bigram[a][b] += n
            for w in ws:
                self.unigram[w] += n

    def next(self, words, k=5):
        """Most likely next words after `words` (list of str)."""
        words = [w.upper() for w in words]
        scores = Counter()
        # phrases that start with everything signed so far -> strongest signal
        n = len(words)
        for ws, c in self.phrases.items():
            if len(ws) > n and ws[:n] == tuple(words):
                scores[ws[n]] += 10 * c
        # phrases containing the last word anywhere
        last = words[-1] if words else START
        for w, c in self.bigram[last].items():
            scores[w] += c
        if len(words) >= 2:
            for ws, c in self.phrases.items():
                for i in range(len(ws) - 2):
                    if ws[i:i + 2] == tuple(words[-2:]):
                        scores[ws[i + 2]] += 3 * c
        if not scores:
            return [w for w, _ in self.unigram.most_common(k)]
        return [w for w, _ in scores.most_common(k)]

    def complete(self, words, k=3):
        """Whole phrases (as tuples of remaining words) that continue `words`."""
        words = tuple(w.upper() for w in words)
        n = len(words)
        out = []
        for ws, c in self.phrases.most_common():
            if n and len(ws) > n and ws[:n] == words:
                out.append((c, ws[n:]))
        out.sort(key=lambda t: (-t[0], len(t[1])))
        return [rest for _, rest in out[:k]]

    def learn(self, phrase):
        ws = tuple(phrase.upper().split())
        if len(ws) < 2 or not all(w in self.vocab for w in ws):
            return
        with open(self.learned_path, "a", encoding="utf-8") as f:
            f.write(" ".join(ws) + "\n")
        self.phrases[ws] += 3
        seq = (START,) + ws
        for a, b in zip(seq, seq[1:]):
            self.bigram[a][b] += 3
        for w in ws:
            self.unigram[w] += 3
