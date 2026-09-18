"""Local BM25 sparse vectors stored in Qdrant's sparse index.

Enterprise text is full of exact identifiers (INC-2026-0618, OPS-RB-004, S001)
that dense embeddings can miss, so each chunk also gets a BM25-weighted sparse
vector under the collection's sparse slot. The vocabulary/IDF statistics are
built from the Phase 2 corpus and persisted so queries encode identically.
"""
from __future__ import annotations

import json
import math
import re

from qdrant_client.models import SparseVector

from . import config

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

K1 = 1.2
B = 0.75


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class SparseEncoder:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    def fit(self, texts: list[str]) -> "SparseEncoder":
        doc_freq: dict[str, int] = {}
        total_len = 0
        for text in texts:
            tokens = tokenize(text)
            total_len += len(tokens)
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        n = max(len(texts), 1)
        self.vocab = {t: i for i, t in enumerate(sorted(doc_freq))}
        self.idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for t, df in doc_freq.items()
        }
        self.avgdl = total_len / n if n else 0.0
        return self

    def encode(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        if not tokens or not self.vocab:
            return SparseVector(indices=[], values=[])
        tf: dict[str, int] = {}
        for token in tokens:
            if token in self.vocab:
                tf[token] = tf.get(token, 0) + 1
        dl = len(tokens)
        pairs: list[tuple[int, float]] = []
        for token, freq in tf.items():
            norm = freq * (K1 + 1) / (
                freq + K1 * (1 - B + B * dl / max(self.avgdl, 1e-6)))
            pairs.append((self.vocab[token], self.idf[token] * norm))
        pairs.sort()
        return SparseVector(
            indices=[i for i, _ in pairs],
            values=[float(v) for _, v in pairs],
        )

    def save(self) -> None:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.VOCAB_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                {"vocab": self.vocab, "idf": self.idf, "avgdl": self.avgdl},
                fh,
            )

    @classmethod
    def load(cls) -> "SparseEncoder":
        with open(config.VOCAB_FILE, encoding="utf-8") as fh:
            raw = json.load(fh)
        enc = cls()
        enc.vocab = {k: int(v) for k, v in raw["vocab"].items()}
        enc.idf = {k: float(v) for k, v in raw["idf"].items()}
        enc.avgdl = float(raw["avgdl"])
        return enc
