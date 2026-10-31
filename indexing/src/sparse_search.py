"""Versioned sparse vectors using BM25 document weights and query tf.

Variant: Robertson/Sparck Jones positive IDF, k1=1.2, b=0.75, with linear
query term frequency.  Document vectors contain IDF * BM25(tf, length); query
vectors contain query term counts only. Their dot product therefore applies
IDF once. Sparse artifacts are coupled to the active retrieval configuration.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path

from qdrant_client.models import SparseVector

from . import config

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
K1 = 1.2
B = 0.75
SPARSE_SCHEMA_VERSION = 2
SPARSE_VARIANT = "bm25-rsj-positive-idf-linear-query-tf-v2"
_HYPHENS = {**dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212\ufe58\ufe63\uff0d"), "-"),
            ord("_"): "-"}


class SparseCompatibilityError(RuntimeError):
    """The persisted sparse artifact cannot be proven compatible with index config."""


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("sparse input must be text")
    text = unicodedata.normalize("NFKC", text).translate(_HYPHENS)
    return text.casefold()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def _retrieval_signature() -> dict:
    cfg = config.DEFAULT_RETRIEVAL
    return {
        "sparse_schema_version": SPARSE_SCHEMA_VERSION,
        "variant": SPARSE_VARIANT,
        "tokenizer": "nfkc-ascii-hyphen-underscore-v2",
        "k1": K1,
        "b": B,
        "retrieval_config": asdict(cfg),
        "collection": config.QDRANT_COLLECTION_NAME,
    }


class SparseEncoder:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    def fit(self, texts: list[str]) -> "SparseEncoder":
        if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
            raise ValueError("BM25 corpus must be a list of strings")
        doc_freq: dict[str, int] = {}
        total_len = 0
        for text in texts:
            tokens = tokenize(text)
            total_len += len(tokens)
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        n = len(texts)
        self.vocab = {term: i for i, term in enumerate(sorted(doc_freq))}
        self.idf = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        self.avgdl = total_len / n if n else 0.0
        self._validate()
        return self

    def _validate(self) -> None:
        if (not math.isfinite(self.avgdl) or self.avgdl < 0
                or len(self.vocab) != len(set(self.vocab.values()))
                or set(self.vocab) != set(self.idf)):
            raise SparseCompatibilityError("Malformed BM25 vocabulary or statistics")
        for term, index in self.vocab.items():
            weight = self.idf[term]
            if (not isinstance(term, str) or not term
                    or not isinstance(index, int) or index < 0
                    or not math.isfinite(weight) or weight < 0):
                raise SparseCompatibilityError("Non-finite or invalid BM25 weight")

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        if not tokens or not self.vocab:
            return SparseVector(indices=[], values=[])
        tf: dict[str, int] = {}
        for term in tokens:
            if term in self.vocab:
                tf[term] = tf.get(term, 0) + 1
        dl = len(tokens)
        pairs = []
        for term, freq in tf.items():
            denominator = freq + K1 * (1.0 - B + B * dl / max(self.avgdl, 1e-12))
            weight = self.idf[term] * freq * (K1 + 1.0) / denominator
            if not math.isfinite(weight):
                raise SparseCompatibilityError("BM25 document weight is not finite")
            pairs.append((self.vocab[term], weight))
        pairs.sort()
        return SparseVector(indices=[i for i, _ in pairs],
                            values=[float(v) for _, v in pairs])

    def encode_query(self, text: str) -> SparseVector:
        """Encode query frequency without IDF; document weights supply it once."""
        tf: dict[str, int] = {}
        for term in tokenize(text):
            if term in self.vocab:
                tf[term] = tf.get(term, 0) + 1
        pairs = sorted((self.vocab[term], float(freq))
                       for term, freq in tf.items())
        if any(not math.isfinite(weight) for _, weight in pairs):
            raise SparseCompatibilityError("BM25 query weight is not finite")
        return SparseVector(indices=[i for i, _ in pairs],
                            values=[weight for _, weight in pairs])

    def encode(self, text: str) -> SparseVector:
        """Compatibility alias for indexing callers: document-side weights."""
        return self.encode_document(text)

    def _fingerprint(self) -> str:
        payload = {"vocab": self.vocab, "idf": self.idf, "avgdl": self.avgdl,
                   "retrieval": _retrieval_signature()}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint()

    def save(self, path: Path | None = None) -> None:
        self._validate()
        path = path or config.VOCAB_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"schema_version": SPARSE_SCHEMA_VERSION,
                       "variant": SPARSE_VARIANT,
                       "retrieval": _retrieval_signature(),
                       "vocab": self.vocab, "idf": self.idf,
                       "avgdl": self.avgdl,
                       "fingerprint": self._fingerprint()}, fh,
                      allow_nan=False)

    @classmethod
    def load(cls, path: Path | None = None) -> "SparseEncoder":
        path = path or config.VOCAB_FILE
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError) as exc:
            raise SparseCompatibilityError(
                "Sparse vocabulary is missing or malformed; it must be published "
                "with the matching index revision") from exc
        if (not isinstance(raw, dict)
                or raw.get("schema_version") != SPARSE_SCHEMA_VERSION
                or raw.get("variant") != SPARSE_VARIANT
                or raw.get("retrieval") != _retrieval_signature()):
            raise SparseCompatibilityError(
                "Sparse vocabulary version/configuration does not match the active "
                "retrieval configuration; publish a matching index revision first")
        try:
            enc = cls()
            enc.vocab = {str(k): int(v) for k, v in raw["vocab"].items()}
            enc.idf = {str(k): float(v) for k, v in raw["idf"].items()}
            enc.avgdl = float(raw["avgdl"])
            enc._validate()
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise SparseCompatibilityError("Malformed sparse vocabulary artifact") from exc
        if raw.get("fingerprint") != enc._fingerprint():
            raise SparseCompatibilityError("Sparse vocabulary fingerprint mismatch")
        return enc
