"""Hybrid retrieval (BM25 + embeddings) with a tax-year filter.

The year filter is the "temporal" part: IRS limits change every year and the
2024 and 2025 publications often contain the same sentence with different
numbers ("The maximum annual exclusion for QCDs is $105,000" vs "$108,000").
A retriever that ignores the year can cite the right sentence from the wrong
publication.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .corpus import ROOT, Chunk

EMBED_MODEL = "text-embedding-3-small"
EMBED_PATH = ROOT / "data" / "embeddings.npy"
CORPUS_YEARS = (2024, 2025)
# Each publication also states next-year amounts, so the corpus can answer
# questions up to one year past its newest edition.
ANSWERABLE_YEARS = range(2023, 2027)

_TOKEN = re.compile(r"[a-z0-9]+(?:\(k\))?")
_STOP = set(
    "a an and are as at be by can do does for from how i if in is it my of on or the "
    "to was what when which who will with you your this that there".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avg = float(self.lens.mean())
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.tfs))
        for i, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


def years_in(question: str) -> list[int]:
    return sorted({int(y) for y in re.findall(r"\b(20\d\d)\b", question)})


@dataclass
class Retrieval:
    chunks: list[Chunk]
    year_filter: int | None
    out_of_coverage_years: list[int]


class Retriever:
    def __init__(self, chunks: list[Chunk], use_embeddings: bool = True, client=None):
        self.chunks = chunks
        self.bm25 = BM25([f"{c.publication} {c.text}" for c in chunks])
        self.client = client
        self.emb = None
        if use_embeddings:
            self.emb = self._load_or_build_embeddings()

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), 256):
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + 256])
            vecs.extend(d.embedding for d in resp.data)
        arr = np.array(vecs, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def _load_or_build_embeddings(self) -> np.ndarray:
        if EMBED_PATH.exists():
            arr = np.load(EMBED_PATH)
            if len(arr) == len(self.chunks):
                return arr
        arr = self._embed([c.text for c in self.chunks])
        np.save(EMBED_PATH, arr)
        return arr

    def search(self, question: str, k: int = 6, year_aware: bool = True) -> Retrieval:
        years = years_in(question)
        # Past years are often conditions ("if I reached 72 after 2022"), so
        # only years beyond what the corpus states count as out of coverage.
        outside = [y for y in years if y > max(ANSWERABLE_YEARS)]
        in_range = [y for y in years if y in ANSWERABLE_YEARS]
        year_filter = None
        if year_aware and in_range:
            # Prefer the edition for the asked year; a question about 2026
            # is best served by the newest edition, which states 2026 amounts.
            target = max(in_range)
            year_filter = min(max(target, CORPUS_YEARS[0]), CORPUS_YEARS[-1])

        idx = np.arange(len(self.chunks))
        if year_filter is not None:
            idx = np.array([i for i, c in enumerate(self.chunks) if c.tax_year == year_filter])

        bm = self.bm25.scores(question)[idx]
        ranked_lists = [idx[np.argsort(-bm)][:50]]
        if self.emb is not None:
            qv = self._embed([question])[0]
            dense = self.emb[idx] @ qv
            ranked_lists.append(idx[np.argsort(-dense)][:50])

        # Reciprocal rank fusion
        fused: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, i in enumerate(ranked):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return Retrieval([self.chunks[i] for i in top], year_filter, outside)


def default_retriever(use_embeddings: bool = True) -> Retriever:
    from .corpus import load_chunks

    client = None
    if use_embeddings:
        from openai import OpenAI

        client = OpenAI()
    return Retriever(load_chunks(), use_embeddings=use_embeddings, client=client)


__all__ = ["BM25", "Retriever", "Retrieval", "default_retriever", "years_in", "Path"]
