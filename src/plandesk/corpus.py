"""Turn IRS publication PDFs into page-anchored chunks.

Each chunk keeps the publication, tax year and printed page so every answer
can cite something a reader can open and check.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

# file stem -> (publication label, tax year)
SOURCES = {
    "p590a--2024": ("590-A", 2024),
    "p590a--2025": ("590-A", 2025),
    "p590b--2024": ("590-B", 2024),
    "p590b--2025": ("590-B", 2025),
    "p560--2024": ("560", 2024),
    "p560--2025": ("560", 2025),
}

TITLES = {
    "590-A": "Contributions to Individual Retirement Arrangements (IRAs)",
    "590-B": "Distributions from Individual Retirement Arrangements (IRAs)",
    "560": "Retirement Plans for Small Business (SEP, SIMPLE, and Qualified Plans)",
}

_PROOF_LINES = (
    re.compile(r"^Page \d+ of \d+\s+Fileid:"),
    re.compile(r"^The type and rule above prints on all proofs"),
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    publication: str
    tax_year: int
    page: int
    text: str

    def citation(self) -> str:
        return f"Pub. {self.publication} ({self.tax_year}), PDF p. {self.page}"


def clean_page(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not any(p.match(ln) for p in _PROOF_LINES)]
    joined = "\n".join(lines)
    # Re-join words hyphenated across line breaks ("contribu-\ntion").
    joined = re.sub(r"(\w)-\n(\w)", r"\1\2", joined)
    joined = re.sub(r"\s*\n\s*", " ", joined)
    return re.sub(r"\s{2,}", " ", joined).strip()


def split_text(text: str, size: int = 1100, overlap: int = 200) -> list[str]:
    """Split on sentence boundaries into windows of roughly `size` characters."""
    sentences = re.split(r"(?<=[.;:])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
            # start the carried-over tail at a word boundary
            current = current[current.find(" ") + 1 :] if " " in current else current
        current = f"{current} {sentence}" if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_chunks(raw_dir: Path = RAW_DIR) -> list[Chunk]:
    from pypdf import PdfReader

    out: list[Chunk] = []
    for stem, (pub, year) in SOURCES.items():
        reader = PdfReader(str(raw_dir / f"{stem}.pdf"))
        for page_no, page in enumerate(reader.pages, start=1):
            text = clean_page(page.extract_text() or "")
            if len(text) < 80:
                continue
            for i, piece in enumerate(split_text(text)):
                cid = f"{pub.replace('-', '')}-{year}-p{page_no}-c{i}"
                out.append(Chunk(cid, pub, year, page_no, piece))
    return out


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    with path.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    with path.open() as fh:
        return [Chunk(**json.loads(line)) for line in fh]
