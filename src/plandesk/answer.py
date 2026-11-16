"""Answer generation with citations, a grounding check and abstention.

The model must return JSON with quoted evidence. Every quote is checked
against the cited chunk; an answer with no verifiable quote is converted to
an abstention instead of being shown to the user.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from .corpus import Chunk
from .retrieve import Retriever

# USD per 1M tokens (input, output). Left empty on purpose: fill in current
# provider prices to get dollar estimates. Reports always include raw token
# counts, which do not go stale.
PRICES: dict[str, tuple[float, float]] = {}

SYSTEM = """You answer questions about U.S. retirement-plan rules using ONLY the numbered IRS publication excerpts provided.

Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge.
- Pay attention to tax years. If the question names a year, use the amount stated for that year.
- Support every factual claim with a citation: the excerpt id and a short verbatim quote (copy the exact words, 5 to 40 words) that contains the fact.
- If the excerpts do not contain the answer, set "abstain" to true and explain what is missing. Do not guess.
- Never give personalized investment, legal or tax advice.

Return JSON only:
{"answer": string, "abstain": boolean, "citations": [{"id": string, "quote": string}]}"""

CLOSED_BOOK_SYSTEM = """You answer questions about U.S. retirement-plan rules from your own knowledge.
If you are not confident, set "abstain" to true. Never give personalized advice.
Return JSON only: {"answer": string, "abstain": boolean, "citations": []}"""


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"-\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Citation:
    id: str
    quote: str
    verified: bool
    source: str = ""


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    citations: list[Citation] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    year_filter: int | None = None
    model: str = ""
    mode: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    guard: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def verify_citations(raw: list[dict], by_id: dict[str, Chunk]) -> list[Citation]:
    out = []
    for c in raw or []:
        cid = str(c.get("id", "")).strip().strip("[]").strip()
        quote = str(c.get("quote", "")).strip()
        chunk = by_id.get(cid)
        ok = bool(chunk and quote and len(quote) >= 12 and _norm(quote) in _norm(chunk.text))
        out.append(Citation(cid, quote, ok, chunk.citation() if chunk else ""))
    return out


def _call(client, model: str, system: str, user: str) -> tuple[dict, int, int, float]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    latency = time.perf_counter() - t0
    usage = resp.usage
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        data = {"answer": "", "abstain": True, "citations": []}
    return data, usage.prompt_tokens, usage.completion_tokens, latency


def _cost(model: str, tin: int, tout: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (tin * pin + tout * pout) / 1_000_000


def answer(
    question: str,
    retriever: Retriever | None,
    client,
    model: str = "gpt-5.4-mini",
    mode: str = "full",
    k: int = 6,
) -> Answer:
    """mode: 'closed_book' (no retrieval), 'naive_rag' (no year filter, no
    grounding guard) or 'full' (year-aware retrieval + grounding guard)."""
    if mode == "closed_book":
        data, tin, tout, lat = _call(client, model, CLOSED_BOOK_SYSTEM, question)
        return Answer(
            question, str(data.get("answer", "")), bool(data.get("abstain")), [], [], None,
            model, mode, tin, tout, lat, _cost(model, tin, tout),
        )

    year_aware = mode == "full"
    r = retriever.search(question, k=k, year_aware=year_aware)

    if year_aware and r.out_of_coverage_years:
        years = ", ".join(map(str, r.out_of_coverage_years))
        return Answer(
            question,
            f"The indexed publications cover tax years 2024 and 2025 (with some 2023 and 2026 "
            f"amounts). They don't state rules for {years}, so I can't answer that.",
            True, [], [c.chunk_id for c in r.chunks], r.year_filter, model, mode,
            guard="year_out_of_coverage",
        )

    by_id = {c.chunk_id: c for c in r.chunks}
    context = "\n\n".join(f"[{c.chunk_id}] ({c.citation()})\n{c.text}" for c in r.chunks)
    user = f"Excerpts:\n{context}\n\nQuestion: {question}"
    data, tin, tout, lat = _call(client, model, SYSTEM, user)

    cites = verify_citations(data.get("citations", []), by_id)
    abstained = bool(data.get("abstain"))
    text = str(data.get("answer", ""))
    guard = ""
    if mode == "full" and not abstained and not any(c.verified for c in cites):
        guard = "no_verified_citation"
        abstained = True
        text = "I couldn't verify a supporting quote in the publications, so I won't answer."

    return Answer(
        question, text, abstained, cites, [c.chunk_id for c in r.chunks], r.year_filter,
        model, mode, tin, tout, lat, _cost(model, tin, tout), guard,
    )
