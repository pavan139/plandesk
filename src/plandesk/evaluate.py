"""Score answers against a hand-built question set.

Metrics (all deterministic, no LLM judge):
- correct: every expected value appears in the answer (numbers compared
  without "$" and commas, so "$7,000" matches "7000" but not "17,000")
- supported: at least one *verified* citation whose chunk also contains every
  expected value, i.e. the cited evidence actually states the answer
- abstention precision/recall on questions the corpus cannot answer
- tokens and latency per query
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from .corpus import ROOT, load_chunks

QUESTIONS = ROOT / "evals" / "questions.jsonl"
RESULTS = ROOT / "evals" / "results"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _flat(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3})", "", text.replace("$", "")).lower()


def contains(text: str, expected: str) -> bool:
    if expected.startswith("re:"):
        return re.search(expected[3:], text, re.I) is not None
    target = _flat(expected)
    if re.fullmatch(r"[\d.%]+", target):
        return re.search(rf"(?<![\d.]){re.escape(target)}(?![\d])", _flat(text)) is not None
    return target in _flat(text)


def score(item: dict, result: dict, chunk_text: dict[str, str]) -> dict:
    should_abstain = bool(item.get("abstain"))
    abstained = bool(result["abstained"])
    row = {"id": item["id"], "category": item["category"], "should_abstain": should_abstain,
           "abstained": abstained}
    if should_abstain:
        row["correct"] = abstained
        row["supported"] = None
    else:
        expected = item["expected"]
        row["correct"] = (not abstained) and all(contains(result["answer"], e) for e in expected)
        verified = [c for c in result["citations"] if c["verified"]]
        row["supported"] = (not abstained) and any(
            all(contains(chunk_text.get(c["id"], ""), e) for e in expected) for c in verified
        )
    return row


def summarize(rows: list[dict], results: list[dict]) -> dict:
    answerable = [r for r in rows if not r["should_abstain"]]
    unanswerable = [r for r in rows if r["should_abstain"]]
    abst = [r for r in rows if r["abstained"]]
    lat = [r["latency_s"] for r in results if r["latency_s"] > 0]
    toks = [r["input_tokens"] + r["output_tokens"] for r in results if r["input_tokens"]]

    def pct(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(bool(r["correct"]))

    return {
        "n": len(rows),
        "answer_accuracy_pct": pct([r["correct"] for r in answerable]),
        "supported_by_citation_pct": pct([bool(r["supported"]) for r in answerable]),
        "false_abstention_pct": pct([r["abstained"] for r in answerable]),
        "abstention_recall_pct": pct([r["abstained"] for r in unanswerable]),
        "abstention_precision_pct": pct([r["should_abstain"] for r in abst]),
        "overall_accuracy_pct": pct([r["correct"] for r in rows]),
        "accuracy_by_category_pct": {k: pct(v) for k, v in sorted(by_cat.items())},
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p95_s": round(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)], 2) if lat else None,
        "tokens_per_query_mean": round(statistics.mean(toks)) if toks else None,
        "tokens_total": sum(toks),
    }


def run(model: str, mode: str, use_embeddings: bool = True, out_dir: Path = RESULTS) -> dict:
    from openai import OpenAI

    from .answer import answer
    from .retrieve import Retriever

    client = OpenAI()
    chunks = load_chunks()
    retriever = None
    if mode != "closed_book":
        retriever = Retriever(chunks, use_embeddings=use_embeddings, client=client)
    chunk_text = {c.chunk_id: c.text for c in chunks}

    questions = load_questions()
    results, rows = [], []
    for item in questions:
        res = answer(item["question"], retriever, client, model=model, mode=mode).to_dict()
        results.append(res)
        rows.append(score(item, res, chunk_text))

    summary = {"model": model, "mode": mode, **summarize(rows, results)}
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{mode}__{model}"
    (out_dir / f"{name}.json").write_text(
        json.dumps({"summary": summary, "rows": rows, "results": results}, indent=2)
    )
    return summary


def rescore(path: Path) -> dict:
    """Re-score saved results without calling any API (used in CI)."""
    data = json.loads(path.read_text())
    chunk_text = {c.chunk_id: c.text for c in load_chunks()}
    items = {q["id"]: q for q in load_questions()}
    rows = [score(items[r["id"]], res, chunk_text) for r, res in zip(data["rows"], data["results"])]
    return {"model": data["summary"]["model"], "mode": data["summary"]["mode"],
            **summarize(rows, data["results"])}
