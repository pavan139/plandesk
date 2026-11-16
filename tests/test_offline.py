"""Offline tests: no API key, no network. They run in CI."""

import json
from pathlib import Path

import pytest

from plandesk.answer import verify_citations
from plandesk.corpus import Chunk, clean_page, load_chunks, split_text
from plandesk.evaluate import RESULTS, contains, load_questions, rescore, score
from plandesk.retrieve import BM25, Retriever, years_in


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def retriever(chunks):
    return Retriever(chunks, use_embeddings=False)


def test_contains_matches_money_without_partial_numbers():
    assert contains("The limit is $7,000.", "$7,000")
    assert contains("limit: 7000 dollars", "$7,000")
    assert not contains("The limit is $17,000.", "$7,000")
    assert contains("a 6% excise tax", "6%")
    assert not contains("a 16% tax", "6%")
    assert contains("by the filing deadline", "re:(filing deadline|due date)")


def test_clean_page_drops_proof_lines_and_rejoins_hyphens():
    raw = (
        "Page 3 of 62  Fileid: something\n"
        "The type and rule above prints on all proofs including x.\n"
        "an excess contribu-\ntion is taxed"
    )
    assert clean_page(raw) == "an excess contribution is taxed"


def test_split_text_respects_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_text(text, size=300, overlap=50)
    assert len(parts) > 5
    assert all(len(p) <= 400 for p in parts)


def test_years_in():
    assert years_in("limits for 2024 and 2026?") == [2024, 2026]
    assert years_in("no year here") == []


def test_bm25_prefers_matching_document():
    bm = BM25(["roth ira income limits", "sep employer contribution", "rmd age"])
    assert bm.scores("employer SEP contribution").argmax() == 1


def test_year_filter_selects_matching_edition(retriever):
    r = retriever.search("What is the maximum annual exclusion for QCDs for 2024?")
    assert r.year_filter == 2024
    assert all(c.tax_year == 2024 for c in r.chunks)
    assert any("$105,000" in c.text for c in r.chunks)


def test_next_year_question_uses_newest_edition(retriever):
    r = retriever.search("What is the SIMPLE IRA salary reduction limit for 2026?")
    assert r.year_filter == 2025


def test_past_year_condition_is_not_out_of_coverage(retriever):
    r = retriever.search("If I reach age 72 after December 31, 2022, when do RMDs start?")
    assert r.out_of_coverage_years == []
    assert r.year_filter is None


def test_future_year_is_out_of_coverage(retriever):
    assert retriever.search("401(k) limit for 2028?").out_of_coverage_years == [2028]


def test_verify_citations_rejects_paraphrase_and_unknown_ids():
    chunk = Chunk("x", "590-B", 2025, 14, "The maximum annual exclusion for QCDs is $108,000.")
    cites = verify_citations(
        [
            {"id": "x", "quote": "The maximum annual exclusion for QCDs is $108,000."},
            {"id": "x", "quote": "QCDs are capped at one hundred eight thousand dollars"},
            {"id": "missing", "quote": "The maximum annual exclusion for QCDs is $108,000."},
            {"id": "[x]", "quote": "The maximum annual exclusion for QCDs is $108,000."},
        ],
        {"x": chunk},
    )
    assert [c.verified for c in cites] == [True, False, False, True]


def test_supported_requires_evidence_that_states_the_answer():
    item = {"id": "q", "category": "limit", "expected": ["$108,000"]}
    chunk_text = {"good": "exclusion is $108,000", "bad": "exclusion is $105,000"}
    base = {"answer": "It is $108,000.", "abstained": False}
    good = score(item, {**base, "citations": [{"id": "good", "verified": True}]}, chunk_text)
    wrong = score(item, {**base, "citations": [{"id": "bad", "verified": True}]}, chunk_text)
    assert good["correct"] and good["supported"]
    assert wrong["correct"] and not wrong["supported"]


def test_question_set_is_well_formed():
    qs = load_questions()
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q.get("abstain") or q.get("expected")


@pytest.mark.parametrize("path", sorted(RESULTS.glob("*.json")), ids=lambda p: p.stem)
def test_recorded_results_rescore_to_saved_summary(path: Path):
    """Release gate: the committed results must reproduce their own summary."""
    saved = json.loads(path.read_text())["summary"]
    fresh = rescore(path)
    for key in ("answer_accuracy_pct", "supported_by_citation_pct", "abstention_recall_pct"):
        assert fresh[key] == saved[key]


def test_full_pipeline_meets_release_floor():
    path = RESULTS / "full__gpt-5.4-mini.json"
    if not path.exists():
        pytest.skip("no recorded run")
    s = rescore(path)
    assert s["answer_accuracy_pct"] >= 90
    assert s["abstention_recall_pct"] == 100
