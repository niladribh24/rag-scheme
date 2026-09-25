"""Tests for the hybrid (dense + BM25) retrieval pipeline in rag_core.py
and the scheme alias cache in rag_tools.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import rag_core
import rag_tools


@pytest.fixture(scope="module")
def collection():
    coll = rag_core.build_vector_store()
    if coll.count() == 0:
        pytest.skip("No documents indexed in data/ — nothing to retrieve.")
    return coll


# ---------------------------------------------------------------------------
# Hybrid dense + BM25 retrieval
# ---------------------------------------------------------------------------

def test_query_vector_store_returns_cited_chunks(collection):
    """A scheme-related query should retrieve non-empty chunks carrying the
    citation header (scheme name) needed for the generator to cite sources."""
    result = rag_core.query_vector_store(collection, "What is PM-KISAN?", n_results=3)
    assert result, "Expected non-empty retrieval result"
    assert "[Scheme:" in result


def test_bm25_search_returns_a_list(collection):
    """BM25 keyword search should run cleanly and return a list of chunk ids
    (possibly empty, depending on what's currently ingested)."""
    rag_core.invalidate_bm25_cache()
    ids = rag_core._bm25_search(collection, "PMEGP loan subsidy", top_k=5)
    assert isinstance(ids, list)


def test_bm25_falls_back_cleanly_while_building(collection, monkeypatch):
    """If another thread is mid-build, _bm25_search must return [] instead
    of blocking or raising, so query_vector_store can fall back to pure
    vector ranking."""
    rag_core.invalidate_bm25_cache()
    rag_core._bm25_state["building"] = True
    try:
        ids = rag_core._bm25_search(collection, "PM-KISAN", top_k=5)
        assert ids == []
    finally:
        rag_core._bm25_state["building"] = False


def test_query_vector_store_still_works_without_bm25(collection):
    """query_vector_store must not error out when BM25 is unavailable —
    it should fall back to the dense-only ranking."""
    rag_core.invalidate_bm25_cache()
    rag_core._bm25_state["building"] = True
    try:
        result = rag_core.query_vector_store(collection, "farmer income support", n_results=2)
        assert isinstance(result, str)
    finally:
        rag_core._bm25_state["building"] = False
        rag_core.invalidate_bm25_cache()


def test_reciprocal_rank_fusion_combines_rankings():
    fused = rag_core._reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert set(fused) == {"a", "b", "c", "d"}
    # "a" and "b" each appear near the top of both lists, so both should
    # outrank "c" and "d", which only appear once.
    assert fused.index("a") < fused.index("c")
    assert fused.index("b") < fused.index("d")


def test_bm25_cache_invalidation_clears_state(collection):
    rag_core._ensure_bm25_index(collection)
    rag_core.invalidate_bm25_cache()
    assert rag_core._bm25_state["index"] is None
    assert rag_core._bm25_state["ids"] is None
    assert rag_core._bm25_state["documents"] is None
    assert rag_core._bm25_state["building"] is False


# ---------------------------------------------------------------------------
# Scheme alias cache (rag_tools.py)
# ---------------------------------------------------------------------------

def test_scheme_alias_cache_is_populated_and_reused():
    rag_tools.invalidate_scheme_cache()
    assert rag_tools._alias_cache is None

    first = rag_tools._get_scheme_aliases()
    assert rag_tools._alias_cache is first  # now cached in module state

    second = rag_tools._get_scheme_aliases()
    assert second is first  # same object: no rebuild happened on 2nd call


def test_scheme_alias_cache_invalidation_forces_rebuild():
    rag_tools._get_scheme_aliases()  # ensure populated
    rag_tools.invalidate_scheme_cache()
    assert rag_tools._alias_cache is None

    rebuilt = rag_tools._get_scheme_aliases()
    assert rag_tools._alias_cache is rebuilt


# ---------------------------------------------------------------------------
# Robust document parsing fallbacks
# ---------------------------------------------------------------------------

def _parse_in_tmp_data_dir(tmp_path, filename: str, content: str) -> list[dict]:
    scheme_dir = tmp_path / "general"
    scheme_dir.mkdir(exist_ok=True)
    f = scheme_dir / filename
    f.write_text(content, encoding="utf-8")

    original_data_dir = rag_core.DATA_DIR
    rag_core.DATA_DIR = tmp_path
    try:
        return rag_core._parse_file(f)
    finally:
        rag_core.DATA_DIR = original_data_dir


def test_markdown_header_fallback_when_no_section_delimiters(tmp_path):
    """A document with no `===` delimiters but with Markdown headers should
    split on those headers instead of collapsing into one oversized chunk."""
    content = (
        "Test Scheme Without Delimiters\n"
        "Level: National\n"
        "Category: Test\n\n"
        "## Eligibility\n"
        + ("Eligibility details. " * 40) + "\n\n"
        "## Benefits\n"
        + ("Benefit details. " * 40) + "\n"
    )
    chunks = _parse_in_tmp_data_dir(tmp_path, "test_scheme.txt", content)

    assert len(chunks) >= 2
    titles = {c["metadata"]["section_title"] for c in chunks}
    assert "Eligibility" in titles
    assert "Benefits" in titles
    for c in chunks:
        assert c["metadata"]["scheme_name"] == "Test Scheme Without Delimiters"
        assert c["metadata"]["level"] == "National"


def test_paragraph_block_fallback_when_no_headers_either(tmp_path):
    """A document with neither `===` delimiters nor Markdown headers should
    fall back to paragraph-block chunking rather than becoming one giant chunk."""
    paragraphs = [f"Paragraph {i}: " + ("filler text " * 30) for i in range(6)]
    content = "Test Scheme Plain Text\nLevel: National\n\n" + "\n\n".join(paragraphs)

    chunks = _parse_in_tmp_data_dir(tmp_path, "plain_scheme.txt", content)

    assert len(chunks) > 1, "Expected paragraph-block fallback to produce multiple chunks"
    for c in chunks:
        # Some slack over MAX_CHUNK_LENGTH since a single paragraph can push
        # a chunk slightly over before the next paragraph starts a new one.
        assert len(c["text"]) <= rag_core.MAX_CHUNK_LENGTH + 400


def test_standard_section_delimited_parsing_still_works(tmp_path):
    """Regression check: the original `===` section format must still parse
    exactly as before after the _parse_header early-stop fix."""
    content = (
        "Standard Scheme\n"
        "Level: National\n"
        "Category: Agriculture\n"
        "Key Benefit: Direct cash transfer\n\n"
        "===================================================\n"
        "ELIGIBILITY CRITERIA\n"
        "===================================================\n\n"
        "All small and marginal farmers with cultivable land are eligible, subject to state-level verification.\n\n"
        "===================================================\n"
        "SCHEME DETAILS & BENEFITS\n"
        "===================================================\n\n"
        "Rs 6000 per year is paid in three equal installments directly to the beneficiary's bank account.\n"
    )
    chunks = _parse_in_tmp_data_dir(tmp_path, "standard_scheme.txt", content)

    assert len(chunks) == 2
    titles = [c["metadata"]["section_title"] for c in chunks]
    assert "ELIGIBILITY CRITERIA" in titles
    assert "SCHEME DETAILS & BENEFITS" in titles
    for c in chunks:
        assert c["metadata"]["category"] == "Agriculture"
        assert c["metadata"]["key_benefit"] == "Direct cash transfer"
