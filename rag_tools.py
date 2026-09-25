"""LangChain tool wrappers for the RAG pipeline."""

import re
import threading
from langchain_core.tools import tool
from rag_core import build_vector_store, query_vector_store

_alias_cache: dict[str, str] | None = None
_alias_cache_lock = threading.Lock()


def invalidate_scheme_cache() -> None:
    """Clear the cached scheme alias mapping.

    Call this after the vector index is rebuilt (e.g. from /reindex) so
    newly added or renamed schemes are picked up on the next lookup instead
    of the stale in-memory mapping.
    """
    global _alias_cache
    with _alias_cache_lock:
        _alias_cache = None


def _build_scheme_aliases() -> dict[str, str]:
    """Build a lookup of known scheme names from the active collection metadata."""
    collection = build_vector_store()
    try:
        metadatas = collection.get(include=["metadatas"])["metadatas"]
    except Exception:
        return {}
    scheme_names = sorted({m.get("scheme_name", "") for m in metadatas} - {""})

    # Aliases shorter than this are almost never a real scheme acronym and
    # are prone to false positives — e.g. a scheme name containing an
    # incidental aside like "[Product (or) Process]" would otherwise
    # register "or" as an alias, matching the substring "or" inside "for",
    # "work", "before", etc. in virtually any English sentence.
    MIN_ALIAS_LEN = 3

    aliases: dict[str, str] = {}
    for name in scheme_names:
        aliases[name.lower()] = name
        if "(" in name:
            short = name[: name.index("(")].strip().lower()
            if len(short) >= MIN_ALIAS_LEN:
                aliases[short] = name
        for abbr in re.findall(r"\(([^)]+)\)", name):
            abbr_clean = abbr.strip().lower()
            if len(abbr_clean) >= MIN_ALIAS_LEN:
                aliases[abbr_clean] = name
        cut = len(name)
        for delim in (" Scheme", " - "):
            pos = name.find(delim)
            if pos != -1:
                cut = min(cut, pos)
        if cut < len(name):
            natural = name[:cut].strip().lower()
            if len(natural) >= MIN_ALIAS_LEN and natural not in aliases:
                aliases[natural] = name
    return aliases


def _get_scheme_aliases() -> dict[str, str]:
    """Return the cached scheme alias mapping, building it on first use.

    Previously this ran ``collection.get(...)`` across the entire database
    on every call (i.e. on every ``detect_scheme()`` invocation). It's now
    computed once and cached in memory until ``invalidate_scheme_cache()``
    is called.
    """
    global _alias_cache
    if _alias_cache is None:
        with _alias_cache_lock:
            if _alias_cache is None:
                _alias_cache = _build_scheme_aliases()
    return _alias_cache


def detect_scheme(text: str) -> list[str]:
    """Return all matching scheme_names found in *text* (case-insensitive).

    Longer alias strings are checked first. Each scheme is returned at most
    once even if multiple aliases for it match. Normalizes hyphens/spaces.
    """
    text_lower = text.lower()
    text_normalized = re.sub(r"[\s\-_]+", " ", text_lower)
    found: list[str] = []
    aliases_map = _get_scheme_aliases()

    for alias, full_name in sorted(aliases_map.items(), key=lambda x: len(x[0]), reverse=True):
        alias_normalized = re.sub(r"[\s\-_]+", " ", alias)
        if alias_normalized in text_normalized or alias in text_lower:
            if full_name not in found:
                found.append(full_name)
    return found


@tool
def search_govt_schemes(query: str) -> str:
    """Search Indian government financial scheme documents for eligibility
    criteria, loan amounts, interest rates, and application procedures.
    Input should be a natural language question about schemes, loans, or
    financial assistance."""
    return query_vector_store(build_vector_store(), query, n_results=3)


if __name__ == "__main__":
    questions = [
        "What is the loan limit under PM SVANidhi?",
        "Am I eligible for Stand-Up India if I'm a woman?",
        "What documents do I need for the Udyogini scheme?",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        detected = detect_scheme(q)
        print(f"Detected scheme: {detected}")
        result = search_govt_schemes.invoke(q)
        print(f"Result ({len(result)} chars):\n{result[:300]}...")

