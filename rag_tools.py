"""LangChain tool wrappers for the RAG pipeline."""

import re
from langchain_core.tools import tool
from rag_core import build_vector_store, query_vector_store

# Load the persisted collection once at import time
_collection = build_vector_store()

# Build a lookup of known scheme names from the indexed metadata.
# Each entry maps a lowercase substring to the full scheme_name value.
_SCHEME_NAMES: list[str] = sorted(
    {m.get("scheme_name", "") for m in _collection.get(include=["metadatas"])["metadatas"]}
    - {""},
)
# Short aliases that appear in user queries → full scheme_name
_SCHEME_ALIASES: dict[str, str] = {}
for _name in _SCHEME_NAMES:
    _SCHEME_ALIASES[_name.lower()] = _name
    # Also register the short prefix before any parenthesis, e.g. "pm svanidhi"
    if "(" in _name:
        _short = _name[: _name.index("(")].strip().lower()
        _SCHEME_ALIASES[_short] = _name
    # Also register the abbreviation inside parentheses, e.g. "pmegp", "naps-2"
    for _abbr in re.findall(r"\(([^)]+)\)", _name):
        _SCHEME_ALIASES[_abbr.strip().lower()] = _name
    # Register a short natural alias: text before " Scheme" or " - "
    _cut = len(_name)
    for _delim in (" Scheme", " - "):
        _pos = _name.find(_delim)
        if _pos != -1:
            _cut = min(_cut, _pos)
    if _cut < len(_name):
        _natural = _name[:_cut].strip().lower()
        if _natural and _natural not in _SCHEME_ALIASES:
            _SCHEME_ALIASES[_natural] = _name


def detect_scheme(text: str) -> list[str]:
    """Return all matching scheme_names found in *text* (case-insensitive).

    Longer alias strings are checked first.  Each scheme is returned at most
    once even if multiple aliases for it match.
    """
    text_lower = text.lower()
    found: list[str] = []
    for alias in sorted(_SCHEME_ALIASES, key=len, reverse=True):
        if alias in text_lower:
            full_name = _SCHEME_ALIASES[alias]
            if full_name not in found:
                found.append(full_name)
    return found


@tool
def search_govt_schemes(query: str) -> str:
    """Search Indian government financial scheme documents for eligibility
    criteria, loan amounts, interest rates, and application procedures.
    Input should be a natural language question about schemes, loans, or
    financial assistance."""
    return query_vector_store(_collection, query, n_results=3)


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

