"""LangChain tool wrappers for the RAG pipeline."""

from langchain_core.tools import tool
from rag_core import build_vector_store, query_vector_store

# Load the persisted collection once at import time
_collection = build_vector_store()


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
        result = search_govt_schemes.invoke(q)
        print(f"Result ({len(result)} chars):\n{result[:300]}...")
