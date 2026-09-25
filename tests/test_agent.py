"""Tests for the LangGraph agent: router classification (scheme vs. direct
vs. citizen-prerequisite questions), the grade/retry loop's MAX_RETRIES
handling, and multi-turn query rewriting.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pytest
from dotenv import load_dotenv

# agent.py loads .env itself on import, but we need GROQ_API_KEY in os.environ
# *before* that import to decide whether to skip this module at all.
load_dotenv(BASE_DIR / ".env")

if not os.environ.get("GROQ_API_KEY"):
    # agent.py instantiates ChatGroq at import time, which fails outright
    # without a key — skip the whole module rather than erroring on collection.
    pytest.skip("GROQ_API_KEY not configured — skipping agent tests.", allow_module_level=True)

import agent as agent_module


# ---------------------------------------------------------------------------
# Fully mocked: the retry-loop fix, verified without any network/LLM calls.
# ---------------------------------------------------------------------------

def test_grade_retry_loop_allows_max_retries_re_retrievals(monkeypatch):
    """route_after_grade/grade must allow up to MAX_RETRIES re-retrievals
    before falling back to the cautious generator — previously it stopped
    one re-retrieval short because route_after_grade re-checked the
    post-increment `retries` count instead of the pre-increment decision."""
    call_log = []

    def fake_invoke(prompt):
        if "relevance grader" in prompt:
            call_log.append("grade")
            return SimpleNamespace(content="not_relevant")
        call_log.append("rewrite")
        return SimpleNamespace(content="expanded english query")

    monkeypatch.setattr(agent_module, "llm", SimpleNamespace(invoke=fake_invoke))

    state = agent_module._initial_state("some obscure question", [], "en")
    state["retrieved_text"] = "irrelevant text"

    re_retrieval_count = 0
    while True:
        state = agent_module.grade(state)
        if not agent_module.route_after_grade(state) == "retrieve":
            break
        re_retrieval_count += 1
        state["retrieved_text"] = "still irrelevant"

    assert re_retrieval_count == agent_module.MAX_RETRIES
    assert state["retries"] == agent_module.MAX_RETRIES
    assert state["relevance"] == "not_relevant"
    assert call_log.count("grade") == agent_module.MAX_RETRIES + 1  # initial + retries
    assert call_log.count("rewrite") == agent_module.MAX_RETRIES


def test_grade_stops_immediately_once_relevant(monkeypatch):
    monkeypatch.setattr(
        agent_module, "llm", SimpleNamespace(invoke=lambda p: SimpleNamespace(content="relevant"))
    )
    state = agent_module._initial_state("a good question", [], "en")
    state["retrieved_text"] = "on-topic context"
    state = agent_module.grade(state)
    assert agent_module.route_after_grade(state) == "generate"
    assert state["retries"] == 0


# ---------------------------------------------------------------------------
# Live-LLM tests: router classification & cross-lingual query translation.
# These hit the real Groq API (cheap, short completions) and are skipped if
# the call fails, e.g. no network access in this environment.
# ---------------------------------------------------------------------------

def _invoke_router(question, history=None, language="en"):
    state = agent_module._initial_state(question, history or [], language)
    try:
        return agent_module.router(state)
    except Exception as e:
        pytest.skip(f"Live Groq call failed (no network access?): {e}")


def test_router_classifies_scheme_question_as_retrieve():
    state = _invoke_router("What is the loan limit under PM SVANidhi?")
    assert state["route"] == "retrieve"


def test_router_classifies_greeting_as_direct():
    state = _invoke_router("Hi, how are you today?")
    assert state["route"] == "direct"


@pytest.mark.parametrize("question", [
    "How do I open a Jan Dhan bank account?",
    "How do I get an income certificate?",
    "How do I update my Aadhaar card?",
])
def test_router_classifies_prerequisites_as_retrieve(question):
    """Citizen prerequisite questions are directly relevant to scheme
    eligibility/applications and must route to retrieval, not be dismissed
    as off-topic small talk."""
    state = _invoke_router(question)
    assert state["route"] == "retrieve"


def test_router_produces_english_retrieval_query_for_hindi():
    state = _invoke_router("पीएम किसान योजना के लिए मैं कैसे आवेदन करूं?", language="hi")
    assert state["retrieval_query"]
    ascii_ratio = sum(ch.isascii() for ch in state["retrieval_query"]) / len(state["retrieval_query"])
    assert ascii_ratio > 0.8, "Expected an English-translated retrieval_query for a Hindi question"


def test_multiturn_query_rewriting_includes_prior_scheme_context():
    """A follow-up question referencing 'it' should be rewritten into a
    standalone question that names the scheme discussed earlier."""
    history = [
        {"sender": "user", "text": "Tell me about PM SVANidhi"},
        {"sender": "bot", "text": "PM SVANidhi offers collateral-free loans to street vendors."},
    ]
    state = _invoke_router("What is the interest rate for it?", history=history)
    assert "svanidhi" in state["rewritten_query"].lower()
