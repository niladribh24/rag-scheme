"""LangGraph agent with router → retrieve → grade → generate workflow."""

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from rag_tools import search_govt_schemes

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── LLM (same model as api.py / generate_answer.py) ──────────────────────
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=400)

MAX_RETRIES = 2


# ── State ─────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    rewritten_query: str
    retrieved_text: str
    route: str            # "retrieve" | "direct"
    relevance: str        # "relevant" | "not_relevant"
    retries: int
    answer: str


# ── Node: router ──────────────────────────────────────────────────────────
def router(state: AgentState) -> AgentState:
    """Classify the question as scheme-related or off-topic."""
    question = state["question"]
    response = llm.invoke(
        f"""You are a classifier. Given the user question below, decide if it 
is about Indian government financial schemes, loans, subsidies, or related 
eligibility/application topics.

Reply with EXACTLY one word: "retrieve" if it is scheme-related, or "direct" 
if it is off-topic, a greeting, or unrelated to government schemes.

User question: {question}"""
    )
    route = response.content.strip().lower()
    if route not in ("retrieve", "direct"):
        route = "retrieve"  # default to retrieval if unsure
    print(f"  [router] question='{question}' → route={route}")
    return {**state, "route": route, "retries": state.get("retries", 0)}


# ── Node: retrieve ────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    """Call search_govt_schemes with the current query."""
    query = state.get("rewritten_query") or state["question"]
    print(f"  [retrieve] searching with query='{query}'")
    retrieved_text = search_govt_schemes.invoke(query)
    print(f"  [retrieve] got {len(retrieved_text)} chars of context")
    return {**state, "retrieved_text": retrieved_text}


# ── Node: grade ───────────────────────────────────────────────────────────
def grade(state: AgentState) -> AgentState:
    """Judge whether retrieved text is relevant enough to answer the question."""
    question = state["question"]
    retrieved_text = state["retrieved_text"]

    response = llm.invoke(
        f"""You are a relevance grader. Given the user question and the 
retrieved context below, decide if the context contains enough information 
to answer the question meaningfully.

Reply with EXACTLY one word: "relevant" or "not_relevant".

User question: {question}

Retrieved context:
{retrieved_text[:2000]}"""
    )
    relevance = response.content.strip().lower()
    if relevance not in ("relevant", "not_relevant"):
        relevance = "relevant"  # default to proceeding

    retries = state.get("retries", 0)
    rewritten_query = state.get("rewritten_query", "")

    if relevance == "not_relevant" and retries < MAX_RETRIES:
        # Rewrite the query for a retry
        rewrite_response = llm.invoke(
            f"""The user asked: "{question}"
The search returned irrelevant results. Rewrite this as a more specific 
search query about Indian government financial schemes. Return ONLY the 
rewritten query, nothing else."""
        )
        rewritten_query = rewrite_response.content.strip()
        retries += 1
        print(f"  [grade] relevance=not_relevant, retry {retries}/{MAX_RETRIES}, rewritten_query='{rewritten_query}'")
    else:
        print(f"  [grade] relevance={relevance}, retries={retries}")

    return {
        **state,
        "relevance": relevance,
        "retries": retries,
        "rewritten_query": rewritten_query,
    }


# ── Node: generate ────────────────────────────────────────────────────────
def generate(state: AgentState) -> AgentState:
    """Produce a grounded answer using the retrieved context."""
    question = state["question"]
    retrieved_text = state["retrieved_text"]
    print(f"  [generate] producing grounded answer")
    response = llm.invoke(
        f"""You are a helpful assistant for a government scheme navigator. 
Answer using ONLY relevant information below — ignore any chunks that don't 
actually relate to the question. Be clear and conversational.

CONTEXT:
{retrieved_text}

USER QUESTION: {question}"""
    )
    return {**state, "answer": response.content}


# ── Node: generate_direct ─────────────────────────────────────────────────
def generate_direct(state: AgentState) -> AgentState:
    """Respond directly without retrieval (off-topic / greetings)."""
    question = state["question"]
    print(f"  [generate_direct] responding without retrieval")
    response = llm.invoke(
        f"""You are a helpful assistant for a government scheme navigator 
called Setu. The user sent a message that is not about government schemes.
Respond politely and briefly, then remind them you can help with questions 
about Indian government financial schemes, loans, and subsidies.

User message: {question}"""
    )
    return {**state, "answer": response.content}


# ── Conditional edges ─────────────────────────────────────────────────────
def route_after_router(state: AgentState) -> str:
    """Route to retrieve or generate_direct based on router's decision."""
    return "retrieve" if state["route"] == "retrieve" else "generate_direct"


def route_after_grade(state: AgentState) -> str:
    """Route to generate, or back to retrieve for a retry."""
    if state["relevance"] == "not_relevant" and state["retries"] < MAX_RETRIES:
        # grade node already incremented retries, so if retries < MAX_RETRIES
        # it means the increment just happened and we should retry
        return "retrieve"
    return "generate"


# ── Build the graph ───────────────────────────────────────────────────────
def build_agent():
    """Construct and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)
    graph.add_node("generate_direct", generate_direct)

    # Edges
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_router)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade)
    graph.add_edge("generate", END)
    graph.add_edge("generate_direct", END)

    return graph.compile()


agent = build_agent()
