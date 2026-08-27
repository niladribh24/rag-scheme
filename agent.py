"""LangGraph agent with router → retrieve → grade → generate workflow."""

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from rag_tools import search_govt_schemes, detect_scheme, _collection
from rag_core import query_vector_store

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── LLM (same model as api.py / generate_answer.py) ──────────────────────
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=1024)

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
    """Call query_vector_store with optional scheme-name filtering."""
    query = state.get("rewritten_query") or state["question"]
    schemes = detect_scheme(query) or detect_scheme(state["question"])

    if len(schemes) == 1:
        scheme_filter = schemes[0]
        n_results = 4
        print(f"  [retrieve] single scheme filter: '{scheme_filter}'")
    elif len(schemes) >= 2:
        scheme_filter = schemes
        n_results = 6
        print(f"  [retrieve] multi-scheme filter: {schemes}")
    else:
        scheme_filter = None
        n_results = 10
        print(f"  [retrieve] general discovery query (no filter, n_results=10)")

    print(f"  [retrieve] searching with query='{query}'")
    retrieved_text = query_vector_store(
        _collection, query, n_results=n_results, scheme_filter=scheme_filter,
    )
    print(f"  [retrieve] got {len(retrieved_text)} chars of context")
    return {**state, "retrieved_text": retrieved_text}


# ── Node: grade ───────────────────────────────────────────────────────────
def grade(state: AgentState) -> AgentState:
    """Judge whether retrieved text is relevant enough to answer the question."""
    question = state["question"]
    retrieved_text = state["retrieved_text"]

    schemes_detected = detect_scheme(question)
    if schemes_detected:
        scheme_instruction = (
            f"The user specifically asked about: {', '.join(schemes_detected)}. "
            "Check if the retrieved context contains relevant details about at least one of these specific schemes."
        )
    else:
        scheme_instruction = (
            "The user is asking a general or recommendation question about schemes, loans, or subsidies. "
            "The retrieved context may contain chunks from multiple DIFFERENT schemes. This is completely NORMAL and EXPECTED. "
            "Mark as 'relevant' if any of the retrieved chunks provide information on government schemes, loans, subsidies, or eligibility that could help answer the user's intent."
        )

    response = llm.invoke(
        f"""You are a relevance grader for an Indian government financial scheme assistant.
Given the user question and the retrieved context below, decide if the context contains enough information to answer the question meaningfully.

GUIDELINES:
{scheme_instruction}

Reply with EXACTLY one word: "relevant" or "not_relevant".

User question: {question}

Retrieved context:
{retrieved_text[:2500]}"""
    )
    relevance = response.content.strip().lower()
    if relevance not in ("relevant", "not_relevant"):
        relevance = "relevant"  # default to proceeding

    retries = state.get("retries", 0)
    rewritten_query = state.get("rewritten_query", "")

    if relevance == "not_relevant" and retries < MAX_RETRIES:
        # Rewrite the query for a retry with domain expansion
        rewrite_response = llm.invoke(
            f"""The user asked: "{question}"
The initial vector search returned insufficient results. Rewrite this user question into an expanded search query targeting Indian government financial scheme documents.
Include relevant domain terms, business category synonyms (e.g. street vendor, micro credit, working capital, small enterprise, collateral free loan, PM SVANidhi, PMEGP, Stand-Up India, Udyogini), and key loan terms if applicable.
Return ONLY the expanded search query string, nothing else."""
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

    exhausted_retries = (
        state.get("retries", 0) >= MAX_RETRIES
        and state.get("relevance") == "not_relevant"
    )

    if exhausted_retries:
        print(f"  [generate] max retries exhausted & still not_relevant — cautious answer")
        response = llm.invoke(
            f"""You are a helpful assistant for a government scheme navigator 
called Setu. The user asked the question below, but after searching our 
database we could NOT find a clearly matching scheme.

Tell the user honestly that no closely matching scheme was found. If the 
retrieved context below contains anything loosely related, you may mention it 
as "possibly related" — but do NOT present it as a definitive answer. 
Suggest they check official portals (myscheme.gov.in) or contact their local 
Common Service Centre for accurate information.

CONTEXT:
{retrieved_text}

USER QUESTION: {question}

INSTRUCTION: Keep your response clear, structured, and under 300 words. Do NOT use tables. Always finish your last sentence completely."""
        )
    else:
        print(f"  [generate] producing grounded answer")
        response = llm.invoke(
            f"""You are a helpful assistant for a government scheme navigator. 
Answer using ONLY relevant information below — ignore any chunks that don't 
actually relate to the question. Be clear and conversational.

CRITICAL: Every specific number you state (loan amount, subsidy percentage, 
age limit, income limit, interest rate, etc.) MUST come from the text 
explicitly labeled with that scheme's name in the context below. If the 
context contains multiple [Scheme: ...] sections, NEVER attribute a number 
from one scheme's section to a different scheme. If you are unsure which 
scheme a number belongs to, do not state it — say the information wasn't 
found in that scheme's context instead.

CONTEXT:
{retrieved_text}

USER QUESTION: {question}

INSTRUCTION: Keep your response clear, well-structured, and concise (under 300 words). Do NOT use tables. Always finish your last sentence completely."""
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

User message: {question}

STRICT LIMIT: Your response MUST be between 200 and 250 words. Do NOT exceed 
250 words under any circumstances. Always finish your last sentence 
completely."""
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
