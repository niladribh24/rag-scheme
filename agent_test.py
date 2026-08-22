"""Test the LangGraph agent with 4 diverse queries."""

from agent import agent

TEST_QUERIES = [
    "what's the weather today",
    "PM SVANidhi loan tranches",
    "help me with money",
    "मुझे अपनी दुकान के लिए 3 लाख का लोन चाहिए",
]


def run_test(question: str):
    print(f"\n{'='*70}")
    print(f"QUERY: {question}")
    print("-" * 70)
    result = agent.invoke({
        "question": question,
        "rewritten_query": "",
        "retrieved_text": "",
        "route": "",
        "relevance": "",
        "retries": 0,
        "answer": "",
    })
    print("-" * 70)
    print(f"FINAL ANSWER:\n{result['answer']}")
    print(f"\nRetries used: {result['retries']}")


if __name__ == "__main__":
    for q in TEST_QUERIES:
        run_test(q)
