from groq import Groq
from rag_core import build_vector_store, query_vector_store

collection = build_vector_store()

query = "I need a loan of 3 lakh rupees for my small shop, which category do I fall under?"
retrieved_text = query_vector_store(collection, query)

client = Groq()

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    max_tokens=300,
    messages=[{
        "role": "user",
        "content": f"""You are a helpful assistant for a government scheme navigator. 
Answer the user's question using ONLY the information below. Be clear and conversational.

CONTEXT:
{retrieved_text}

USER QUESTION: {query}
"""
    }]
)

print("USER QUESTION:", query)
print("\nAI ANSWER:\n")
print(response.choices[0].message.content)