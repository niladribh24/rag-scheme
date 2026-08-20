from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent

client_db = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
files = [BASE_DIR / "data" / "pmmy_categories.txt", BASE_DIR / "data" / "pmmy_myscheme.txt"]
all_text = ""
for f in files:
    with open(f, "r", encoding="utf-8") as file:
        all_text += file.read() + "\n\n"
raw_chunks = [c.strip() for c in all_text.split("\n\n") if len(c.strip()) > 50]

collection = client_db.create_collection(
    name="mudra_scheme_docs",
    embedding_function=embedding_fn
)
collection.add(documents=raw_chunks, ids=[f"chunk_{i}" for i in range(len(raw_chunks))])

query = "I need a loan of 3 lakh rupees for my small shop, which category do I fall under?"
results = collection.query(query_texts=[query], n_results=2)
retrieved_text = "\n\n".join(results["documents"][0])

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