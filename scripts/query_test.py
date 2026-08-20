import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# NOTE: this creates a fresh empty DB each time (in-memory), 
# so we need to rebuild the index in this same script for now
files = ["pmmy_categories.txt", "pmmy_myscheme.txt"]
all_text = ""
for f in files:
    with open(f, "r", encoding="utf-8") as file:
        all_text += file.read() + "\n\n"

raw_chunks = [c.strip() for c in all_text.split("\n\n") if len(c.strip()) > 50]

collection = client.create_collection(
    name="mudra_scheme_docs",
    embedding_function=embedding_fn
)
collection.add(
    documents=raw_chunks,
    ids=[f"chunk_{i}" for i in range(len(raw_chunks))]
)

# Now the actual test: ask a question
query = "I need a loan of 3 lakh rupees for my small shop, which category do I fall under?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print("QUERY:", query)
print("\nTOP MATCHING CHUNKS:\n")
for i, doc in enumerate(results["documents"][0]):
    print(f"--- Match {i+1} ---")
    print(doc[:400])
    print()