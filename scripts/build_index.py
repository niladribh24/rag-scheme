import chromadb
from chromadb.utils import embedding_functions

# Load all our text files
files = ["pmmy_categories.txt", "pmmy_myscheme.txt"]

all_text = ""
for f in files:
    with open(f, "r", encoding="utf-8") as file:
        all_text += file.read() + "\n\n"

# Simple chunking: split by paragraphs (double newlines), 
# then group small chunks together so they're not too tiny
raw_chunks = [c.strip() for c in all_text.split("\n\n") if len(c.strip()) > 50]

print(f"Created {len(raw_chunks)} chunks")
print("Example chunk:\n", raw_chunks[0][:300])

# Set up ChromaDB with a free local embedding model
client = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large"
)

collection = client.create_collection(
    name="mudra_scheme_docs",
    embedding_function=embedding_fn
)

# Add chunks to the vector database
collection.add(
    documents=raw_chunks,
    ids=[f"chunk_{i}" for i in range(len(raw_chunks))]
)

print("Index built successfully with", len(raw_chunks), "chunks")