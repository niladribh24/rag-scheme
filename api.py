from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Setu — AI Scheme Navigator API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Build index once at startup
client_db = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
files = [BASE_DIR / "data" / "pmmy_categories.txt", BASE_DIR / "data" / "pmmy_myscheme.txt"]
all_text = ""
for f in files:
    with open(f, "r", encoding="utf-8") as file:
        all_text += file.read() + "\n\n"
raw_chunks = [c.strip() for c in all_text.split("\n\n") if len(c.strip()) > 50]
collection = client_db.create_collection(name="mudra_scheme_docs", embedding_function=embedding_fn)
collection.add(documents=raw_chunks, ids=[f"chunk_{i}" for i in range(len(raw_chunks))])

groq_client = Groq()

class Query(BaseModel):
    question: str

@app.get("/")
def serve_frontend():
    """Serve the frontend HTML at the root URL."""
    return FileResponse(BASE_DIR / "setu-scheme-navigator.html", media_type="text/html")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "chunks_indexed": len(raw_chunks)}

@app.post("/ask")
def ask(q: Query):
    results = collection.query(query_texts=[q.question], n_results=2)
    retrieved_text = "\n\n".join(results["documents"][0])
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""You are a helpful assistant for a government scheme navigator. 
Answer the user's question using ONLY the information below. Be clear and conversational.

CONTEXT:
{retrieved_text}

USER QUESTION: {q.question}
"""
        }]
    )
    return {"answer": response.choices[0].message.content}