import json
import os
import sys
from pathlib import Path
import httpx

# Windows' default console codepage (cp1252) can't encode some characters an
# LLM response may contain (e.g. U+2011 non-breaking hyphen), which crashes
# any print()/logging call that hits stdout — reconfigure to UTF-8 up front.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_core import build_vector_store, force_rebuild_index, get_index_stats
from rag_tools import invalidate_scheme_cache
from agent import agent, stream_answer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

VOICE_SERVICE_URL = os.environ.get("VOICE_SERVICE_URL", "http://localhost:8001")

# CORS: local dev servers (Vite/CRA-style ports) plus the deployed frontend
# origin, when configured. `ALLOWED_ORIGINS` is a comma-separated list read
# from .env — falls back to common localhost origins if unset.
_default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()
]

app = FastAPI(title="Setu — AI Scheme Navigator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=BASE_DIR / "assets"), name="assets")

# Build index once at startup
collection = build_vector_store()


class Query(BaseModel):
    question: str
    history: list = []
    language: str = "en"


@app.get("/")
def serve_landing():
    """Serve the landing/about page HTML at the root URL."""
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")

@app.get("/chat")
def serve_chat():
    """Serve the chat interface HTML."""
    return FileResponse(BASE_DIR / "chat.html", media_type="text/html")

@app.get("/health")
def health_check():
    """Health check with dynamic corpus stats (total chunks, unique schemes, categories)."""
    stats = get_index_stats(collection)
    return {"status": "ok", "chunks_indexed": stats["total_chunks"], **stats}

@app.post("/reindex")
def reindex():
    """Force rebuild the vector store index from data/ directory."""
    global collection
    collection = force_rebuild_index()
    invalidate_scheme_cache()
    stats = get_index_stats(collection)
    return {"status": "success", "chunks_indexed": stats["total_chunks"], **stats}

@app.post("/ask")
def ask(q: Query):
    result = agent.invoke({
        "question": q.question,
        "chat_history": q.history,
        "language": q.language,
        "rewritten_query": "",
        "retrieval_query": "",
        "retrieved_text": "",
        "route": "",
        "relevance": "",
        "retries": 0,
        "should_retry": False,
        "answer": "",
    })
    return {"answer": result["answer"]}


@app.post("/ask/stream")
def ask_stream(q: Query):
    """SSE endpoint: streams the answer token-by-token as it's generated,
    instead of the client waiting 4-8s for the full non-streaming response."""
    def event_stream():
        try:
            for token in stream_answer(q.question, q.history, q.language):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class SpeakRequest(BaseModel):
    text: str
    description: str | None = None


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    """Proxy: forwards recorded audio to the voice service and returns its transcript."""
    raw = await audio.read()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{VOICE_SERVICE_URL}/transcribe",
                files={"audio": (audio.filename, raw, audio.content_type)},
            )
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Voice service is unreachable.")
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = "Transcription failed."
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()


@app.post("/speak")
async def speak(req: SpeakRequest):
    """Proxy: forwards text to the voice service and streams back synthesized audio."""
    body = {"text": req.text}
    if req.description:
        body["description"] = req.description
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{VOICE_SERVICE_URL}/speak", json=body)
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Voice service is unreachable.")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Speech synthesis failed.")
    return Response(content=resp.content, media_type=resp.headers.get("content-type", "audio/wav"))