import os
from pathlib import Path
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from rag_core import build_vector_store
from agent import agent

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

VOICE_SERVICE_URL = os.environ.get("VOICE_SERVICE_URL", "http://localhost:8001")

app = FastAPI(title="Setu — AI Scheme Navigator API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Build index once at startup
collection = build_vector_store()



class Query(BaseModel):
    question: str

@app.get("/")
def serve_frontend():
    """Serve the frontend HTML at the root URL."""
    return FileResponse(BASE_DIR / "setu-scheme-navigator.html", media_type="text/html")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "chunks_indexed": collection.count()}

@app.post("/ask")
def ask(q: Query):
    result = agent.invoke({
        "question": q.question,
        "rewritten_query": "",
        "retrieved_text": "",
        "route": "",
        "relevance": "",
        "retries": 0,
        "answer": "",
    })
    return {"answer": result["answer"]}


class SpeakRequest(BaseModel):
    text: str
    description: str | None = None


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    """Proxy: forwards recorded audio to the voice service and returns its transcript."""
    raw = await audio.read()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
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
    return Response(content=resp.content, media_type="audio/wav")