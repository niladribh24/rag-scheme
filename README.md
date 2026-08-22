# Setu — AI Scheme Navigator

RAG-powered chatbot that helps citizens discover government schemes they're eligible for. Uses ChromaDB (multilingual embeddings) for vector search and Groq for LLM answers. An optional `voice_service/` microservice adds speech-in/speech-out across English, Hindi, Kannada, and Telugu.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv -r voice_service/requirements.txt   # only needed for voice features
```

Copy `.env.example` to `.env` and fill in:

- `GROQ_API_KEY` — required, for `api.py`. Get one at https://console.groq.com/keys.
- `HF_TOKEN` — only needed to run `voice_service` (its models are gated on Hugging Face). See `voice_service/README.md` for the access request steps.
- `VOICE_SERVICE_URL` — where `api.py` looks for `voice_service`. Defaults to `http://localhost:8001`; only set it if running voice_service elsewhere.

`ffmpeg` (system package, not pip) is also required if you're running `voice_service` — it normalizes recorded audio before transcription.

## Run

Two independent services. The main app works fine with just the first one running — voice features degrade gracefully (clean `503`s) if `voice_service` isn't up.

```bash
# Terminal 1 — main RAG backend + frontend, port 8000
.venv/bin/uvicorn api:app --reload

# Terminal 2 — voice service (speech-to-text / text-to-speech), port 8001 — optional
.venv/bin/uvicorn voice_service.main:app --port 8001
```

Open `http://localhost:8000` in your browser — `api.py` serves the frontend directly and proxies `/transcribe` and `/speak` through to `voice_service`.

First run of `voice_service` needs its gated Hugging Face models downloaded and cached — see `voice_service/README.md` for the one-time `download_models.py` step and access-request instructions before starting it.

## Project Structure

```
├── api.py                       # FastAPI backend (RAG + /ask, plus /transcribe and /speak proxies)
├── setu-scheme-navigator.html   # Frontend UI (mic input, markdown-rendered replies, 4 languages)
├── generate_answer.py           # Standalone test script
├── data/                        # Scheme text files + source PDFs
├── scripts/                     # Utility scripts (extract, index, query test)
└── voice_service/                # Separate FastAPI microservice: ASR (SraVaani) + TTS (edge-tts / Parler-TTS)
```

See `voice_service/README.md` for voice-service-specific details (models, VRAM handling, TTS fallback behavior).
