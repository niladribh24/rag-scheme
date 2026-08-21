# Voice Service

Multilingual voice layer for Setu: speech-to-text (ASR) via **SraVaani-1.0** (IISc SPIRE Lab) and text-to-speech (TTS) via **Indic Parler-TTS** (AI4Bharat). Runs as its own component, separate from the main RAG backend (`api.py`), so the heavy `torch`/`transformers` stack doesn't bloat that service.

## Status

`main.py` runs a working, hardened FastAPI service with `/transcribe` and `/speak`, wired into `api.py` (as a proxy — see its `/transcribe` and `/speak` endpoints) and into the frontend (mic button in `setu-scheme-navigator.html`). Run this service alongside `api.py` for voice features to work; if it's down, `api.py`'s voice endpoints return a clean `503` and text-only Q&A keeps working.

- `main.py` — the service (`/health`, `/transcribe`, `/speak`).
- `test_transcribe.py` / `test_tts.py` — standalone one-off sanity scripts (predate `main.py`).
- `download_models.py` — one-time model download/cache script.

### Running it

```bash
.venv/bin/uvicorn voice_service.main:app --port 8001
```

Run alongside `api.py` (`uvicorn api:app --reload`, port 8000). `api.py` looks for the voice service at `http://localhost:8001` by default — override with a `VOICE_SERVICE_URL` env var if it's running elsewhere.

### Endpoints

- `POST /transcribe` — multipart form upload, field name `audio`. Any format `ffmpeg` can decode (webm/opus from browser `MediaRecorder`, wav, mp3, ogg, ...) is normalized to 16kHz mono WAV before transcription. Returns `{"text": "..."}`.
- `POST /speak` — JSON body `{"text": "...", "description": "..."}` (`description` optional, controls voice characteristics per Parler-TTS's prompt format). Returns a `audio/wav` stream. Language is inferred automatically from the script of `text` — no language field needed.
- `GET /health` — `{"status": "ok", "device": "cuda:0" | "cpu"}`.

Both endpoints validate input (empty/oversized uploads, empty/oversized text, undecodable audio) and return proper 4xx errors rather than crashing. GPU inference is serialized behind a lock, since both models share one GPU with limited VRAM and concurrent `.generate()` calls can race for memory.

### System requirement: ffmpeg

`/transcribe` shells out to `ffmpeg` to normalize incoming audio — this is a **system binary**, not a pip package. Install it via your OS package manager (e.g. `sudo dnf install ffmpeg` / `sudo apt install ffmpeg`) before running the service.

## Models

| | SraVaani-1.0 | Indic Parler-TTS |
|---|---|---|
| Task | ASR (speech → text) | TTS (text → speech) |
| Org | IISc SPIRE Lab / ARTPARK | AI4Bharat |
| Languages | 65 Indic languages/dialects | 21 Indic languages + English |
| Size | ~900MB (FP16) | ~2-3GB |
| HF repo | [`ARTPARK-IISc/SraVaani-1.0`](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) |
| Gating | Gated, **manual approval required** | Gated, auto-approved |

Since TTS only covers 21 of SraVaani's 65 languages, voice *replies* are scoped to that shared 21-language set — outside it, we fall back to text-only responses.

### Note: transformers version

`parler-tts` pins `transformers==4.46.1`, which downgrades from the `transformers 5.15.1` pulled in by `sentence-transformers` elsewhere in the project. Both models share the same `.venv` — this has been verified to work fine: SraVaani produces identical transcription output under 4.46.1 as under 5.15.1, and a Hindi TTS → ASR round trip (Indic Parler-TTS output fed back into SraVaani) transcribed correctly.

## Setup

From the repo root:

```bash
uv venv --python 3.12 .venv          # if not already created
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv -r voice_service/requirements.txt
```

### Hugging Face access

Both models are gated. You need your own HF account and access token:

1. Create a token at https://huggingface.co/settings/tokens (read access is enough).
2. Add it to `.env` in the repo root: `HF_TOKEN=hf_...`
3. **SraVaani** requires manual approval per account — visit https://huggingface.co/ARTPARK-IISc/SraVaani-1.0, request access, and wait for it to be granted before proceeding.
4. **Indic Parler-TTS** auto-approves — visiting https://huggingface.co/ai4bharat/indic-parler-tts and accepting the terms is enough.

### Download models (one-time)

```bash
.venv/bin/python voice_service/download_models.py
```

This downloads and caches both models to `~/.cache/huggingface/hub/` (not part of the repo). It exits with a clear error message if access hasn't been granted yet for either model. Expect this to take a while on first run (~3-4GB total download) — subsequent runs of the service reuse the cache and start fast.

## Hardware notes

GPU strongly recommended (both models will run on CPU but much slower). Indic Parler-TTS + SraVaani together need a few GB of VRAM — if you're also running other local models (e.g. Ollama), free that VRAM first.

## Test scripts

```bash
.venv/bin/python voice_service/test_transcribe.py   # ASR sanity check
.venv/bin/python voice_service/test_tts.py           # TTS sanity check, writes voice_service/test_samples/tts_*.wav
```
