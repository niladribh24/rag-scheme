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

## Automated Scheme Ingestion & Categorization

To expand the dataset without manual data entry, Setu includes an automated fetcher (`scripts/fetch_schemes.py`) that pulls official Central and State government schemes directly from [MyScheme.gov.in](https://www.myscheme.gov.in/).

It normalizes eligibility rules, benefits, application steps, and document checklists, saves them into dedicated category subfolders in `data/`, and automatically rebuilds the ChromaDB vector index.

```bash
# List all 10+ supported scheme categories & folders
python scripts/fetch_schemes.py --list-categories

# Fetch 5 schemes under Agriculture into data/agriculture/
python scripts/fetch_schemes.py --category agriculture --count 5

# Fetch 5 schemes under Business/MSME into data/business_and_msme/
python scripts/fetch_schemes.py --category business --count 5

# Fetch 3 schemes from EVERY category in one command
python scripts/fetch_schemes.py --category all --count 3

# Fetch a single scheme by its official slug
python scripts/fetch_schemes.py --slug pm-kisan

# Search & fetch schemes by keyword query
python scripts/fetch_schemes.py --query "solar subsidy" --count 3
```

> **Note:** The fetcher automatically detects existing schemes across all subfolders to prevent duplicates.

## Project Structure

```
├── api.py                       # FastAPI backend (RAG + /ask, plus /transcribe and /speak proxies)
├── chat.html / index.html       # Frontend UI (mic input, markdown-rendered replies, multi-language)
├── rag_core.py                  # Core RAG pipeline (recursive loader, E5 embeddings, ChromaDB)
├── data/                        # Categorized scheme repository
│   ├── agriculture/             # Agriculture, Rural & Farming schemes
│   ├── business_and_msme/       # MSME, Startups & Entrepreneurship
│   ├── education_and_learning/  # Scholarships & Student Welfare
│   ├── women_and_child/         # Women Empowerment & Maternity
│   ├── health_and_wellness/     # Healthcare & Insurance
│   ├── housing_and_shelter/     # Housing & Urban/Rural Shelter
│   └── social_welfare/          # Pensions, Minorities & Disability
├── scripts/                     # Utility scripts
│   ├── fetch_schemes.py         # Automated MyScheme fetcher & categorizer
│   ├── build_index.py           # Force rebuild ChromaDB vector store
│   └── query_test.py            # CLI query retrieval tester
└── voice_service/               # Separate FastAPI microservice: ASR + TTS
```

See `voice_service/README.md` for voice-service-specific details (models, VRAM handling, TTS fallback behavior).
