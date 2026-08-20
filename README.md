# Setu — AI Scheme Navigator

RAG-powered chatbot that helps citizens discover government schemes they're eligible for. Uses ChromaDB for vector search and Groq for LLM answers.

## Setup

```bash
python -m venv venv
venv\Scripts\Activate
pip install fastapi uvicorn chromadb sentence-transformers groq
```

## Run

```bash
set GROQ_API_KEY=your-groq-api-key
uvicorn api:app --reload
```

Open `http://localhost:8000` in your browser.

## Project Structure

```
├── api.py                       # FastAPI backend (RAG + /ask endpoint)
├── setu-scheme-navigator.html   # Frontend UI
├── generate_answer.py           # Standalone test script
├── data/                        # Scheme text files + source PDFs
└── scripts/                     # Utility scripts (extract, index, query test)
```
