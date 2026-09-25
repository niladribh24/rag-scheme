"""Shared RAG pipeline: load → chunk → embed → query."""

import logging
import re
import threading
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = str(BASE_DIR / "chroma_db")

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "scheme_docs_e5"
MAX_CHUNK_LENGTH = 1100
MIN_CHUNK_LENGTH = 50
RRF_K = 60  # standard reciprocal-rank-fusion damping constant

logger = logging.getLogger(__name__)

_SECTION_DELIM_RE = re.compile(r"^={3,}\s*$")
_SUBSECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")
_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_HEADER_FIELDS = {"Level": "level", "Category": "category", "Key Benefit": "key_benefit"}

# Lazy-loaded model singleton
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model (cached after first call)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _embed(texts: list[str], prefix: str, show_progress: bool = False) -> list[list[float]]:
    """Embed texts with the given E5 prefix ('passage: ' or 'query: ')."""
    model = _get_model()
    prefixed = [f"{prefix}{t}" for t in texts]
    return model.encode(
        prefixed,
        batch_size=32,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    ).tolist()


def _get_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DIR)


# ---------------------------------------------------------------------------
# Structural document parsing
# ---------------------------------------------------------------------------

def _parse_header(lines: list[str]) -> tuple[dict, int]:
    """Parse the header block at the top of a scheme file.

    Returns (metadata dict, index of first body line).
    """
    metadata: dict[str, str] = {}
    if not lines:
        return metadata, 0

    metadata["scheme_name"] = lines[0].strip()
    body_start = 1

    for i in range(1, len(lines)):
        line = lines[i].strip()
        if _SECTION_DELIM_RE.match(line):
            body_start = i
            break
        if not line:
            body_start = i + 1
            continue
        matched_field = False
        for prefix, key in _HEADER_FIELDS.items():
            if line.startswith(f"{prefix}:"):
                metadata[key] = line[len(prefix) + 1 :].strip()
                matched_field = True
                break
        if not matched_field:
            # First line that's neither blank, a section delimiter, nor a
            # recognized "Field: value" header — the header block is over.
            # Without this, a document with no `===` delimiters anywhere
            # would have every line swallowed as "header", leaving nothing
            # for the Markdown-header/paragraph fallback splitters below.
            body_start = i
            break
        body_start = i + 1

    return metadata, body_start


def _split_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split body lines on ``===`` delimited section titles.

    Returns a list of ``(section_title, content_lines)`` tuples.
    """
    sections: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(lines):
        if _SECTION_DELIM_RE.match(lines[i].strip()):
            if i + 2 < len(lines) and _SECTION_DELIM_RE.match(lines[i + 2].strip()):
                title = lines[i + 1].strip()
                i += 3
                content: list[str] = []
                while i < len(lines) and not _SECTION_DELIM_RE.match(lines[i].strip()):
                    content.append(lines[i])
                    i += 1
                sections.append((title, content))
            else:
                i += 1
        else:
            i += 1
    return sections


def _split_markdown_headers(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Fallback split: break body lines on Markdown ATX headers (#, ##, ###...).

    Used when a document has no ``===`` section delimiters at all, so it
    doesn't collapse into a single oversized chunk. Returns ``[]`` (rather
    than a single untitled section) when no header actually matched, so the
    caller can fall through to the paragraph-block fallback instead.
    """
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    found_header = False
    for line in lines:
        m = _MD_HEADER_RE.match(line.strip())
        if m:
            # Flush whatever came before this header — including any
            # leading intro text before the very first header — as its own
            # section rather than silently dropping it.
            sections.append((current_title, current_lines))
            current_title = m.group(2).strip()
            current_lines = []
            found_header = True
        else:
            current_lines.append(line)
    if not found_header:
        return []
    sections.append((current_title, current_lines))
    return sections


def _split_paragraph_blocks(
    lines: list[str], max_length: int = MAX_CHUNK_LENGTH
) -> list[tuple[str, list[str]]]:
    """Last-resort fallback: group double-newline-separated paragraphs into
    chunks that stay under *max_length*, for documents with neither ``===``
    delimiters nor Markdown headers.
    """
    text = "\n".join(lines)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    blocks: list[tuple[str, list[str]]] = []
    current_parts: list[str] = []
    current_length = 0
    for para in paragraphs:
        if current_length + len(para) > max_length and current_parts:
            blocks.append(("", "\n\n".join(current_parts).split("\n")))
            current_parts = []
            current_length = 0
        current_parts.append(para)
        current_length += len(para)
    if current_parts:
        blocks.append(("", "\n\n".join(current_parts).split("\n")))
    return blocks


def _chunk_section(
    section_title: str,
    content_lines: list[str],
    max_length: int = MAX_CHUNK_LENGTH,
) -> list[dict]:
    """Chunk a single section, splitting at subsection boundaries if too long.

    Returns a list of ``{"section_title": ..., "text": ...}`` dicts.
    """
    full_text = "\n".join(line.rstrip() for line in content_lines).strip()

    # Short enough → keep as one chunk
    if len(full_text) <= max_length:
        return [{"section_title": section_title, "text": full_text}]

    # Find numbered-subsection boundary lines (e.g. "5.1 Working Capital …")
    boundaries = [
        i for i, line in enumerate(content_lines) if _SUBSECTION_RE.match(line.strip())
    ]

    if not boundaries:
        # No subsection markers to split on → return as one (oversized) chunk
        return [{"section_title": section_title, "text": full_text}]

    # Build ordered text blocks: pre-subsection intro, then each subsection
    blocks: list[tuple[str, str, str]] = []  # (num, title, text)

    if boundaries[0] > 0:
        pre_text = "\n".join(
            line.rstrip() for line in content_lines[: boundaries[0]]
        ).strip()
        if pre_text:
            blocks.append(("", "", pre_text))

    for j, start in enumerate(boundaries):
        end = boundaries[j + 1] if j + 1 < len(boundaries) else len(content_lines)
        m = _SUBSECTION_RE.match(content_lines[start].strip())
        num, title = m.group(1), m.group(2)
        block_text = "\n".join(
            line.rstrip() for line in content_lines[start:end]
        ).strip()
        blocks.append((num, title, block_text))

    # Group blocks into chunks that stay under max_length
    chunks: list[dict] = []
    current_parts: list[str] = []
    current_length = 0

    for _num, _title, text in blocks:
        if current_length + len(text) > max_length and current_parts:
            chunk_text = f"{section_title}\n\n" + "\n\n".join(current_parts)
            chunks.append({"section_title": section_title, "text": chunk_text})
            current_parts = []
            current_length = 0
        current_parts.append(text)
        current_length += len(text)

    if current_parts:
        chunk_text = f"{section_title}\n\n" + "\n\n".join(current_parts)
        chunks.append({"section_title": section_title, "text": chunk_text})

    return chunks


def _parse_file(filepath: Path) -> list[dict]:
    """Parse one ``.txt`` scheme file into structured chunks.

    Returns a list of ``{"text": ..., "metadata": {...}}`` dicts.
    Falls back to treating the whole file as one chunk if the expected
    header or section delimiters are not found.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    metadata, body_start = _parse_header(lines)

    try:
        source_rel = str(filepath.relative_to(DATA_DIR)).replace("\\", "/")
    except ValueError:
        source_rel = filepath.name

    if not metadata.get("scheme_name"):
        logger.warning(
            "No scheme name found in %s, treating as single chunk", filepath.name
        )
        return [{"text": content.strip(), "metadata": {"source_file": source_rel}}]

    body_lines = lines[body_start:]
    sections = _split_sections(body_lines)

    if not sections:
        logger.info(
            "No === sections found in %s, falling back to Markdown headers", filepath.name
        )
        sections = _split_markdown_headers(body_lines)

    if not sections:
        logger.info(
            "No Markdown headers found in %s, falling back to paragraph blocks", filepath.name
        )
        sections = _split_paragraph_blocks(body_lines)

    if not sections:
        logger.warning(
            "No structure found in %s, treating as single chunk", filepath.name
        )
        return [
            {
                "text": content.strip(),
                "metadata": {**metadata, "source_file": source_rel},
            }
        ]

    chunks: list[dict] = []
    for section_title, section_lines in sections:
        for sc in _chunk_section(section_title, section_lines):
            if len(sc["text"]) >= MIN_CHUNK_LENGTH:
                chunk_meta = {
                    **metadata,
                    "section_title": sc["section_title"],
                    "source_file": source_rel,
                }
                chunks.append({"text": sc["text"], "metadata": chunk_meta})

    return chunks


def load_and_chunk_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load all .txt files in *data_dir* (including subfolders) and return structured chunks.

    Each element is ``{"text": str, "metadata": dict}``.
    """
    chunks: list[dict] = []
    for f in sorted(data_dir.rglob("*.txt")):
        chunks.extend(_parse_file(f))
    return chunks


# ---------------------------------------------------------------------------
# BM25 keyword index (kept alongside the dense vector index for hybrid search)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_bm25_lock = threading.Lock()
_bm25_state: dict = {
    "index": None,
    "ids": None,
    "documents": None,
    "metadatas": None,
    "building": False,
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def invalidate_bm25_cache() -> None:
    """Clear the cached BM25 index so the next query rebuilds it from scratch.

    Call this whenever the vector index is rebuilt (e.g. after /reindex) so
    keyword search reflects newly added or removed documents.
    """
    with _bm25_lock:
        _bm25_state["index"] = None
        _bm25_state["ids"] = None
        _bm25_state["documents"] = None
        _bm25_state["metadatas"] = None
        _bm25_state["building"] = False


def _ensure_bm25_index(collection: chromadb.Collection) -> dict:
    """Return the cached BM25 state, building it on first use.

    If another thread is already building the index, returns immediately
    with ``index=None`` so callers can fall back to pure vector search
    instead of blocking.
    """
    if _bm25_state["index"] is not None or _bm25_state["building"]:
        return _bm25_state
    with _bm25_lock:
        if _bm25_state["index"] is not None or _bm25_state["building"]:
            return _bm25_state
        _bm25_state["building"] = True
    try:
        data = collection.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        tokenized = [_tokenize(doc) for doc in documents]
        index = BM25Okapi(tokenized) if tokenized else None
        _bm25_state["ids"] = data.get("ids") or []
        _bm25_state["documents"] = documents
        _bm25_state["metadatas"] = data.get("metadatas") or []
        _bm25_state["index"] = index
    except Exception:
        logger.exception("Failed to build BM25 index; falling back to vector-only search")
        _bm25_state["ids"] = None
        _bm25_state["documents"] = None
        _bm25_state["metadatas"] = None
        _bm25_state["index"] = None
    finally:
        _bm25_state["building"] = False
    return _bm25_state


def _bm25_search(
    collection: chromadb.Collection,
    query: str,
    top_k: int,
    scheme_filter: str | list[str] | None = None,
) -> list[str]:
    """Return the top-k chunk ids ranked by BM25, optionally restricted to
    *scheme_filter*. Returns an empty list if the index is unavailable or
    still (re)building — callers should fall back cleanly to vector-only
    results in that case.
    """
    state = _ensure_bm25_index(collection)
    if state["building"] or state["index"] is None:
        return []

    allowed = None
    if scheme_filter:
        allowed = set(scheme_filter) if isinstance(scheme_filter, list) else {scheme_filter}

    scores = state["index"].get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results: list[str] = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            break
        if allowed is not None:
            scheme_name = state["metadatas"][idx].get("scheme_name")
            if scheme_name not in allowed:
                continue
        results.append(state["ids"][idx])
        if len(results) >= top_k:
            break
    return results


def _reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Combine several ranked id lists into one ranking via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


# ---------------------------------------------------------------------------
# Vector store operations
# ---------------------------------------------------------------------------

def build_vector_store() -> chromadb.Collection:
    """Return the persisted collection, building it from data/ if it doesn't exist yet."""
    client_db = _get_client()
    existing = client_db.list_collections()
    if COLLECTION_NAME in [c.name for c in existing]:
        return client_db.get_collection(name=COLLECTION_NAME)
    # First run — build from scratch
    print("[INFO] Initializing ChromaDB vector collection...")
    chunks = load_and_chunk_documents()
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    print(f"[INFO] Loaded {len(texts)} chunks across data/ repository.")
    print(f"[INFO] Computing neural embeddings (model: {EMBEDDING_MODEL})...")
    collection = client_db.create_collection(name=COLLECTION_NAME)
    doc_embeddings = _embed(texts, "passage: ", show_progress=True)
    collection.add(
        documents=texts,
        embeddings=doc_embeddings,
        metadatas=metadatas,
        ids=[f"chunk_{i}" for i in range(len(texts))],
    )
    print(f"[SUCCESS] Index built successfully with {collection.count()} chunks.")
    return collection


def force_rebuild_index() -> chromadb.Collection:
    """Delete the existing collection (if any) and rebuild from data/."""
    client_db = _get_client()
    existing = client_db.list_collections()
    if COLLECTION_NAME in [c.name for c in existing]:
        client_db.delete_collection(name=COLLECTION_NAME)
    chunks = load_and_chunk_documents()
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    print(f"[INFO] Loaded {len(texts)} document chunks from data/ (including subfolders).")
    print(f"[INFO] Computing neural embeddings with {EMBEDDING_MODEL} (CPU/GPU)...")
    collection = client_db.create_collection(name=COLLECTION_NAME)
    doc_embeddings = _embed(texts, "passage: ", show_progress=True)
    collection.add(
        documents=texts,
        embeddings=doc_embeddings,
        metadatas=metadatas,
        ids=[f"chunk_{i}" for i in range(len(texts))],
    )
    invalidate_bm25_cache()
    print(f"[SUCCESS] Vector store rebuilt! Total chunks indexed: {collection.count()}")
    return collection


def get_index_stats(collection: chromadb.Collection) -> dict:
    """Return dynamic stats about the indexed corpus for the /health endpoint."""
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    schemes = {m.get("scheme_name") for m in metadatas if m.get("scheme_name")}
    categories = {m.get("category") for m in metadatas if m.get("category")}
    return {
        "total_chunks": collection.count(),
        "unique_schemes": len(schemes),
        "categories": sorted(categories),
    }


def _format_citation_header(meta: dict) -> str:
    """Build a bracketed metadata header carrying enough context for the
    generator to produce verifiable ``[Scheme: <Name> | Section: ...]`` citations."""
    parts = [f"Scheme: {meta.get('scheme_name', 'Unknown')}"]
    if meta.get("level"):
        parts.append(f"Level: {meta['level']}")
    if meta.get("category"):
        parts.append(f"Category: {meta['category']}")
    if meta.get("section_title"):
        parts.append(f"Section: {meta['section_title']}")
    if meta.get("source_file"):
        parts.append(f"Source: {meta['source_file']}")
    return "[" + " | ".join(parts) + "]"


def query_vector_store(
    collection: chromadb.Collection, question: str, n_results: int = 2,
    scheme_filter: str | list[str] | None = None,
) -> str:
    """Hybrid dense + BM25 retrieval, fused via Reciprocal Rank Fusion (RRF).

    Dense (multilingual-e5) search captures semantic similarity; BM25 catches
    exact-token matches that embeddings can blur — scheme acronyms
    (PM-KISAN, PMEGP, MUDRA), specific loan amounts, and age thresholds.
    Falls back cleanly to pure vector ranking if the BM25 index is
    unavailable or still (re)building.
    """
    fetch_k = max(n_results * 4, 10)

    query_embedding = _embed([question], "query: ")
    kwargs: dict = {"query_embeddings": query_embedding, "n_results": fetch_k}
    if scheme_filter:
        if isinstance(scheme_filter, list):
            kwargs["where"] = {"scheme_name": {"$in": scheme_filter}}
        else:
            kwargs["where"] = {"scheme_name": scheme_filter}
    dense_results = collection.query(**kwargs, include=["documents", "metadatas"])

    dense_ids = dense_results["ids"][0]
    lookup: dict[str, tuple[str, dict]] = {
        doc_id: (doc, meta)
        for doc_id, doc, meta in zip(
            dense_ids, dense_results["documents"][0], dense_results["metadatas"][0]
        )
    }

    bm25_ids = _bm25_search(collection, question, top_k=fetch_k, scheme_filter=scheme_filter)
    fused_ids = _reciprocal_rank_fusion([dense_ids, bm25_ids]) if bm25_ids else dense_ids
    top_ids = fused_ids[:n_results]

    # A BM25-only hit won't be in `lookup` yet (it wasn't in the dense fetch) —
    # backfill its text/metadata from the BM25 index's own cached corpus.
    missing = [doc_id for doc_id in top_ids if doc_id not in lookup]
    if missing and _bm25_state.get("ids"):
        state_lookup = {
            i: (d, m)
            for i, d, m in zip(
                _bm25_state["ids"], _bm25_state["documents"] or [], _bm25_state["metadatas"] or []
            )
        }
        for doc_id in missing:
            if doc_id in state_lookup:
                lookup[doc_id] = state_lookup[doc_id]

    chunks = []
    for doc_id in top_ids:
        if doc_id not in lookup:
            continue
        doc, meta = lookup[doc_id]
        print(f"    [retrieval] chunk id={doc_id} scheme_name={meta.get('scheme_name', 'Unknown')}")
        chunks.append(f"{_format_citation_header(meta)}\n{doc}")
    return "\n\n".join(chunks)
