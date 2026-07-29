"""
rag_service.py

Retrieval-Augmented Generation layer for FinSight AI.

Instead of letting the LLM freely recall (and potentially misstate) financial
concepts, ratio definitions, and sector context from its own parametric
knowledge, this module retrieves the most relevant passages from a small,
curated, static financial knowledge base (data/knowledge_base.json) and
injects them into the prompt sent to the report-generation model in llm.py.

Retrieval approach:
  - Primary: semantic search using sentence-transformers
    ('all-MiniLM-L6-v2') to embed both the knowledge base and the query,
    ranked by cosine similarity. Runs locally, no API key needed.
  - Fallback: if sentence-transformers/torch aren't installed, falls back to
    a lightweight keyword-overlap scorer so retrieval still works (with
    lower precision) without the extra dependency.

This module is intentionally decoupled from the sentiment_service.py model
(FinBERT) — they load independently and a failure in one does not affect
the other.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K_PER_QUERY = 2
MAX_TOTAL_DOCS = 6


@lru_cache(maxsize=1)
def _load_knowledge_base() -> List[Dict]:
    if not KB_PATH.exists():
        return []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _get_embedder():
    """
    Lazily load the sentence-transformers embedding model, cached for the
    life of the process. Returns None if the dependency isn't installed,
    signalling callers to use the keyword-overlap fallback instead.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_doc_embeddings():
    """
    Compute (and cache) embeddings for every document in the knowledge base.
    Only runs once per process. Returns None if embeddings aren't available.
    """
    embedder = _get_embedder()
    if embedder is None:
        return None

    docs = _load_knowledge_base()
    texts = [f"{d['title']}. {d['content']}" for d in docs]
    if not texts:
        return None

    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def _cosine_search(query: str, top_k: int) -> List[Tuple[Dict, float]]:
    """Semantic search via sentence-transformers embeddings + cosine similarity."""
    import numpy as np

    embedder = _get_embedder()
    doc_embeddings = _get_doc_embeddings()
    docs = _load_knowledge_base()

    if embedder is None or doc_embeddings is None or not docs:
        return []

    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = doc_embeddings @ query_embedding  # cosine similarity (vectors are normalized)

    ranked_idx = np.argsort(-scores)[:top_k]
    return [(docs[i], float(scores[i])) for i in ranked_idx]


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _keyword_search(query: str, top_k: int) -> List[Tuple[Dict, float]]:
    """
    Fallback retrieval when sentence-transformers isn't installed: scores
    documents by Jaccard overlap between query tokens and document tokens.
    Lower precision than embeddings, but dependency-free and keeps RAG
    functional in minimal environments.
    """
    docs = _load_knowledge_base()
    query_tokens = _tokenize(query)
    if not query_tokens or not docs:
        return []

    scored = []
    for doc in docs:
        doc_tokens = _tokenize(f"{doc['title']} {doc['content']} {doc['category']}")
        overlap = query_tokens & doc_tokens
        union = query_tokens | doc_tokens
        score = len(overlap) / len(union) if union else 0.0
        scored.append((doc, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def retrieve(query: str, top_k: int = DEFAULT_TOP_K_PER_QUERY) -> List[Dict]:
    """
    Retrieve the top_k most relevant knowledge base documents for a query.
    Each returned dict has: id, title, category, content, score.
    """
    results = _cosine_search(query, top_k)
    if not results:
        results = _keyword_search(query, top_k)

    output = []
    for doc, score in results:
        if score <= 0:
            continue
        output.append({**doc, "score": round(score, 4)})
    return output


def retrieve_for_report(info: Dict) -> List[Dict]:
    """
    Build a small set of targeted queries from the stock's profile (sector,
    industry, valuation shape) and retrieve relevant knowledge base context
    for each, then dedupe and cap the total so the prompt stays a
    reasonable size.
    """
    sector = info.get("Sector", "") or ""
    industry = info.get("Industry", "") or ""

    queries = [
        f"valuation ratios PE PEG EV EBITDA interpretation for {sector} {industry}",
        "profitability margins return on equity return on assets",
        "liquidity leverage debt to equity current ratio financial health",
        "free cash flow operating cash flow quality",
        "moving average technical analysis price trend momentum volume",
        f"{sector} {industry} sector specific valuation considerations and risk",
    ]

    seen_ids = set()
    combined: List[Dict] = []

    for query in queries:
        for doc in retrieve(query, top_k=DEFAULT_TOP_K_PER_QUERY):
            if doc["id"] in seen_ids:
                continue
            seen_ids.add(doc["id"])
            combined.append(doc)

    combined.sort(key=lambda d: d["score"], reverse=True)
    return combined[:MAX_TOTAL_DOCS]


def format_context(docs: List[Dict]) -> str:
    """Format retrieved documents into a prompt-ready context block."""
    if not docs:
        return "No additional reference material retrieved."

    lines = []
    for doc in docs:
        lines.append(f"[{doc['category'].upper()}] {doc['title']}: {doc['content']}")
    return "\n\n".join(lines)
