"""
sentiment_service.py

Adds a local, transformer-based financial sentiment layer on top of the
existing Groq/Llama report pipeline.

Model: ProsusAI/finbert
  - A BERT model further pre-trained and fine-tuned by ProsusAI specifically
    on financial text (10-Ks, earnings calls, analyst reports) to classify
    sentiment as positive / negative / neutral.
  - Runs fully locally via Hugging Face `transformers` — no API key needed,
    no network dependency at inference time (after the one-time model
    download/cache), CPU is fine for single-document scoring.

This is intentionally kept separate from llm.py: the Groq/Llama model still
writes the full narrative report, this module gives you a fast, deterministic,
reproducible sentiment score to display alongside it.
"""

from functools import lru_cache
from typing import Dict, List

MODEL_NAME = "ProsusAI/finbert"

# FinBERT (like most BERT-family models) has a 512 token limit per input.
# We chunk long text and aggregate rather than silently truncating it.
MAX_CHARS_PER_CHUNK = 800


@lru_cache(maxsize=1)
def _get_pipeline():
    """
    Lazily load the FinBERT sentiment pipeline once per process and cache it.
    Import is deferred inside the function so the rest of the app can run
    even in environments where torch/transformers aren't installed yet —
    the ImportError only surfaces when sentiment analysis is actually used.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' and 'torch' packages are required for "
            "sentiment analysis. Install them with:\n"
            "  pip install transformers torch"
        ) from exc

    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=512,
    )


def _chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Split text into rough chunks on sentence boundaries to stay under FinBERT's token limit."""
    if not text:
        return []

    sentences = text.replace("\n", " ").split(". ")
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current}. {sentence}" if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current.strip())

    return chunks


def analyze_sentiment(text: str) -> Dict:
    """
    Run FinBERT sentiment analysis on a piece of financial text.

    Returns:
        {
            "label": "positive" | "negative" | "neutral",
            "score": float,               # confidence of the winning label, 0-1
            "breakdown": {                 # averaged probability per label across chunks
                "positive": float,
                "negative": float,
                "neutral": float,
            },
            "chunks_analyzed": int,
        }

    Returns a safe "neutral" / zero-confidence result on empty input or model
    load failure so callers (the Streamlit UI) never crash on this step.
    """
    if not text or not text.strip() or text.strip().upper() == "N/A":
        return {
            "label": "neutral",
            "score": 0.0,
            "breakdown": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            "chunks_analyzed": 0,
        }

    try:
        sentiment_pipeline = _get_pipeline()
    except ImportError:
        return {
            "label": "unavailable",
            "score": 0.0,
            "breakdown": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            "chunks_analyzed": 0,
        }

    chunks = _chunk_text(text)
    if not chunks:
        chunks = [text]

    totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    for chunk in chunks:
        result = sentiment_pipeline(chunk)[0]
        label = result["label"].lower()
        score = float(result["score"])
        if label in totals:
            totals[label] += score
        else:
            totals["neutral"] += score

    n = len(chunks)
    breakdown = {label: round(total / n, 4) for label, total in totals.items()}
    winning_label = max(breakdown, key=breakdown.get)

    return {
        "label": winning_label,
        "score": breakdown[winning_label],
        "breakdown": breakdown,
        "chunks_analyzed": n,
    }


def analyze_headlines(headlines: List[str]) -> Dict:
    """
    Convenience helper for scoring a list of short texts (e.g. news headlines)
    and returning an aggregated sentiment, in case a news source is wired in
    later (services/news_service.py is currently a stub).
    """
    if not headlines:
        return analyze_sentiment("")

    combined = ". ".join(h.strip() for h in headlines if h and h.strip())
    return analyze_sentiment(combined)
