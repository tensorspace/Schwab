# app/pipeline.py
import re
from typing import Any
import pandas as pd
from .summarizer import summarize
from .retriever import HybridRetriever

TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")

def run_query(df: pd.DataFrame, retriever: HybridRetriever, query: str, k: int = 3, use_llm: bool = False) -> list[dict[str, Any]]:
    """Runs a query against the document set."""
    tickers = set(TICKER_RE.findall(query))

    # Retrieve from the full document set
    hits = retriever.retrieve(query, k=k * 3)  # Retrieve more to allow for filtering

    results = []
    seen_ids = set()

    for doc_idx, score in hits:
        if len(results) >= k:
            break

        row = df.iloc[doc_idx]
        # Filter by ticker if specified
        if tickers and row["ticker"] not in tickers:
            continue

        # Avoid duplicates
        if row["id"] in seen_ids:
            continue

        seen_ids.add(row["id"])

        summary = summarize(row["sentences"], llm=use_llm)
        results.append({
            "id": row["id"],
            "ticker": row["ticker"],
            "title": row["title"],
            "link": row["link"],
            "summary": summary,
            "score": score,
        })

    return results