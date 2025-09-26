# app/ingest.py
from pathlib import Path
import json
import pandas as pd
import nltk

def load_articles(path: str | Path) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for ticker, items in raw.items():
        for idx, art in enumerate(items):
            rows.append({
                "id": f"{ticker}_{idx}",
                "ticker": art.get("ticker", ticker),
                "title": art["title"].strip(),
                "link": art.get("link", ""),
                "full_text": art.get("full_text", "").strip(),
            })
    df = pd.DataFrame(rows)
    df["text"] = (df["title"].fillna("") + ". " + df["full_text"].fillna("")).str.strip()
    df["sentences"] = df["text"].apply(lambda t: nltk.sent_tokenize(t) if t else [])
    return df