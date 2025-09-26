import pytest
import nltk
from app.ingest import load_articles
from app.retriever import HybridRetriever

@pytest.fixture(scope="session", autouse=True)
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

def test_retrieval_not_empty():
    df = load_articles("data/stock_news.json")
    r = HybridRetriever(df["text"].tolist())
    hits = r.retrieve("AAPL results beat", k=3)
    assert len(hits) > 0
    assert all(isinstance(h[0], int) for h in hits)