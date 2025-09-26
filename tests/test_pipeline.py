import pytest
import nltk
from app.ingest import load_articles
from app.retriever import HybridRetriever
from app.pipeline import run_query

@pytest.fixture(scope="session", autouse=True)
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

def test_pipeline_end_to_end():
    df = load_articles("data/stock_news.json")
    r = HybridRetriever(df["text"].tolist())
    ans = run_query(df, r, "What happened with MSFT cloud?")
    assert isinstance(ans, list) and len(ans) > 0
    assert {"id","ticker","title","summary"}.issubset(ans[0].keys())