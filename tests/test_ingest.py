import pytest
import nltk
from app.ingest import load_articles

@pytest.fixture(scope="session", autouse=True)
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

def test_load_articles_columns():
    df = load_articles("data/stock_news.json")
    assert {"id","ticker","title","link","full_text","text","sentences"}.issubset(df.columns)
    assert len(df) > 0