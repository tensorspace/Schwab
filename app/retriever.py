# app/retriever.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, docs: list[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=2)
        self.tfidf = self.vectorizer.fit_transform(docs)
        self.tokens = [d.split() for d in docs]
        self.bm25 = BM25Okapi(self.tokens)

    def retrieve(self, query: str, k: int = 5):
        if not query.strip():
            return []
        tfidf_q = self.vectorizer.transform([query])
        tfidf_scores = (tfidf_q @ self.tfidf.T).toarray().flatten()
        bm25_scores = np.array(self.bm25.get_scores(query.split()))
        # z-normalize safely
        def z(s):
            sd = s.std() or 1.0
            return (s - s.mean()) / sd
        scores = z(tfidf_scores) + z(bm25_scores)
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx]